use crate::preprocess::LetterboxGeometry;
use detect_core::{BBox, Detection, non_max_suppression};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// Асинхронный воркер инференса модели ONNX в фоновом потоке
pub struct OnnxInferenceWorker {
    pending_frame: Arc<(Mutex<Option<Vec<f32>>>, Condvar)>,
    running: Arc<AtomicBool>,
    result_rx: Receiver<Vec<Detection>>,
    child: Option<Child>,
    pub input_size: u32,
}

impl OnnxInferenceWorker {
    pub fn spawn(model_path: &str, input_size: u32) -> Result<Self, String> {
        let python_paths = [
            "/home/work/VideoT16/.venv/bin/python",
            "python3",
            "python",
        ];

        let mut chosen_python = "python3";
        for p in &python_paths {
            if std::path::Path::new(p).exists() {
                chosen_python = p;
                break;
            }
        }

        let worker_script = "scripts/onnx_worker.py";

        let mut child = Command::new(chosen_python)
            .arg(worker_script)
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Не удалось запустить onnx_worker: {}", e))?;

        let stdin = child.stdin.take().ok_or("Не удалось захватить stdin воркера")?;
        let stdout = child.stdout.take().ok_or("Не удалось захватить stdout воркера")?;

        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| format!("Ошибка чтения READY от воркера: {}", e))?;

        if !line.contains("READY") {
            return Err(format!("Воркер не прислал READY: {}", line));
        }

        // Одноэлементный буфер последнего кадра (политика: LATEST ONLY)
        // Исключает накопление очереди и устраняет отставание рамок от реального объекта
        let pending_frame = Arc::new((Mutex::new(None::<Vec<f32>>), Condvar::new()));
        let pending_clone = pending_frame.clone();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let (result_tx, result_rx) = channel::<Vec<Detection>>();

        thread::spawn(move || {
            let mut mut_stdin = stdin;
            let mut mut_reader = reader;

            while running_clone.load(Ordering::Relaxed) {
                // Извлекаем САМЫЙ СВЕЖИЙ кадр, отбрасывая пропущенные за время прошлого инференса
                let tensor = {
                    let (lock, cvar) = &*pending_clone;
                    let mut slot = lock.lock().unwrap();
                    while slot.is_none() && running_clone.load(Ordering::Relaxed) {
                        slot = cvar.wait(slot).unwrap();
                    }
                    if !running_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    slot.take().unwrap()
                };

                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        tensor.as_ptr() as *const u8,
                        tensor.len() * std::mem::size_of::<f32>(),
                    )
                };

                if mut_stdin.write_all(byte_slice).is_err() || mut_stdin.flush().is_err() {
                    break;
                }

                let mut count_buf = [0u8; 4];
                if mut_reader.read_exact(&mut count_buf).is_err() {
                    break;
                }
                let count = u32::from_le_bytes(count_buf) as usize;

                let mut detections = Vec::with_capacity(count);
                let mut det_buf = [0u8; 24];

                for _ in 0..count {
                    if mut_reader.read_exact(&mut det_buf).is_err() {
                        break;
                    }
                    let cx = f32::from_le_bytes(det_buf[0..4].try_into().unwrap());
                    let cy = f32::from_le_bytes(det_buf[4..8].try_into().unwrap());
                    let w = f32::from_le_bytes(det_buf[8..12].try_into().unwrap());
                    let h = f32::from_le_bytes(det_buf[12..16].try_into().unwrap());
                    let conf = f32::from_le_bytes(det_buf[16..20].try_into().unwrap());
                    let class_id = u32::from_le_bytes(det_buf[20..24].try_into().unwrap());

                    detections.push(Detection::new(BBox::new(cx, cy, w, h), conf, class_id));
                }

                let _ = result_tx.send(detections);
            }
        });

        Ok(Self {
            pending_frame,
            running,
            result_rx,
            child: Some(child),
            input_size,
        })
    }

    pub fn try_detect(&self, tensor: &[f32]) {
        let (lock, cvar) = &*self.pending_frame;
        if let Ok(mut slot) = lock.lock() {
            // Перезаписываем слот: всегда отправляем только самый последний кадр
            *slot = Some(tensor.to_vec());
            cvar.notify_one();
        }
    }

    /// Проверяет наличие готовых результатов инференса и преобразует координаты
    /// с учетом пропорционального Letterbox (вычитание pad_top/pad_left и деление на scale)
    pub fn poll_detections(
        &self,
        geo: LetterboxGeometry,
        orig_width: f32,
        orig_height: f32,
        nms_iou_threshold: f32,
        confidence_threshold: f32,
    ) -> Option<Vec<Detection>> {
        let mut latest_res = None;
        while let Ok(res) = self.result_rx.try_recv() {
            latest_res = Some(res);
        }

        if let Some(raw_dets) = latest_res {
            if geo.scale <= 0.0 {
                return None;
            }

            let scaled: Vec<Detection> = raw_dets
                .into_iter()
                .filter(|d| d.confidence >= confidence_threshold)
                .filter_map(|d| {
                    // Честное обратное проецирование координат из Python detector.py
                    let real_cx = (d.bbox.cx - geo.pad_left) / geo.scale;
                    let real_cy = (d.bbox.cy - geo.pad_top) / geo.scale;
                    let real_w = d.bbox.w / geo.scale;
                    let real_h = d.bbox.h / geo.scale;

                    // Отсекаем координаты, выпавшие на черные/серые поля
                    if real_cx < 0.0 || real_cx > orig_width || real_cy < 0.0 || real_cy > orig_height {
                        return None;
                    }

                    let bbox = BBox::new(real_cx, real_cy, real_w, real_h);
                    Some(Detection::new(bbox, d.confidence, d.class_id))
                })
                .collect();

            Some(non_max_suppression(scaled, nms_iou_threshold, confidence_threshold))
        } else {
            None
        }
    }
}

impl Drop for OnnxInferenceWorker {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let (_, cvar) = &*self.pending_frame;
        cvar.notify_all();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
        }
    }
}
