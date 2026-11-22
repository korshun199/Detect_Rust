use detect_core::{BBox, Detection, non_max_suppression};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

/// Асинхронный воркер инференса модели ONNX в фоновом потоке
pub struct OnnxInferenceWorker {
    frame_tx: Sender<Vec<f32>>,
    result_rx: Receiver<Vec<Detection>>,
    child: Option<Child>,
    pub input_size: u32,
}

impl OnnxInferenceWorker {
    pub fn spawn(model_path: &str, input_size: u32) -> Result<Self, String> {
        // Ищем python с установленным onnxruntime
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

        let (frame_tx, frame_rx) = channel::<Vec<f32>>();
        let (result_tx, result_rx) = channel::<Vec<Detection>>();

        // Фоновый поток обмена данными с воркером
        thread::spawn(move || {
            let mut mut_stdin = stdin;
            let mut mut_reader = reader;

            while let Ok(tensor) = frame_rx.recv() {
                // Преобразуем &[f32] в сырые байты
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        tensor.as_ptr() as *const u8,
                        tensor.len() * std::mem::size_of::<f32>(),
                    )
                };

                if mut_stdin.write_all(byte_slice).is_err() || mut_stdin.flush().is_err() {
                    break;
                }

                // Читаем количество найденных объектов (u32 little-endian)
                let mut count_buf = [0u8; 4];
                if mut_reader.read_exact(&mut count_buf).is_err() {
                    break;
                }
                let count = u32::from_le_bytes(count_buf) as usize;

                let mut detections = Vec::with_capacity(count);
                // Каждая детекция: cx, cy, w, h, conf, class_id (4 * 5 + 4 = 24 байта)
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
            frame_tx,
            result_rx,
            child: Some(child),
            input_size,
        })
    }

    /// Отправляет тензор кадра в фоновый поток инференса (если воркер свободен)
    pub fn try_detect(&self, tensor: &[f32]) {
        // Отправляем копию тензора в канал
        let _ = self.frame_tx.send(tensor.to_vec());
    }

    /// Проверяет наличие готовых результатов инференса без блокировки (non-blocking)
    pub fn poll_detections(
        &self,
        orig_width: f32,
        orig_height: f32,
        nms_iou_threshold: f32,
        confidence_threshold: f32,
    ) -> Option<Vec<Detection>> {
        let mut latest_res = None;
        // Забираем самый свежий результат из канала (сливаем промежуточные)
        while let Ok(res) = self.result_rx.try_recv() {
            latest_res = Some(res);
        }

        if let Some(raw_dets) = latest_res {
            let scale_x = orig_width / (self.input_size as f32);
            let scale_y = orig_height / (self.input_size as f32);

            let scaled: Vec<Detection> = raw_dets
                .into_iter()
                .filter(|d| d.confidence >= confidence_threshold)
                .map(|d| {
                    let bbox = BBox::new(
                        d.bbox.cx * scale_x,
                        d.bbox.cy * scale_y,
                        d.bbox.w * scale_x,
                        d.bbox.h * scale_y,
                    );
                    Detection::new(bbox, d.confidence, d.class_id)
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
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
        }
    }
}
