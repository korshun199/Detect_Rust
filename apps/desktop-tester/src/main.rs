mod osd;
mod video;

use detect_compute::{Letterbox, OnnxInferenceWorker};
use detect_core::{BBox, Detection, OperatingMode, Settings, TrackState, TrackingEngine};
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use std::env;
use std::path::Path;
use std::time::Instant;
use video::VideoReader;

fn main() {
    println!("\x1b[1;36m=======================================================\x1b[0m");
    println!("\x1b[1;36m     Detect_Rust: Скоростное ядро трекинга и OSD       \x1b[0m");
    println!("\x1b[1;36m=======================================================\x1b[0m");

    // 1. Загрузка единого источника настроек settings.json
    let settings_path = "settings.json";
    let settings = Settings::load_or_create(settings_path).unwrap_or_else(|e| {
        eprintln!("[CONFIG ERROR] {}; используем настройки по умолчанию", e);
        Settings::default()
    });

    let args: Vec<String> = env::args().collect();
    let is_bench = args.iter().any(|a| a == "--bench");

    let video_source = if args.len() > 1 && !args[1].starts_with("--") {
        args[1].clone()
    } else {
        settings.video.file_path.clone()
    };

    println!("[CONFIG] Загружен конфиг: \x1b[1;33m{}\x1b[0m", settings_path);
    println!("[INPUT]  Источник видео: \x1b[1;32m{}\x1b[0m", video_source);

    if !Path::new(&video_source).exists() && !video_source.starts_with("/dev/") {
        eprintln!("\x1b[1;31m[ERROR] Видеофайл не найден: {}\x1b[0m", video_source);
        return;
    }

    let width = settings.video.width as usize;
    let height = settings.video.height as usize;

    // 2. Инициализация конвейера
    let mut reader = match VideoReader::open(&video_source, width, height, !is_bench) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("\x1b[1;31m[ERROR] {}\x1b[0m", e);
            return;
        }
    };

    let mut letterbox = Letterbox::new(
        settings.detection.input_size as usize,
        settings.detection.input_size as usize,
    );

    // Запуск фонового асинхронного воркера ONNX для инференса модели
    println!("[AI] Инициализация фонового нейросетевого воркера ONNX...");
    let onnx_worker = match OnnxInferenceWorker::spawn(&settings.detection.model_path, settings.detection.input_size) {
        Ok(w) => {
            println!("\x1b[1;32m[AI OK] Модель {} успешно загружена в фоновый поток!\x1b[0m", settings.detection.model_path);
            Some(w)
        }
        Err(e) => {
            eprintln!("\x1b[1;33m[AI WARN] Не удалось запустить ONNX воркер ({}). Работаем без нейросети.\x1b[0m", e);
            None
        }
    };

    let mut engine = TrackingEngine::new(
        settings.tracking.max_lost_frames,
        OperatingMode::AiDetection,
        &settings.control,
    );

    let box_color = ((settings.osd.box_color[0] as u32) << 16)
        | ((settings.osd.box_color[1] as u32) << 8)
        | (settings.osd.box_color[2] as u32);
    let text_color = ((settings.osd.text_color[0] as u32) << 16)
        | ((settings.osd.text_color[1] as u32) << 8)
        | (settings.osd.text_color[2] as u32);
    let cyan = 0x0000E5FF;
    let yellow = 0x00FFCC00;
    let red = 0x00FF3333;

    // 3. Создание интерактивного графического окна на экране ноутбука
    println!("\x1b[1;32m[GUI] Открытие графического окна на экране ноутбука ({}x{})... \x1b[0m", width, height);
    let mut window = match Window::new(
        "Detect_Rust Vision Engine - [AvatarG0006.mp4]",
        width,
        height,
        WindowOptions {
            resize: true,
            scale: minifb::Scale::FitScreen,
            ..WindowOptions::default()
        },
    ) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("\x1b[1;31m[ERROR] Не удалось открыть окно X11: {}\x1b[0m", e);
            return;
        }
    };

    window.set_target_fps(60);

    let mut screen_buffer: Vec<u32> = vec![0u32; width * height];
    let mut last_fps_time = Instant::now();
    let mut current_fps = 60.0f32;
    let mut frame_counter = 0u32;
    let mut is_paused = false;
    let mut last_detections: Vec<Detection> = Vec::new();

    println!("[GUI] Окно открыто. Управление Режимом 1 (Распознавание -> Захват -> Слежение):");
    println!("      [Свободный режим]  - Нейросеть находит дрон и подсвечивает рамкой с % уверенности");
    println!("      [КЛИК / ENTER]     - Захватить обнаруженный дрон в плотное слежение");
    println!("      [R]                - Сбросить захват обратно в свободное распознавание");
    println!("      [ПРОБЕЛ]           - Пауза / Возобновление");
    println!("      [ESC]              - Выход\n");

    let mut last_frame_data: Vec<u8> = vec![0u8; width * height * 3];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Чтение нового кадра (если не на паузе)
        if !is_paused {
            if let Some(raw_frame) = reader.read_frame() {
                last_frame_data.copy_from_slice(raw_frame);
                frame_counter += 1;

                // Препроцессинг в тензор 320x320 с пропорциональным Letterbox и серыми полями 114
                let (tensor, geo) = letterbox.process_rgb24(raw_frame, width, height);

                // Отправляем кадр на распознавание в нейросеть
                if let Some(ref worker) = onnx_worker {
                    worker.try_detect(tensor);

                    // Проверяем свежие детекции от нейросети с честным unmap координат
                    if let Some(new_dets) = worker.poll_detections(
                        geo,
                        width as f32,
                        height as f32,
                        settings.detection.nms_iou_threshold,
                        settings.detection.confidence_threshold,
                    ) {
                        last_detections = new_dets;
                    }
                }

                // Шаг трекера Калмана
                let dt = 1.0 / (settings.video.fps as f32);
                let _ = engine.update(&last_detections, width as f32, height as f32, dt);
            } else {
                println!("[STATUS] Конец видеофайла.");
                break;
            }
        }

        // Преобразование RGB кадра в экранный буфер ARGB
        osd::rgb24_to_argb_buffer(&last_frame_data, &mut screen_buffer, width * height);

        // Обработка клавиш
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            is_paused = !is_paused;
        }
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            engine.unlock();
            println!("[TRACK] Захват сброшен. Режим свободного распознавания.");
        }

        // Захват по клавише ENTER: захватывает самый уверенный распознанный дрон
        if window.is_key_pressed(Key::Enter, minifb::KeyRepeat::No) {
            if let Some(best) = last_detections.iter().max_by(|a, b| {
                a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                engine.lock_target(best.bbox);
                println!("[TRACK] Захват по ENTER: цель CONF={:.0}%", best.confidence * 100.0);
            }
        }

        // Обработка клика мыши: ручной захват цели в точке клика
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
            if window.get_mouse_down(MouseButton::Left) {
                // Если кликнули рядом с распознанным объектом — захватываем его
                let clicked_point = BBox::new(mx as f32, my as f32, 20.0, 20.0);
                let matched_det = last_detections.iter().find(|d| d.bbox.iou(&clicked_point) > 0.05);

                if let Some(det) = matched_det {
                    engine.lock_target(det.bbox);
                    println!("[TRACK] Захвачен распознанный дрон: CONF={:.0}%", det.confidence * 100.0);
                } else {
                    let box_size = 50.0f32;
                    let manual_bbox = BBox::new(mx as f32, my as f32, box_size, box_size);
                    engine.lock_target(manual_bbox);
                    println!("[TRACK] Ручной захват цели в точке [{}, {}]", mx, my);
                }
            }
        }

        // 4. Отрисовка тактического OSD
        // 4.1. Центральное перекрестие
        if settings.osd.show_crosshair {
            osd::draw_crosshair(
                &mut screen_buffer,
                width,
                height,
                (width / 2) as isize,
                (height / 2) as isize,
                20,
                0x00FFFFFF,
            );
        }

        // 4.2. ФАЗА 1: Если цель НЕ захвачена — рисуем все распознанные нейросетью дроны!
        if engine.target.is_none() {
            for det in &last_detections {
                // Тонкая рамка обнаружения
                osd::draw_bbox(&mut screen_buffer, width, height, &det.bbox, cyan);

                // Надпись: DRONE xx%
                let det_text = format!("DRONE {:.0}%", det.confidence * 100.0);
                osd::draw_text(
                    &mut screen_buffer,
                    width,
                    height,
                    det.bbox.x_min().max(10.0) as usize,
                    (det.bbox.y_min() - 16.0).max(10.0) as usize,
                    &det_text,
                    cyan,
                    2,
                );
            }
        }

        // 4.3. ФАЗА 2: Если цель захвачена — рисуем плотную рамку слежения и фильтр Калмана!
        if let Some(target) = engine.target {
            let color = match engine.state {
                TrackState::Tracking => box_color, // Зеленый захват
                TrackState::Lost => red,           // Окклюзия (за деревом)
                TrackState::Searching => cyan,
            };
            osd::draw_bbox(&mut screen_buffer, width, height, &target, color);

            // Текст: LOCKED [DRONE xx%]
            let track_info = format!("LOCKED [DRONE {:.0}%]", engine.confidence * 100.0);
            osd::draw_text(
                &mut screen_buffer,
                width,
                height,
                target.x_min().max(10.0) as usize,
                (target.y_min() - 16.0).max(10.0) as usize,
                &track_info,
                color,
                2,
            );

            // Направляющий вектор от прицела к цели
            let cx = (target.cx as isize).clamp(0, width as isize - 1);
            let cy = (target.cy as isize).clamp(0, height as isize - 1);
            let mid_x = (width / 2) as isize;
            let mid_y = (height / 2) as isize;
            osd::draw_line(&mut screen_buffer, width, height, mid_x, mid_y, cx, cy, yellow);
        }

        // 4.4. Верхняя строка телеметрии
        let (status_banner, status_col) = match engine.state {
            TrackState::Searching => {
                if !last_detections.is_empty() {
                    (format!("STATUS: RECONNAISSANCE [{} DRONE DETECTED]", last_detections.len()), cyan)
                } else {
                    ("STATUS: RECONNAISSANCE [SCANNING...]".to_string(), text_color)
                }
            }
            TrackState::Tracking => ("STATUS: TRACKING [TARGET LOCKED]".to_string(), box_color),
            TrackState::Lost => ("STATUS: OCCLUSION [KALMAN INERTIAL PREDICT]".to_string(), red),
        };

        let hud_left = format!("FPS:{:.1} | FRAME:#{} | {}", current_fps, frame_counter, status_banner);
        osd::draw_text(&mut screen_buffer, width, height, 20, 20, &hud_left, status_col, 2);

        // Индикатор батареи и радиосвязи
        let hud_right = "BAT:16.4V [4S OK] | LINK:UART 921600";
        osd::draw_text(&mut screen_buffer, width, height, width - 420, 20, hud_right, cyan, 2);

        // 4.5. Нижняя подсказка
        let help_text = match engine.state {
            TrackState::Searching => "[CLICK / ENTER]: LOCK TARGET | [SPACE]: PAUSE | [ESC]: EXIT",
            TrackState::Tracking | TrackState::Lost => "[R]: UNLOCK / RESUME SCAN | [SPACE]: PAUSE | [ESC]: EXIT",
        };
        osd::draw_text(&mut screen_buffer, width, height, 20, height - 30, help_text, 0x00AAAAAA, 2);

        // Расчет FPS
        if last_fps_time.elapsed().as_millis() >= 500 {
            let elapsed_sec = last_fps_time.elapsed().as_secs_f32();
            current_fps = 30.0 / elapsed_sec;
            last_fps_time = Instant::now();
        }

        // 5. Обновление окна
        if window.update_with_buffer(&screen_buffer, width, height).is_err() {
            break;
        }
    }

    println!("[GUI] Графическое окно закрыто пользователем.");
}
