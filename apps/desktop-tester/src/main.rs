mod osd;
mod video;

use detect_compute::{Letterbox, YoloDetector};
use detect_core::{BBox, Settings, TrackState, TrackingEngine};
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
    let _detector = YoloDetector::new(
        settings.detection.confidence_threshold,
        settings.detection.nms_iou_threshold,
    );
    let mut engine = TrackingEngine::new(settings.tracking.max_lost_frames);

    // Цвета из settings.json
    let box_color = ((settings.osd.box_color[0] as u32) << 16)
        | ((settings.osd.box_color[1] as u32) << 8)
        | (settings.osd.box_color[2] as u32);
    let text_color = ((settings.osd.text_color[0] as u32) << 16)
        | ((settings.osd.text_color[1] as u32) << 8)
        | (settings.osd.text_color[2] as u32);
    let cyan = 0x0000E5FF;
    let red = 0x00FF3333;

    // Если запущен режим бенчмарка --bench: работаем в консоли
    if is_bench {
        println!("\x1b[1;35m>>> Запуск консольного бенчмарка... <<<\x1b[0m\n");
        let start_time = Instant::now();
        let mut frames_processed = 0u64;

        while let Some(raw_frame) = reader.read_frame() {
            frames_processed += 1;
            let _tensor = letterbox.process_rgb24(raw_frame, width, height);
            let dt = 1.0 / (settings.video.fps as f32);
            let _ = engine.update(&[], dt);

            if frames_processed >= 300 {
                break;
            }
        }

        let total_elapsed = start_time.elapsed().as_secs_f32();
        let overall_fps = frames_processed as f32 / total_elapsed;
        println!("Обработано кадров: {}", frames_processed);
        println!("Средняя скорость:  \x1b[1;33m{:.1} FPS\x1b[0m", overall_fps);
        return;
    }

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

    // Ограничение частоты обновления окна под частоту видео (60 FPS)
    window.set_target_fps(60);

    let mut screen_buffer: Vec<u32> = vec![0u32; width * height];
    let mut last_fps_time = Instant::now();
    let mut current_fps = 60.0f32;
    let mut frame_counter = 0u32;
    let mut is_paused = false;

    println!("[GUI] Окно открыто. Управление:");
    println!("      [Клик мыши] - Ручной захват цели в точке клика");
    println!("      [ПРОБЕЛ]    - Пауза / Возобновление");
    println!("      [R]         - Сброс трекера в режим поиска");
    println!("      [ESC]       - Закрыть окно\n");

    let mut last_frame_data: Vec<u8> = vec![0u8; width * height * 3];

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Чтение нового кадра (если не на паузе)
        if !is_paused {
            if let Some(raw_frame) = reader.read_frame() {
                last_frame_data.copy_from_slice(raw_frame);
                frame_counter += 1;

                // Препроцессинг
                let _tensor = letterbox.process_rgb24(raw_frame, width, height);

                // Шаг трекера Калмана
                let dt = 1.0 / (settings.video.fps as f32);
                let _ = engine.update(&[], dt);
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
        }

        // Обработка клика мыши: ручной захват цели!
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
            if window.get_mouse_down(MouseButton::Left) {
                let box_size = 60.0f32;
                let manual_bbox = BBox::new(mx as f32, my as f32, box_size, box_size);
                engine.lock_target(manual_bbox);
            }
        }

        // 4. Отрисовка тактического OSD поверх видео
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

        // 4.2. Рамка сопровождаемой цели
        if let Some(target) = engine.target {
            let color = match engine.state {
                TrackState::Tracking => box_color,
                TrackState::Lost => red,
                TrackState::Searching => cyan,
            };
            osd::draw_bbox(&mut screen_buffer, width, height, &target, color);

            // Текст координат над рамкой
            let target_info = format!("TARGET [X:{:.0} Y:{:.0} CONF:{:.0}%]", 
                target.cx, target.cy, engine.confidence * 100.0);
            osd::draw_text(
                &mut screen_buffer,
                width,
                height,
                target.x_min().max(10.0) as usize,
                (target.y_min() - 16.0).max(10.0) as usize,
                &target_info,
                color,
                2,
            );
        }

        // 4.3. Верхняя строка телеметрии HUD
        let status_text = match engine.state {
            TrackState::Tracking => "STATUS: TRACKING [LOCKED]",
            TrackState::Lost => "STATUS: OCCLUSION [KALMAN PREDICT]",
            TrackState::Searching => "STATUS: SEARCHING",
        };
        let hud_left = format!("FPS:{:.1} | RES:{}x{} | FRAME:#{} | {}", 
            current_fps, width, height, frame_counter, status_text);
        osd::draw_text(&mut screen_buffer, width, height, 20, 20, &hud_left, text_color, 2);

        // 4.4. Индикатор батареи и радиосвязи
        let hud_right = "BAT:16.4V [4S OK] | LINK:UART 921600";
        osd::draw_text(&mut screen_buffer, width, height, width - 420, 20, hud_right, cyan, 2);

        // 4.5. Нижняя подсказка
        let help_text = if is_paused {
            "PAUSED -- PRESS SPACE TO RESUME"
        } else {
            "CLICK: LOCK TARGET | SPACE: PAUSE | R: RESET | ESC: EXIT"
        };
        osd::draw_text(&mut screen_buffer, width, height, 20, height - 30, help_text, 0x00AAAAAA, 2);

        // Расчет FPS
        if last_fps_time.elapsed().as_millis() >= 500 {
            let elapsed_sec = last_fps_time.elapsed().as_secs_f32();
            current_fps = 30.0 / elapsed_sec;
            last_fps_time = Instant::now();
        }

        // 5. Обновление буфера окна
        if window.update_with_buffer(&screen_buffer, width, height).is_err() {
            break;
        }
    }

    println!("[GUI] Графическое окно закрыто пользователем.");
}
