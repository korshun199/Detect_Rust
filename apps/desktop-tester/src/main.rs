mod video;

use detect_compute::{Letterbox, YoloDetector};
use detect_core::{FrameFormat, FrameView, Settings, TrackingEngine};
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
    let video_source = if args.len() > 1 && !args[1].starts_with("--") {
        args[1].clone()
    } else {
        settings.video.file_path.clone()
    };

    let is_bench = args.iter().any(|a| a == "--bench");

    println!("[CONFIG] Загружен конфиг: \x1b[1;33m{}\x1b[0m", settings_path);
    println!("[INPUT]  Источник видео: \x1b[1;32m{}\x1b[0m (Режим: {})", 
        video_source, if is_bench { "Максимальный бенчмарк" } else { "Реальное время (60 FPS)" });

    if !Path::new(&video_source).exists() && !video_source.starts_with("/dev/") {
        eprintln!("\x1b[1;31m[ERROR] Видеофайл не найден: {}\x1b[0m", video_source);
        return;
    }

    // 2. Инициализация конвейера
    let width = settings.video.width as usize;
    let height = settings.video.height as usize;

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

    println!("[INIT]   Конвейер видеозахвата и математическое ядро готовы!");
    println!("\x1b[1;35m>>> Запуск обработки видеопотока... <<<\x1b[0m\n");

    let start_time = Instant::now();
    let mut last_stat_time = Instant::now();
    let mut frames_processed = 0u64;
    let mut total_preprocess_micros = 0u128;

    // 3. Основной горячий цикл обработки кадров
    while let Some(raw_frame) = reader.read_frame() {
        frames_processed += 1;
        let t0 = Instant::now();

        // 3.1. Zero-copy представление кадра
        let _frame_view = FrameView::new(
            raw_frame,
            width as u32,
            height as u32,
            (width * 3) as u32,
            FrameFormat::Rgb8,
        );

        // 3.2. Высокоскоростной ресайз и нормализация в тензор 320x320
        let _tensor = letterbox.process_rgb24(raw_frame, width, height);
        let preprocess_elapsed = t0.elapsed().as_micros();
        total_preprocess_micros += preprocess_elapsed;

        // 3.3. Шаг трекера и фильтра Калмана (интервал кадра dt ~ 1/60 c)
        let dt = 1.0 / (settings.video.fps as f32);
        let (state, target) = engine.update(&[], dt);

        // 3.4. Вывод статистики каждые 60 кадров (примерно раз в секунду при 60 FPS)
        if frames_processed % 60 == 0 {
            let elapsed_sec = last_stat_time.elapsed().as_secs_f32();
            let current_fps = 60.0 / elapsed_sec;
            let avg_preprocess_us = total_preprocess_micros / frames_processed as u128;

            let target_str = match target {
                Some(b) => format!("[x={:.1}, y={:.1}, w={:.1}, h={:.1}]", b.cx, b.cy, b.w, b.h),
                None => "Поиск...".to_string(),
            };

            print!("\r\x1b[KКадр: \x1b[1;32m#{:<5}\x1b[0m | Скорость: \x1b[1;33m{:.1} FPS\x1b[0m | Препроцессинг: \x1b[1;34m{} мкс\x1b[0m | Статус: \x1b[1;36m{:?}\x1b[0m | Цель: {}",
                frames_processed, current_fps, avg_preprocess_us, state, target_str);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            last_stat_time = Instant::now();
        }

        // Для быстрого теста в режиме бенчмарка ограничимся первыми 300 кадрами
        if is_bench && frames_processed >= 300 {
            break;
        }
    }

    let total_elapsed = start_time.elapsed().as_secs_f32();
    let overall_fps = frames_processed as f32 / total_elapsed;
    println!("\n\n\x1b[1;32m=== ИТОГ ОБРАБОТКИ ВИДЕО ===\x1b[0m");
    println!("Обработано кадров: {}", frames_processed);
    println!("Затраченное время: {:.2} с", total_elapsed);
    println!("Средняя скорость:  \x1b[1;33m{:.1} FPS\x1b[0m", overall_fps);
    if frames_processed > 0 {
        println!("Средний ресайз 1280x720 -> 320x320: \x1b[1;34m{} мкс\x1b[0m ({:.2} мс)", 
            total_preprocess_micros / frames_processed as u128,
            (total_preprocess_micros as f32 / frames_processed as f32) / 1000.0);
    }
}
