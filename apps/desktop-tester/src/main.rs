use detect_compute::YoloDetector;
use detect_core::{Settings, TrackingEngine};
use std::path::Path;

fn main() {
    println!("=======================================================");
    println!("     Detect_Rust: Скоростное ядро трекинга и OSD       ");
    println!("=======================================================");

    // 1. Загрузка единого источника настроек settings.json
    let settings_path = "settings.json";
    let settings = match Settings::load_or_create(settings_path) {
        Ok(s) => {
            println!("[CONFIG] Настройки успешно загружены из {}", settings_path);
            s
        }
        Err(e) => {
            eprintln!("[CONFIG ERROR] {}; используем настройки по умолчанию", e);
            Settings::default()
        }
    };

    println!("[VIDEO] Источник: {} ({})", settings.video.source_type, settings.video.file_path);
    println!("[DETECTION] Модель: {}, порог={}", settings.detection.model_path, settings.detection.confidence_threshold);
    println!("[TRACKING] Алгоритм: {}, защита от окклюзий={} кадров", settings.tracking.algorithm, settings.tracking.max_lost_frames);

    // 2. Инициализация компонентов ядра
    let _detector = YoloDetector::new(
        settings.detection.confidence_threshold,
        settings.detection.nms_iou_threshold,
    );
    let mut _engine = TrackingEngine::new(settings.tracking.max_lost_frames);

    println!("[CORE] Ядро и фильтр Калмана готовы к обработке кадров!");
    if Path::new(&settings.video.file_path).exists() {
        println!("[STATUS] Тестовый видеофайл обнаружен на диске: {}", settings.video.file_path);
    } else {
        println!("[STATUS] Тестовый файл пока не найден по пути: {}", settings.video.file_path);
    }
}
