use detect_compute::YoloDetector;

fn main() {
    println!("=== Detect_Rust Desktop Tester ===");
    println!("Инициализация алгоритмического ядра и детектора...");

    let detector = YoloDetector::new(0.40, 0.45);
    println!("Детектор готов: порог уверенности={}, вход={}x{}", 
        detector.conf_threshold, detector.input_size, detector.input_size);

    println!("Готов к приему видеопотока из файла или оцифровщика /dev/video4!");
}
