//! Вычислительные бэкенды для инференса моделей и ускорения математики.

pub mod preprocess;
pub mod yolo;

pub use preprocess::Letterbox;
pub use yolo::YoloDetector;
