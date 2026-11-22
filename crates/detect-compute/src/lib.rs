//! Вычислительные бэкенды для инференса моделей и ускорения математики.

pub mod preprocess;
pub mod worker;
pub mod yolo;

pub use preprocess::Letterbox;
pub use worker::OnnxInferenceWorker;
pub use yolo::YoloDetector;
