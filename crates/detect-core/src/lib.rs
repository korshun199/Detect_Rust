//! Алгоритмическое ядро системы распознавания и трекинга Detect_Rust.
//! 
//! Данный модуль не зависит от операционной системы и сторонних фреймворков.
//! Содержит базовые типы кадра, структуры ограничивающих рамок (BBox),
//! алгоритмы подавления немаксимумов (NMS), ПИД-регулирование и интерфейсы трекера.

pub mod bbox;
pub mod filter;
pub mod frame;
pub mod nms;
pub mod tracker;

pub use bbox::{BBox, Detection};
pub use filter::{LowPassIirFilter, PidController};
pub use frame::{FrameFormat, FrameView};
pub use nms::non_max_suppression;
pub use tracker::{TrackState, Tracker};
