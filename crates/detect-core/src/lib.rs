//! Алгоритмическое ядро системы распознавания и трекинга Detect_Rust.
//! 
//! Данный модуль не зависит от операционной системы и сторонних графических фреймворков.
//! Содержит базовые типы кадра, структуры рамок (BBox), единые настройки (settings.json),
//! фильтр Калмана для инерциального сопровождения при окклюзиях и движок трекинга
//! с расчетом команд управления полетом (ПИД-руление).

pub mod bbox;
pub mod engine;
pub mod filter;
pub mod frame;
pub mod kalman;
pub mod nms;
pub mod settings;
pub mod tracker;

pub use bbox::{BBox, Detection};
pub use engine::{FlightControlCommand, TrackingEngine};
pub use filter::{LowPassIirFilter, PidController};
pub use frame::{FrameFormat, FrameView};
pub use kalman::KalmanBBoxFilter;
pub use nms::non_max_suppression;
pub use settings::{ControlConfig, OperatingMode, Settings};
pub use tracker::{TrackState, Tracker};
