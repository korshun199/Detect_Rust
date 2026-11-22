use crate::bbox::BBox;
use crate::frame::FrameView;

/// Состояние сопровождения цели трекером
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    /// Поиск / ожидание цели
    Searching,
    /// Уверенный захват и сопровождение
    Tracking,
    /// Временная окклюзия / потеря (инерциальное ведение)
    Lost,
}

/// Абстрактный интерфейс алгоритма трекинга (KCF, MOSSE, оптический поток)
pub trait Tracker {
    /// Инициализация трекера по первичному прямоугольнику детектора
    fn init(&mut self, frame: &FrameView, target: BBox);

    /// Обновление положения цели на новом кадре
    /// Возвращает уточненный BBox и оценку качества захвата (0.0..1.0)
    fn update(&mut self, frame: &FrameView) -> (BBox, f32);

    /// Текущее состояние
    fn state(&self) -> TrackState;

    /// Сброс трекера
    fn reset(&mut self);
}
