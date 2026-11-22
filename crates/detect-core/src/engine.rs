use crate::bbox::{BBox, Detection};
use crate::kalman::KalmanBBoxFilter;
use crate::tracker::TrackState;

/// Высокоуровневый движок трекинга и сопровождения цели.
/// Объединяет результаты детектора, фильтра Калмана и логику захвата.
#[derive(Debug, Clone)]
pub struct TrackingEngine {
    pub state: TrackState,
    pub target: Option<BBox>,
    pub confidence: f32,
    pub lost_frame_count: u32,
    pub max_lost_frames: u32,
    pub kalman: KalmanBBoxFilter,
}

impl TrackingEngine {
    pub fn new(max_lost_frames: u32) -> Self {
        Self {
            state: TrackState::Searching,
            target: None,
            confidence: 0.0,
            lost_frame_count: 0,
            max_lost_frames,
            kalman: KalmanBBoxFilter::new(),
        }
    }

    /// Ручной захват цели оператором
    pub fn lock_target(&mut self, bbox: BBox) {
        self.kalman.init(&bbox);
        self.target = Some(bbox);
        self.confidence = 1.0;
        self.state = TrackState::Tracking;
        self.lost_frame_count = 0;
    }

    /// Сброс захвата в режим поиска
    pub fn unlock(&mut self) {
        self.state = TrackState::Searching;
        self.target = None;
        self.confidence = 0.0;
        self.lost_frame_count = 0;
        self.kalman.reset();
    }

    /// Шаг обновления на новом кадре по списку детекций от детектора
    /// dt — время с прошлого кадра в секундах
    pub fn update(&mut self, detections: &[Detection], dt: f32) -> (TrackState, Option<BBox>) {
        match self.state {
            TrackState::Searching => {
                // В режиме поиска автоматически захватываем цель с наивысшей уверенностью
                if let Some(best) = detections.iter().max_by(|a, b| {
                    a.confidence
                        .partial_cmp(&b.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    self.lock_target(best.bbox);
                    self.confidence = best.confidence;
                    (self.state, self.target)
                } else {
                    (TrackState::Searching, None)
                }
            }
            TrackState::Tracking | TrackState::Lost => {
                // 1. Прогноз Калмана
                let predicted = self.kalman.predict(dt);

                // 2. Ищем детекцию, наиболее близкую к прогнозируемой позиции (по IoU или дистанции)
                let matched = detections
                    .iter()
                    .filter(|d| d.bbox.iou(&predicted) > 0.15)
                    .max_by(|a, b| {
                        a.confidence
                            .partial_cmp(&b.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some(det) = matched {
                    // Успешное подтверждение целью
                    let smoothed = self.kalman.update(&det.bbox);
                    self.target = Some(smoothed);
                    self.confidence = det.confidence;
                    self.state = TrackState::Tracking;
                    self.lost_frame_count = 0;
                    (self.state, self.target)
                } else {
                    // Окклюзия (цель скрылась за деревом или ландшафтом)
                    self.lost_frame_count += 1;
                    if self.lost_frame_count <= self.max_lost_frames {
                        // Ведем по инерции фильтром Калмана!
                        self.state = TrackState::Lost;
                        self.target = Some(predicted);
                        // Постепенно снижаем уверенность
                        self.confidence *= 0.90;
                        (self.state, self.target)
                    } else {
                        // Полная потеря цели — переход в поиск
                        self.unlock();
                        (TrackState::Searching, None)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_auto_lock_and_track() {
        let mut engine = TrackingEngine::new(5);
        let det = Detection::new(BBox::new(100.0, 100.0, 30.0, 30.0), 0.85, 0);

        let (state, bbox) = engine.update(&[det], 0.033);
        assert_eq!(state, TrackState::Tracking);
        assert!(bbox.is_some());

        // Кадр с окклюзией (пустой список детекций — цель за деревом)
        let (state2, bbox2) = engine.update(&[], 0.033);
        assert_eq!(state2, TrackState::Lost);
        assert!(bbox2.is_some()); // Инерциальный трек работает!
    }
}
