use crate::bbox::{BBox, Detection};
use crate::filter::PidController;
use crate::kalman::KalmanBBoxFilter;
use crate::settings::{ControlConfig, OperatingMode};
use crate::tracker::TrackState;

/// Команда управления приводами дрона (рулями/моторами через STM32)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightControlCommand {
    /// Угловая скорость рыскания (градусы/сек)
    pub yaw_rate: f32,
    /// Угловая скорость тангажа (градусы/сек)
    pub pitch_rate: f32,
    /// Флаг активности наведения (true — цель удерживается, рули активны)
    pub active: bool,
}

impl Default for FlightControlCommand {
    fn default() -> Self {
        Self {
            yaw_rate: 0.0,
            pitch_rate: 0.0,
            active: false,
        }
    }
}

/// Высокоуровневый движок трекинга и сопровождения цели.
/// Объединяет результаты детектора, фильтра Калмана и логику захвата.
#[derive(Debug, Clone)]
pub struct TrackingEngine {
    pub mode: OperatingMode,
    pub state: TrackState,
    pub target: Option<BBox>,
    pub confidence: f32,
    pub lost_frame_count: u32,
    pub max_lost_frames: u32,
    pub kalman: KalmanBBoxFilter,
    pub pid_yaw: PidController,
    pub pid_pitch: PidController,
    pub last_command: FlightControlCommand,
}

impl TrackingEngine {
    pub fn new(max_lost_frames: u32, mode: OperatingMode, control: &ControlConfig) -> Self {
        Self {
            mode,
            state: TrackState::Searching,
            target: None,
            confidence: 0.0,
            lost_frame_count: 0,
            max_lost_frames,
            kalman: KalmanBBoxFilter::new(),
            pid_yaw: PidController::new(control.kp_yaw, control.ki_yaw, control.kd_yaw, 50.0, 100.0),
            pid_pitch: PidController::new(control.kp_pitch, control.ki_pitch, control.kd_pitch, 50.0, 100.0),
            last_command: FlightControlCommand::default(),
        }
    }

    /// Переключение режима работы (Нейросеть / Оптический трекер)
    pub fn set_mode(&mut self, mode: OperatingMode) {
        self.mode = mode;
        if mode == OperatingMode::OpticalGuidance {
            // В режиме доводки сбрасываем интеграторы ПИД
            self.pid_yaw.reset();
            self.pid_pitch.reset();
        }
    }

    /// Ручной захват цели оператором (или перехватчиком)
    pub fn lock_target(&mut self, bbox: BBox) {
        self.kalman.init(&bbox);
        self.target = Some(bbox);
        self.confidence = 1.0;
        self.state = TrackState::Tracking;
        self.lost_frame_count = 0;
        self.pid_yaw.reset();
        self.pid_pitch.reset();
    }

    /// Сброс захвата в режим поиска
    pub fn unlock(&mut self) {
        self.state = TrackState::Searching;
        self.target = None;
        self.confidence = 0.0;
        self.lost_frame_count = 0;
        self.kalman.reset();
        self.pid_yaw.reset();
        self.pid_pitch.reset();
        self.last_command = FlightControlCommand::default();
    }

    /// Шаг обновления на новом кадре
    /// screen_w, screen_h — разрешение экрана/кадра (для центровки прицела)
    /// dt — время с прошлого кадра в секундах
    pub fn update(
        &mut self,
        detections: &[Detection],
        screen_w: f32,
        screen_h: f32,
        dt: f32,
    ) -> (TrackState, Option<BBox>, FlightControlCommand) {
        let (new_state, new_target) = match self.mode {
            // РЕЖИМ 1: НЕЙРОСЕТЕВОЙ (Поиск, распознавание и ручной захват по кнопке)
            OperatingMode::AiDetection => {
                match self.state {
                    TrackState::Searching => {
                        // В свободном режиме распознавания цель отображается рамкой обнаружения,
                        // захват включается оператором по ENTER или клику
                        (TrackState::Searching, None)
                    }
                    TrackState::Tracking | TrackState::Lost => {
                        self.track_with_kalman(detections, dt)
                    }
                }
            }
            // РЕЖИМ 2: ОПТИЧЕСКИЙ ТРЕКЕР (Без нейронки: удержание запомненного объекта Калманом)
            OperatingMode::OpticalGuidance => {
                match self.state {
                    TrackState::Searching => (TrackState::Searching, None),
                    TrackState::Tracking | TrackState::Lost => {
                        self.track_with_kalman(detections, dt)
                    }
                }
            }
        };

        // Расчет команд руления ПИД-регулятором
        let command = if let Some(target) = new_target {
            let center_x = screen_w * 0.5;
            let center_y = screen_h * 0.5;

            // Ошибка отклонения цели от центра экрана в пикселях
            let error_x = target.cx - center_x;
            let error_y = target.cy - center_y;

            let yaw_rate = self.pid_yaw.compute(error_x, dt);
            let pitch_rate = self.pid_pitch.compute(error_y, dt);

            FlightControlCommand {
                yaw_rate,
                pitch_rate,
                active: true,
            }
        } else {
            FlightControlCommand::default()
        };

        self.last_command = command;
        (new_state, new_target, command)
    }

    /// Вспомогательное ведение цели фильтром Калмана
    fn track_with_kalman(&mut self, detections: &[Detection], dt: f32) -> (TrackState, Option<BBox>) {
        let predicted = self.kalman.predict(dt);

        let matched = detections
            .iter()
            .filter(|d| d.bbox.iou(&predicted) > 0.15)
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(det) = matched {
            let smoothed = self.kalman.update(&det.bbox);
            self.target = Some(smoothed);
            self.confidence = det.confidence;
            self.state = TrackState::Tracking;
            self.lost_frame_count = 0;
            (self.state, self.target)
        } else if self.mode == OperatingMode::OpticalGuidance && self.state == TrackState::Tracking {
            // В автономном режиме трекер плавно держит объект даже без детекций
            let smoothed = self.kalman.update(&predicted);
            self.target = Some(smoothed);
            (TrackState::Tracking, self.target)
        } else {
            self.lost_frame_count += 1;
            if self.lost_frame_count <= self.max_lost_frames {
                self.state = TrackState::Lost;
                self.target = Some(predicted);
                self.confidence *= 0.90;
                (self.state, self.target)
            } else {
                self.unlock();
                (TrackState::Searching, None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_modes_and_pid() {
        let control = ControlConfig::default();
        let mut engine = TrackingEngine::new(5, OperatingMode::OpticalGuidance, &control);

        // Ручной захват со смещением вправо и вниз (+50, +30)
        let manual_target = BBox::new(690.0, 390.0, 40.0, 40.0);
        engine.lock_target(manual_target);

        let (state, target, cmd) = engine.update(&[], 1280.0, 720.0, 0.033);
        assert_eq!(state, TrackState::Tracking);
        assert!(target.is_some());
        assert!(cmd.active);
        assert!(cmd.yaw_rate > 0.0); // Коррекция вправо
        assert!(cmd.pitch_rate > 0.0); // Коррекция вниз
    }
}
