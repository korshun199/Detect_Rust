use crate::bbox::BBox;

/// Линейный фильтр Калмана для оценки траектории и скорости движения цели.
/// Вектор состояния: [cx, cy, w, h, vx, vy]
#[derive(Debug, Clone)]
pub struct KalmanBBoxFilter {
    // Вектор состояния
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub vx: f32,
    pub vy: f32,

    // Ковариация ошибки оценки
    p: [f32; 6],

    // Шум процесса
    q_pos: f32,
    q_vel: f32,

    // Шум измерений
    r_pos: f32,

    initialized: bool,
}

impl KalmanBBoxFilter {
    pub fn new() -> Self {
        Self {
            cx: 0.0,
            cy: 0.0,
            w: 0.0,
            h: 0.0,
            vx: 0.0,
            vy: 0.0,
            p: [10.0, 10.0, 10.0, 10.0, 100.0, 100.0],
            q_pos: 1.0,
            q_vel: 10.0,
            r_pos: 4.0,
            initialized: false,
        }
    }

    /// Инициализация фильтра по первичному измерению
    pub fn init(&mut self, bbox: &BBox) {
        self.cx = bbox.cx;
        self.cy = bbox.cy;
        self.w = bbox.w;
        self.h = bbox.h;
        self.vx = 0.0;
        self.vy = 0.0;
        self.p = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0];
        self.initialized = true;
    }

    /// Шаг прогноза (экстраполяция состояния по времени dt)
    pub fn predict(&mut self, dt: f32) -> BBox {
        if !self.initialized {
            return BBox::new(0.0, 0.0, 0.0, 0.0);
        }

        // Экстраполяция координат
        self.cx += self.vx * dt;
        self.cy += self.vy * dt;

        // Обновление ковариации
        self.p[0] += (self.p[4] + self.q_pos) * dt;
        self.p[1] += (self.p[5] + self.q_pos) * dt;
        self.p[4] += self.q_vel * dt;
        self.p[5] += self.q_vel * dt;

        BBox::new(self.cx, self.cy, self.w, self.h)
    }

    /// Шаг коррекции по новому измерению от детектора или трекера
    pub fn update(&mut self, measurement: &BBox) -> BBox {
        if !self.initialized {
            self.init(measurement);
            return *measurement;
        }

        // Коэффициенты Калмана (Калмановское усиление K)
        let k_x = self.p[0] / (self.p[0] + self.r_pos);
        let k_y = self.p[1] / (self.p[1] + self.r_pos);
        let k_w = self.p[2] / (self.p[2] + self.r_pos);
        let k_h = self.p[3] / (self.p[3] + self.r_pos);

        // Ошибка измерения
        let err_x = measurement.cx - self.cx;
        let err_y = measurement.cy - self.cy;

        // Коррекция координат
        self.cx += k_x * err_x;
        self.cy += k_y * err_y;
        self.w += k_w * (measurement.w - self.w);
        self.h += k_h * (measurement.h - self.h);

        // Коррекция скорости (примерная производная с калмановским сглаживанием)
        self.vx += (k_x * 0.5) * err_x;
        self.vy += (k_y * 0.5) * err_y;

        // Обновление ковариации
        self.p[0] *= 1.0 - k_x;
        self.p[1] *= 1.0 - k_y;
        self.p[2] *= 1.0 - k_w;
        self.p[3] *= 1.0 - k_h;

        BBox::new(self.cx, self.cy, self.w, self.h)
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.vx = 0.0;
        self.vy = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_prediction() {
        let mut kf = KalmanBBoxFilter::new();
        let b1 = BBox::new(100.0, 100.0, 30.0, 30.0);
        kf.init(&b1);

        // Зададим фиктивную скорость
        kf.vx = 20.0;
        kf.vy = 10.0;

        let predicted = kf.predict(0.1);
        assert!((predicted.cx - 102.0).abs() < 1e-3);
        assert!((predicted.cy - 101.0).abs() < 1e-3);
    }
}
