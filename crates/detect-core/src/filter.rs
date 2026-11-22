/// Цифровой IIR-фильтр нижних частот 1-го порядка (сглаживание шума детекций)
#[derive(Debug, Clone, Copy)]
pub struct LowPassIirFilter {
    alpha: f32,
    prev_output: f32,
    initialized: bool,
}

impl LowPassIirFilter {
    /// Создает фильтр с коэффициентом сглаживания alpha (0.0 < alpha <= 1.0)
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 1.0),
            prev_output: 0.0,
            initialized: false,
        }
    }

    /// Пропускает очередное значение через фильтр
    pub fn update(&mut self, input: f32) -> f32 {
        if !self.initialized {
            self.prev_output = input;
            self.initialized = true;
            return input;
        }

        let output = self.alpha * input + (1.0 - self.alpha) * self.prev_output;
        self.prev_output = output;
        output
    }

    /// Сброс состояния
    pub fn reset(&mut self) {
        self.initialized = false;
        self.prev_output = 0.0;
    }
}

/// ПИД-регулятор с защитой от интегрального насыщения (Anti-windup)
#[derive(Debug, Clone, Copy)]
pub struct PidController {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub max_integral: f32,
    pub max_output: f32,

    integral: f32,
    prev_error: f32,
    initialized: bool,
}

impl PidController {
    pub fn new(kp: f32, ki: f32, kd: f32, max_integral: f32, max_output: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            max_integral,
            max_output,
            integral: 0.0,
            prev_error: 0.0,
            initialized: false,
        }
    }

    /// Рассчитывает управляющее воздействие по ошибке рассогласования (error)
    /// dt — интервал времени в секундах между кадрами
    pub fn compute(&mut self, error: f32, dt: f32) -> f32 {
        if dt <= 0.0 {
            return 0.0;
        }

        // Пропорциональная составляющая
        let p = self.kp * error;

        // Интегральная составляющая с ограничением насыщения
        self.integral += error * dt;
        self.integral = self.integral.clamp(-self.max_integral, self.max_integral);
        let i = self.ki * self.integral;

        // Дифференциальная составляющая
        let d = if self.initialized {
            self.kd * (error - self.prev_error) / dt
        } else {
            self.initialized = true;
            0.0
        };

        self.prev_error = error;

        (p + i + d).clamp(-self.max_output, self.max_output)
    }

    /// Сброс интегратора
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iir_filter() {
        let mut filter = LowPassIirFilter::new(0.5);
        assert_eq!(filter.update(10.0), 10.0);
        let v2 = filter.update(20.0);
        assert!((v2 - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_pid_controller() {
        let mut pid = PidController::new(1.0, 0.1, 0.05, 50.0, 100.0);
        let out = pid.compute(10.0, 0.05);
        assert!(out > 0.0);
    }
}
