/// Метаданные пропорционального масштабирования (Letterbox)
#[derive(Debug, Clone, Copy)]
pub struct LetterboxGeometry {
    pub scale: f32,
    pub pad_left: f32,
    pub pad_top: f32,
    pub resized_w: usize,
    pub resized_h: usize,
}

/// Высокоскоростной пропорциональный ресайз кадра в тензор [1, 3, target_h, target_w]
/// с заполнением полей нейтральным серым цветом 114 (YOLO format)
pub struct Letterbox {
    pub target_w: usize,
    pub target_h: usize,
    pub tensor_buffer: Vec<f32>,
    pub last_geometry: LetterboxGeometry,
}

impl Letterbox {
    pub fn new(target_w: usize, target_h: usize) -> Self {
        let size = 1 * 3 * target_h * target_w;
        // Заполняем серым 114.0 / 255.0 = 0.44705883
        let gray_val = 114.0f32 / 255.0f32;
        Self {
            target_w,
            target_h,
            tensor_buffer: vec![gray_val; size],
            last_geometry: LetterboxGeometry {
                scale: 1.0,
                pad_left: 0.0,
                pad_top: 0.0,
                resized_w: target_w,
                resized_h: target_h,
            },
        }
    }

    /// Пропорционально масштабирует изображение (src_w x src_h) в центр тензора target_w x target_h,
    /// сохраняя истинную форму объекта (aspect ratio) и заполняя поля серым цветом 114.
    pub fn process_rgb24(&mut self, src: &[u8], src_w: usize, src_h: usize) -> (&[f32], LetterboxGeometry) {
        let tw = self.target_w;
        let th = self.target_h;
        let channel_stride = tw * th;

        // 1. Вычисляем точный коэффициент масштабирования без искажения пропорций
        let scale = (tw as f32 / src_w as f32).min(th as f32 / src_h as f32);
        let resized_w = ((src_w as f32 * scale).round() as usize).max(1).min(tw);
        let resized_h = ((src_h as f32 * scale).round() as usize).max(1).min(th);

        let pad_left = (tw - resized_w) / 2;
        let pad_top = (th - resized_h) / 2;

        self.last_geometry = LetterboxGeometry {
            scale,
            pad_left: pad_left as f32,
            pad_top: pad_top as f32,
            resized_w,
            resized_h,
        };

        let gray_val = 114.0f32 / 255.0f32;
        let norm = 1.0f32 / 255.0f32;

        // Заливаем фоновым серым цветом 114
        self.tensor_buffer.fill(gray_val);

        let inv_scale_x = src_w as f32 / resized_w as f32;
        let inv_scale_y = src_h as f32 / resized_h as f32;

        // 2. Вписываем несжатое изображение строго по центру
        for ry in 0..resized_h {
            let sy = ((ry as f32 * inv_scale_y) as usize).min(src_h - 1);
            let src_row = sy * src_w * 3;
            let dst_y = pad_top + ry;
            let dst_row = dst_y * tw;

            for rx in 0..resized_w {
                let sx = ((rx as f32 * inv_scale_x) as usize).min(src_w - 1);
                let src_idx = src_row + sx * 3;

                let r = src[src_idx] as f32 * norm;
                let g = src[src_idx + 1] as f32 * norm;
                let b = src[src_idx + 2] as f32 * norm;

                let dst_idx = dst_row + (pad_left + rx);
                self.tensor_buffer[dst_idx] = r;
                self.tensor_buffer[channel_stride + dst_idx] = g;
                self.tensor_buffer[2 * channel_stride + dst_idx] = b;
            }
        }

        (&self.tensor_buffer, self.last_geometry)
    }

    /// Преобразует координаты из пространства модели 320x320 обратно в оригинальные пиксели экрана
    pub fn unmap_box(&self, cx: f32, cy: f32, w: f32, h: f32, orig_w: f32, orig_h: f32) -> (f32, f32, f32, f32) {
        let geo = self.last_geometry;
        if geo.scale <= 0.0 {
            return (cx, cy, w, h);
        }

        let real_cx = ((cx - geo.pad_left) / geo.scale).clamp(0.0, orig_w);
        let real_cy = ((cy - geo.pad_top) / geo.scale).clamp(0.0, orig_h);
        let real_w = (w / geo.scale).min(orig_w);
        let real_h = (h / geo.scale).min(orig_h);

        (real_cx, real_cy, real_w, real_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proportional_letterbox() {
        let mut lb = Letterbox::new(320, 320);
        let src = vec![200u8; 1280 * 720 * 3];
        let (tensor, geo) = lb.process_rgb24(&src, 1280, 720);

        assert_eq!(tensor.len(), 1 * 3 * 320 * 320);
        assert_eq!(geo.resized_w, 320);
        assert_eq!(geo.resized_h, 180);
        assert_eq!(geo.pad_left, 0.0);
        assert_eq!(geo.pad_top, 70.0);

        // Проверяем обратное преобразование координат:
        // Центр в модели (160, 160) должен спроецироваться ровно в центр 1280x720: (640, 360)!
        let (cx, cy, _, _) = lb.unmap_box(160.0, 160.0, 10.0, 10.0, 1280.0, 720.0);
        assert!((cx - 640.0).abs() < 1e-3);
        assert!((cy - 360.0).abs() < 1e-3);
    }
}
