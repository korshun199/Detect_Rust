/// Высокоскоростной ресайз и нормализация кадра в тензор [1, 3, target_h, target_w] для YOLO
pub struct Letterbox {
    pub target_w: usize,
    pub target_h: usize,
    pub tensor_buffer: Vec<f32>,
}

impl Letterbox {
    pub fn new(target_w: usize, target_h: usize) -> Self {
        let size = 1 * 3 * target_h * target_w;
        Self {
            target_w,
            target_h,
            tensor_buffer: vec![0.0f32; size],
        }
    }

    /// Преобразует RGB24 буфер (src_w x src_h x 3) в тензор NCHW [1, 3, 320, 320]
    /// с нормализацией пикселей 0..255 -> 0.0..1.0
    pub fn process_rgb24(&mut self, src: &[u8], src_w: usize, src_h: usize) -> &[f32] {
        let tw = self.target_w;
        let th = self.target_h;
        let channel_stride = tw * th;

        let scale_x = src_w as f32 / tw as f32;
        let scale_y = src_h as f32 / th as f32;

        let norm = 1.0 / 255.0;

        for ty in 0..th {
            let sy = ((ty as f32 * scale_y) as usize).min(src_h - 1);
            let src_row_offset = sy * src_w * 3;
            let dst_row_offset = ty * tw;

            for tx in 0..tw {
                let sx = ((tx as f32 * scale_x) as usize).min(src_w - 1);
                let src_idx = src_row_offset + sx * 3;

                let r = src[src_idx] as f32 * norm;
                let g = src[src_idx + 1] as f32 * norm;
                let b = src[src_idx + 2] as f32 * norm;

                let dst_idx = dst_row_offset + tx;
                self.tensor_buffer[dst_idx] = r;
                self.tensor_buffer[channel_stride + dst_idx] = g;
                self.tensor_buffer[2 * channel_stride + dst_idx] = b;
            }
        }

        &self.tensor_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_processing() {
        let mut lb = Letterbox::new(320, 320);
        let src = vec![255u8; 1280 * 720 * 3];
        let tensor = lb.process_rgb24(&src, 1280, 720);
        assert_eq!(tensor.len(), 1 * 3 * 320 * 320);
        assert!((tensor[0] - 1.0).abs() < 1e-4);
    }
}
