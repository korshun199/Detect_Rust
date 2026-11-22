use detect_core::{BBox, Detection, non_max_suppression};

/// Детектор на базе модели YOLOv8/v11 (например, anti_uav_fast_320_int8_pc.onnx)
pub struct YoloDetector {
    /// Порог уверенности для отсечения шума
    pub conf_threshold: f32,
    /// Порог IoU для NMS
    pub nms_threshold: f32,
    /// Целевой размер входа модели (320x320)
    pub input_size: u32,
}

impl YoloDetector {
    pub fn new(conf_threshold: f32, nms_threshold: f32) -> Self {
        Self {
            conf_threshold,
            nms_threshold,
            input_size: 320,
        }
    }

    /// Преобразование сырого тензора [1, 6, 2100] в список детекций с NMS
    pub fn decode_output(
        &self,
        raw_output: &[f32],
        orig_width: f32,
        orig_height: f32,
    ) -> Vec<Detection> {
        let num_anchors = 2100;
        let num_channels = 6;
        let mut candidates = Vec::new();

        if raw_output.len() < num_anchors * num_channels {
            return candidates;
        }

        let scale_x = orig_width / (self.input_size as f32);
        let scale_y = orig_height / (self.input_size as f32);

        // Формат выхода YOLOv8/v11: [batch=1, channels=6, anchors=2100]
        // Канал 0: cx
        // Канал 1: cy
        // Канал 2: w
        // Канал 3: h
        // Канал 4: class_0 score
        // Канал 5: class_1 score
        for i in 0..num_anchors {
            let cx = raw_output[0 * num_anchors + i];
            let cy = raw_output[1 * num_anchors + i];
            let w = raw_output[2 * num_anchors + i];
            let h = raw_output[3 * num_anchors + i];

            let score0 = raw_output[4 * num_anchors + i];
            let score1 = raw_output[5 * num_anchors + i];

            let (best_score, class_id) = if score0 >= score1 {
                (score0, 0)
            } else {
                (score1, 1)
            };

            if best_score >= self.conf_threshold {
                let bbox = BBox::new(
                    cx * scale_x,
                    cy * scale_y,
                    w * scale_x,
                    h * scale_y,
                );
                candidates.push(Detection::new(bbox, best_score, class_id));
            }
        }

        non_max_suppression(candidates, self.nms_threshold, self.conf_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_decode() {
        let detector = YoloDetector::new(0.5, 0.45);
        let mut raw = vec![0.0f32; 6 * 2100];
        
        // Кандидат #10: cx=160, cy=160, w=30, h=30, score=0.85
        raw[0 * 2100 + 10] = 160.0;
        raw[1 * 2100 + 10] = 160.0;
        raw[2 * 2100 + 10] = 30.0;
        raw[3 * 2100 + 10] = 30.0;
        raw[4 * 2100 + 10] = 0.85;

        let detections = detector.decode_output(&raw, 1280.0, 720.0);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].confidence, 0.85);
        assert_eq!(detections[0].bbox.cx, 160.0 * (1280.0 / 320.0));
    }
}
