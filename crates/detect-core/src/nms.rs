use crate::bbox::Detection;

/// Алгоритм подавления немаксимумов (Non-Maximum Suppression, NMS).
/// Отсекает перекрывающиеся рамки с меньшей уверенностью.
pub fn non_max_suppression(
    mut candidates: Vec<Detection>,
    iou_threshold: f32,
    confidence_threshold: f32,
) -> Vec<Detection> {
    // 1. Фильтруем по минимальному порогу уверенности
    candidates.retain(|d| d.confidence >= confidence_threshold);

    // 2. Сортируем по убыванию уверенности
    candidates.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected: Vec<Detection> = Vec::with_capacity(candidates.len());
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }

        let best = candidates[i];
        selected.push(best);

        for j in (i + 1)..candidates.len() {
            if suppressed[j] {
                continue;
            }

            // Подавляем только рамки того же класса
            if candidates[j].class_id == best.class_id {
                let iou = best.bbox.iou(&candidates[j].bbox);
                if iou >= iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::BBox;

    #[test]
    fn test_nms_suppression() {
        let det1 = Detection::new(BBox::new(100.0, 100.0, 40.0, 40.0), 0.90, 0);
        let det2 = Detection::new(BBox::new(102.0, 101.0, 39.0, 41.0), 0.85, 0); // дубликат
        let det3 = Detection::new(BBox::new(300.0, 300.0, 20.0, 20.0), 0.70, 0); // другой объект

        let result = non_max_suppression(vec![det1, det2, det3], 0.5, 0.5);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].confidence, 0.90);
        assert_eq!(result[1].confidence, 0.70);
    }
}
