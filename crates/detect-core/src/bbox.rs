/// Ограничивающая рамка обнаруженного объекта.
/// Координаты могут храниться как в абсолютных пикселях, так и в нормализованных (0.0..1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Координата центра по горизонтали
    pub cx: f32,
    /// Координата центра по вертикали
    pub cy: f32,
    /// Ширина рамки
    pub w: f32,
    /// Высота рамки
    pub h: f32,
}

impl BBox {
    /// Создает рамку по центру и размерам
    pub fn new(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self { cx, cy, w, h }
    }

    /// Левая граница (X min)
    #[inline]
    pub fn x_min(&self) -> f32 {
        self.cx - self.w * 0.5
    }

    /// Верхняя граница (Y min)
    #[inline]
    pub fn y_min(&self) -> f32 {
        self.cy - self.h * 0.5
    }

    /// Правая граница (X max)
    #[inline]
    pub fn x_max(&self) -> f32 {
        self.cx + self.w * 0.5
    }

    /// Нижняя граница (Y max)
    #[inline]
    pub fn y_max(&self) -> f32 {
        self.cy + self.h * 0.5
    }

    /// Площадь рамки
    #[inline]
    pub fn area(&self) -> f32 {
        (self.w.max(0.0)) * (self.h.max(0.0))
    }

    /// Коэффициент пересечения по объединению (Intersection over Union, IoU)
    pub fn iou(&self, other: &BBox) -> f32 {
        let x1 = self.x_min().max(other.x_min());
        let y1 = self.y_min().max(other.y_min());
        let x2 = self.x_max().min(other.x_max());
        let y2 = self.y_max().min(other.y_max());

        let intersection_w = (x2 - x1).max(0.0);
        let intersection_h = (y2 - y1).max(0.0);
        let intersection_area = intersection_w * intersection_h;

        if intersection_area <= 0.0 {
            return 0.0;
        }

        let union_area = self.area() + other.area() - intersection_area;
        if union_area <= 0.0 {
            0.0
        } else {
            intersection_area / union_area
        }
    }
}

/// Результат единичной детекции
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Геометрия рамки
    pub bbox: BBox,
    /// Уверенность детектора (0.0 .. 1.0)
    pub confidence: f32,
    /// Идентификатор класса
    pub class_id: u32,
}

impl Detection {
    pub fn new(bbox: BBox, confidence: f32, class_id: u32) -> Self {
        Self {
            bbox,
            confidence,
            class_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_iou() {
        let b1 = BBox::new(100.0, 100.0, 50.0, 50.0);
        let b2 = BBox::new(100.0, 100.0, 50.0, 50.0);
        assert!((b1.iou(&b2) - 1.0).abs() < 1e-5);

        let b3 = BBox::new(200.0, 200.0, 50.0, 50.0);
        assert_eq!(b1.iou(&b3), 0.0);
    }
}
