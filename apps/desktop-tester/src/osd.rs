use detect_core::BBox;

/// Встроенный микро-шрифт 5x7 для вывода системной информации без внешних файлов шрифтов
const FONT_5X7: [[u8; 5]; 96] = {
    let mut table = [[0u8; 5]; 96];
    // Пробел
    table[0] = [0, 0, 0, 0, 0];
    // Цифры 0-9 (ASCII 48-57 -> индексы 16-25)
    table[16] = [0x3E, 0x51, 0x49, 0x45, 0x3E]; // 0
    table[17] = [0x00, 0x42, 0x7F, 0x40, 0x00]; // 1
    table[18] = [0x42, 0x61, 0x51, 0x49, 0x46]; // 2
    table[19] = [0x21, 0x41, 0x45, 0x4B, 0x31]; // 3
    table[20] = [0x18, 0x14, 0x12, 0x7F, 0x10]; // 4
    table[21] = [0x27, 0x45, 0x45, 0x45, 0x39]; // 5
    table[22] = [0x3C, 0x4A, 0x49, 0x49, 0x30]; // 6
    table[23] = [0x01, 0x71, 0x09, 0x05, 0x03]; // 7
    table[24] = [0x36, 0x49, 0x49, 0x49, 0x36]; // 8
    table[25] = [0x06, 0x49, 0x49, 0x29, 0x1E]; // 9
    // Двоеточие : (58 -> 26)
    table[26] = [0x00, 0x36, 0x36, 0x00, 0x00];
    // Точка . (46 -> 14)
    table[14] = [0x00, 0x60, 0x60, 0x00, 0x00];
    // Буквы A-Z (ASCII 65-90 -> 33-58)
    table[33] = [0x7E, 0x11, 0x11, 0x11, 0x7E]; // A
    table[34] = [0x7F, 0x49, 0x49, 0x49, 0x36]; // B
    table[35] = [0x3E, 0x41, 0x41, 0x41, 0x22]; // C
    table[36] = [0x7F, 0x41, 0x41, 0x22, 0x1C]; // D
    table[37] = [0x7F, 0x49, 0x49, 0x49, 0x41]; // E
    table[38] = [0x7F, 0x09, 0x09, 0x09, 0x01]; // F
    table[39] = [0x3E, 0x41, 0x49, 0x49, 0x7A]; // G
    table[40] = [0x7F, 0x08, 0x08, 0x08, 0x7F]; // H
    table[41] = [0x00, 0x41, 0x7F, 0x41, 0x00]; // I
    table[42] = [0x20, 0x40, 0x41, 0x3F, 0x01]; // J
    table[43] = [0x7F, 0x08, 0x14, 0x22, 0x41]; // K
    table[44] = [0x7F, 0x40, 0x40, 0x40, 0x40]; // L
    table[45] = [0x7F, 0x02, 0x0C, 0x02, 0x7F]; // M
    table[46] = [0x7F, 0x04, 0x08, 0x10, 0x7F]; // N
    table[47] = [0x3E, 0x41, 0x41, 0x41, 0x3E]; // O
    table[48] = [0x7F, 0x09, 0x09, 0x09, 0x06]; // P
    table[49] = [0x3E, 0x41, 0x51, 0x21, 0x5E]; // Q
    table[50] = [0x7F, 0x09, 0x19, 0x29, 0x46]; // R
    table[51] = [0x46, 0x49, 0x49, 0x49, 0x31]; // S
    table[52] = [0x01, 0x01, 0x7F, 0x01, 0x01]; // T
    table[53] = [0x3F, 0x40, 0x40, 0x40, 0x3F]; // U
    table[54] = [0x1F, 0x20, 0x40, 0x20, 0x1F]; // V
    table[55] = [0x7F, 0x20, 0x18, 0x20, 0x7F]; // W
    table[56] = [0x63, 0x14, 0x08, 0x14, 0x63]; // X
    table[57] = [0x07, 0x08, 0x70, 0x08, 0x07]; // Y
    table[58] = [0x61, 0x51, 0x49, 0x45, 0x43]; // Z
    // Дефис / минус - (45 -> 13)
    table[13] = [0x08, 0x08, 0x08, 0x08, 0x08];
    // Плюс + (43 -> 11)
    table[11] = [0x08, 0x08, 0x3E, 0x08, 0x08];
    // Скобки [ ] (91 -> 59, 93 -> 61)
    table[59] = [0x7F, 0x41, 0x41, 0x00, 0x00];
    table[61] = [0x00, 0x00, 0x41, 0x41, 0x7F];
    table
};

/// Преобразует срез RGB24 в буфер пикселей 0x00RRGGBB для экрана
pub fn rgb24_to_argb_buffer(src: &[u8], dst: &mut [u32], count: usize) {
    for i in 0..count {
        let r = src[i * 3] as u32;
        let g = src[i * 3 + 1] as u32;
        let b = src[i * 3 + 2] as u32;
        dst[i] = (r << 16) | (g << 8) | b;
    }
}

/// Рисует линию по алгоритму Брезенхэма (для направляющего вектора прицела)
pub fn draw_line(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    mut x0: isize,
    mut y0: isize,
    x1: isize,
    y1: isize,
    color: u32,
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && x0 < width as isize && y0 >= 0 && y0 < height as isize {
            buffer[(y0 as usize) * width + x0 as usize] = color;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Рисует перекрестие прицела (Crosshair)
pub fn draw_crosshair(buffer: &mut [u32], width: usize, height: usize, cx: isize, cy: isize, size: isize, color: u32) {
    let gap = 5;
    for dx in gap..=size {
        if cx + dx < width as isize && cy >= 0 && cy < height as isize {
            buffer[(cy as usize) * width + (cx + dx) as usize] = color;
        }
        if cx - dx >= 0 && cy >= 0 && cy < height as isize {
            buffer[(cy as usize) * width + (cx - dx) as usize] = color;
        }
    }
    for dy in gap..=size {
        if cy + dy < height as isize && cx >= 0 && cx < width as isize {
            buffer[((cy + dy) as usize) * width + cx as usize] = color;
        }
        if cy - dy >= 0 && cx >= 0 && cx < width as isize {
            buffer[((cy - dy) as usize) * width + cx as usize] = color;
        }
    }
    if cx >= 0 && cx < width as isize && cy >= 0 && cy < height as isize {
        buffer[(cy as usize) * width + cx as usize] = color;
    }
}

/// Рисует тактическую рамку захвата цели (угловые скобки / BBox)
pub fn draw_bbox(buffer: &mut [u32], width: usize, height: usize, bbox: &BBox, color: u32) {
    let x1 = (bbox.x_min() as isize).clamp(0, width as isize - 1);
    let y1 = (bbox.y_min() as isize).clamp(0, height as isize - 1);
    let x2 = (bbox.x_max() as isize).clamp(0, width as isize - 1);
    let y2 = (bbox.y_max() as isize).clamp(0, height as isize - 1);

    if x2 <= x1 || y2 <= y1 {
        return;
    }

    let corner_len = (((x2 - x1).min(y2 - y1)) / 3).max(6).min(25);

    for x in x1..=(x1 + corner_len).min(x2) {
        buffer[(y1 as usize) * width + x as usize] = color;
        buffer[(y2 as usize) * width + x as usize] = color;
    }
    for x in (x2 - corner_len).max(x1)..=x2 {
        buffer[(y1 as usize) * width + x as usize] = color;
        buffer[(y2 as usize) * width + x as usize] = color;
    }
    for y in y1..=(y1 + corner_len).min(y2) {
        buffer[(y as usize) * width + x1 as usize] = color;
        buffer[(y as usize) * width + x2 as usize] = color;
    }
    for y in (y2 - corner_len).max(y1)..=y2 {
        buffer[(y as usize) * width + x1 as usize] = color;
        buffer[(y as usize) * width + x2 as usize] = color;
    }

    let cx = (bbox.cx as isize).clamp(0, width as isize - 1);
    let cy = (bbox.cy as isize).clamp(0, height as isize - 1);
    draw_crosshair(buffer, width, height, cx, cy, 6, color);
}

/// Выводит текст на экран растровым шрифтом 5x7 с масштабированием
pub fn draw_text(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    scale: usize,
) {
    let mut cursor_x = x;

    for ch in text.chars() {
        let ascii = ch as usize;
        let glyph_idx = if ascii >= 32 && ascii < 128 {
            ascii - 32
        } else {
            0
        };

        let glyph = FONT_5X7[glyph_idx];

        for col in 0..5 {
            let col_bits = glyph[col];
            for row in 0..7 {
                if (col_bits & (1 << row)) != 0 {
                    for sx in 0..scale {
                        for sy in 0..scale {
                            let px = cursor_x + col * scale + sx;
                            let py = y + row * scale + sy;
                            if px < width && py < height {
                                buffer[py * width + px] = color;
                            }
                        }
                    }
                }
            }
        }

        cursor_x += 6 * scale;
    }
}

/// Отрисовывает тактические границы рабочей зоны обнаружения (отсечение вращающихся боковых винтов)
pub fn draw_roi_boundaries(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    margin_left_pct: f32,
    margin_right_pct: f32,
    color: u32,
) {
    let left_x = (width as f32 * (margin_left_pct / 100.0)).round() as usize;
    let right_x = (width as f32 * (1.0 - margin_right_pct / 100.0)).round() as usize;

    if left_x > 0 && left_x < width {
        // Пунктирная вертикальная разделительная линия слева
        for y in (0..height).step_by(12) {
            for dy in 0..6 {
                if y + dy < height {
                    buffer[(y + dy) * width + left_x] = color;
                }
            }
        }
    }

    if right_x > 0 && right_x < width {
        // Пунктирная вертикальная разделительная линия справа
        for y in (0..height).step_by(12) {
            for dy in 0..6 {
                if y + dy < height {
                    buffer[(y + dy) * width + right_x] = color;
                }
            }
        }
    }
}
