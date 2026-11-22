/// Форматы представления пикселей в кадре
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    Rgb8,
    Bgr8,
    Gray8,
    Nv12,
}

/// Легковесное представление сырого кадра (zero-copy ссылка на буфер в памяти)
#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: FrameFormat,
}

impl<'a> FrameView<'a> {
    pub fn new(data: &'a [u8], width: u32, height: u32, stride: u32, format: FrameFormat) -> Self {
        Self {
            data,
            width,
            height,
            stride,
            format,
        }
    }
}
