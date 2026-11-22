use std::io::Read;
use std::process::{Child, ChildStdout, Command, Stdio};

/// Источник видеопотока через пайп ffmpeg (поддерживает файлы MP4/AVI и устройства V4L2)
#[allow(dead_code)]
pub struct VideoReader {
    pub width: usize,
    pub height: usize,
    child: Child,
    stdout: ChildStdout,
    buffer: Vec<u8>,
    pub frame_count: u64,
}

impl VideoReader {
    /// Открывает видеофайл или устройство
    /// realtime: true — воспроизведение со скоростью 60 FPS (-re), false — максимальная скорость для бенчмарка
    pub fn open(source: &str, width: usize, height: usize, realtime: bool) -> Result<Self, String> {
        let frame_size_bytes = width * height * 3;

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-hide_banner")
            .arg("-loglevel").arg("error");

        if realtime {
            cmd.arg("-re");
        }

        cmd.arg("-i").arg(source)
            .arg("-f").arg("rawvideo")
            .arg("-pix_fmt").arg("rgb24")
            .arg("-s").arg(format!("{}x{}", width, height))
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd.spawn()
            .map_err(|e| format!("Не удалось запустить ffmpeg для {}: {}", source, e))?;

        let stdout = child.stdout.take()
            .ok_or_else(|| "Не удалось захватить stdout процесса ffmpeg".to_string())?;

        Ok(Self {
            width,
            height,
            child,
            stdout,
            buffer: vec![0u8; frame_size_bytes],
            frame_count: 0,
        })
    }

    /// Читает следующий кадр в буфер. Возвращает срез байтов кадра или None при завершении потока
    pub fn read_frame(&mut self) -> Option<&[u8]> {
        if self.stdout.read_exact(&mut self.buffer).is_ok() {
            self.frame_count += 1;
            Some(&self.buffer)
        } else {
            None
        }
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
