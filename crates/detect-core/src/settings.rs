use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Глобальные настройки системы Detect_Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub system: SystemConfig,
    pub video: VideoConfig,
    pub detection: DetectionConfig,
    pub tracking: TrackingConfig,
    pub osd: OsdConfig,
    pub control: ControlConfig,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            video: VideoConfig::default(),
            detection: DetectionConfig::default(),
            tracking: TrackingConfig::default(),
            osd: OsdConfig::default(),
            control: ControlConfig::default(),
        }
    }
}

impl Settings {
    /// Загружает настройки из файла settings.json, либо создает файл с настройками по умолчанию
    pub fn load_or_create<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        if path_ref.exists() {
            let content = fs::read_to_string(path_ref)
                .map_err(|e| format!("Ошибка чтения файла конфигурации: {}", e))?;
            serde_json::from_str(&content)
                .map_err(|e| format!("Ошибка парсинга settings.json: {}", e))
        } else {
            let default_settings = Self::default();
            default_settings.save(path_ref)?;
            Ok(default_settings)
        }
    }

    /// Сохраняет текущие настройки в settings.json
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Ошибка сериализации настроек: {}", e))?;
        fs::write(path, serialized)
            .map_err(|e| format!("Ошибка записи settings.json: {}", e))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub log_level: String,
    pub hot_reload: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            hot_reload: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// "file" или "device"
    pub source_type: String,
    pub file_path: String,
    pub device_path: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            source_type: "file".to_string(),
            file_path: "/home/work/tmp/AvatarG0006.mp4".to_string(),
            device_path: "/dev/video4".to_string(),
            width: 1280,
            height: 720,
            fps: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub model_path: String,
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
    pub input_size: u32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            model_path: "models/anti_uav_fast_320_int8_pc.onnx".to_string(),
            confidence_threshold: 0.40,
            nms_iou_threshold: 0.45,
            input_size: 320,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    pub enabled: bool,
    pub algorithm: String,
    pub max_lost_frames: u32,
    pub iir_alpha: f32,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "kcf".to_string(),
            max_lost_frames: 15,
            iir_alpha: 0.65,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsdConfig {
    pub show_fps: bool,
    pub show_crosshair: bool,
    pub show_telemetry: bool,
    pub box_color: [u8; 3],
    pub text_color: [u8; 3],
    pub line_thickness: u32,
}

impl Default for OsdConfig {
    fn default() -> Self {
        Self {
            show_fps: true,
            show_crosshair: true,
            show_telemetry: true,
            box_color: [0, 255, 0],
            text_color: [255, 255, 255],
            line_thickness: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlConfig {
    pub kp_yaw: f32,
    pub ki_yaw: f32,
    pub kd_yaw: f32,
    pub kp_pitch: f32,
    pub ki_pitch: f32,
    pub kd_pitch: f32,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            kp_yaw: 1.0,
            ki_yaw: 0.05,
            kd_yaw: 0.1,
            kp_pitch: 1.0,
            ki_pitch: 0.05,
            kd_pitch: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_serialization() {
        let settings = Settings::default();
        let json = serde_json::to_string(&settings).unwrap();
        let restored: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.video.width, 1280);
        assert_eq!(restored.detection.input_size, 320);
    }
}
