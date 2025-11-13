use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::sim::SimSettings;

#[derive(Clone, Serialize, Deserialize)]
pub struct SettingsPreset {
    pub name: String,
    pub settings: SimSettings,
}

#[derive(Serialize, Deserialize, Default)]
struct PresetFile {
    presets: Vec<SettingsPreset>,
}

fn preset_file_path() -> Result<PathBuf> {
    ProjectDirs::from("org", "dust", "Dust")
        .map(|dirs| dirs.config_dir().join("presets.json"))
        .ok_or_else(|| anyhow!("unable to resolve config directory"))
}

pub fn load_presets() -> Result<Vec<SettingsPreset>> {
    let path = preset_file_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = fs::read(&path).with_context(|| format!("failed to read {path:?}"))?;
    let parsed: PresetFile =
        serde_json::from_slice(&data).context("failed to parse presets file")?;
    Ok(parsed.presets)
}

pub fn save_presets(presets: &[SettingsPreset]) -> Result<()> {
    let path = preset_file_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("failed to create dir {parent:?}"))?;
    }
    let payload = PresetFile {
        presets: presets.to_vec(),
    };
    let data = serde_json::to_vec_pretty(&payload).context("failed to serialize presets")?;
    fs::write(&path, data).with_context(|| format!("failed to write {path:?}"))?;
    Ok(())
}
