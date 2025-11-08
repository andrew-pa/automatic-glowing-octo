use egui::{ComboBox, Context, DragValue, Grid, Slider};

use crate::audio::AudioTarget;
use crate::sim::SimSettings;

pub fn band_name(index: usize) -> &'static str {
    match index {
        0 => "Low",
        1 => "Low-mid",
        2 => "High-mid",
        _ => "High",
    }
}

pub fn gravity_window(ctx: &Context, settings: &mut SimSettings, audio_bands: [f32; 4]) {
    use egui::pos2;

    egui::Window::new("Gravity Wells")
        .default_width(360.0)
        .default_pos(pos2(360.0, 16.0))
        .show(ctx, |ui| {
            ui.label("Per-well offsets and audio routing.");
            ui.separator();
            for (idx, well) in settings.gravity_wells.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.label(format!("Well {}", idx + 1));
                    Grid::new(format!("well_grid_{idx}"))
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("Position");
                            ui.horizontal(|ui| {
                                for (value, label) in well.position.iter_mut().zip(["x", "y", "z"])
                                {
                                    ui.add(
                                        DragValue::new(value)
                                            .speed(0.1)
                                            .range(-50.0..=50.0)
                                            .prefix(format!("{label}: ")),
                                    );
                                }
                            });
                            ui.end_row();
                            ui.label("Strength");
                            ui.add(Slider::new(&mut well.strength, 0.0..=50.0).text("strength"));
                            ui.end_row();
                            ui.label("Audio band");
                            ComboBox::from_id_salt(format!("well_band_{idx}"))
                                .selected_text(band_name(well.audio_band))
                                .show_ui(ui, |ui| {
                                    for (band_idx, label) in
                                        ["Low", "Low-mid", "High-mid", "High"].iter().enumerate()
                                    {
                                        ui.selectable_value(&mut well.audio_band, band_idx, *label);
                                    }
                                });
                            ui.end_row();
                            ui.label("Strength mix");
                            ui.add(Slider::new(&mut well.strength_mod, 0.0..=1.5).text("depth"));
                            ui.end_row();
                            ui.label("Position mix");
                            ui.add(Slider::new(&mut well.position_mod, 0.0..=1.5).text("depth"));
                            ui.end_row();
                        });
                });
            }
            ui.separator();
            ui.label(format!(
                "Band envelopes · L {:.2}  LM {:.2}  HM {:.2}  H {:.2}",
                audio_bands[0], audio_bands[1], audio_bands[2], audio_bands[3]
            ));
        });
}

pub fn audio_window(ctx: &Context, settings: &mut SimSettings, targets: &[AudioTarget]) {
    use egui::pos2;

    egui::Window::new("Audio Reactive")
        .default_width(320.0)
        .default_pos(pos2(740.0, 16.0))
        .show(ctx, |ui| {
            ui.checkbox(&mut settings.audio.enabled, "Enable modulation");
            ui.horizontal(|ui| {
                ui.label("Target node");
                ui.text_edit_singleline(&mut settings.audio.device);
            });
            if !targets.is_empty() {
                let selected = if settings.audio.device.is_empty() {
                    "Auto (PipeWire default)".to_string()
                } else {
                    settings.audio.device.clone()
                };
                ComboBox::from_id_salt("audio_target_combo")
                    .selected_text(selected)
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(
                                settings.audio.device.is_empty(),
                                "Auto (PipeWire default)",
                            )
                            .clicked()
                        {
                            settings.audio.device.clear();
                        }
                        for target in targets {
                            let label = target.label();
                            if ui
                                .selectable_label(settings.audio.device == target.name, label)
                                .clicked()
                            {
                                settings.audio.device = target.name.clone();
                            }
                        }
                    });
            } else {
                ui.label("Scanning PipeWire nodes…");
            }
            ui.checkbox(&mut settings.audio.capture_sink, "Capture sink monitor");
            ui.add(
                Slider::new(&mut settings.audio.gain, 0.1..=20.0)
                    .logarithmic(true)
                    .text("Gain"),
            );
            ui.add(Slider::new(&mut settings.audio.gate, 0.0..=0.2).text("Gate"));
            ui.add(Slider::new(&mut settings.audio.smoothing, 0.0..=0.99).text("Smoothing"));
            ui.separator();
            ui.checkbox(&mut settings.audio.modulate_strength, "Modulate strength");
            ui.add(
                Slider::new(&mut settings.audio.strength_depth, 0.0..=24.0).text("Strength depth"),
            );
            ui.checkbox(&mut settings.audio.modulate_position, "Modulate position");
            ui.add(
                Slider::new(&mut settings.audio.position_depth, 0.0..=10.0).text("Position depth"),
            );
        });
}
