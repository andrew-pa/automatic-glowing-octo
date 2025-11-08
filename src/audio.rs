use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use parking_lot::{Mutex, RwLock};
use pipewire as pw;
use pw::{
    channel::channel,
    properties::{PropertiesBox, properties},
    spa,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::sim::{AudioSettings, GRAVITY_WELL_COUNT};

static PIPEWIRE_INIT: OnceLock<()> = OnceLock::new();

pub struct AudioEngine {
    shared: Arc<AudioShared>,
    pipeline: Option<AudioPipeline>,
    targets: Arc<RwLock<Vec<AudioTarget>>>,
}

#[derive(Clone, Debug)]
pub struct AudioTarget {
    pub id: u32,
    pub name: String,
    pub description: String,
    pub media_class: String,
}

impl AudioTarget {
    pub fn label(&self) -> String {
        if self.description.is_empty() {
            format!("{} ({})", self.name, self.media_class)
        } else {
            format!(
                "{} — {} ({})",
                self.description, self.name, self.media_class
            )
        }
    }
}

impl AudioEngine {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(AudioShared::new()),
            pipeline: None,
            targets: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn refresh(&mut self, settings: &AudioSettings) {
        self.shared.update_config(settings);
        if !settings.enabled {
            if let Some(mut pipeline) = self.pipeline.take() {
                pipeline.shutdown();
            }
            return;
        }

        let key = AudioPipelineKey::from(settings);
        let rebuild = self
            .pipeline
            .as_ref()
            .map(|pipe| pipe.key != key)
            .unwrap_or(true);

        if rebuild {
            if let Some(mut pipeline) = self.pipeline.take() {
                pipeline.shutdown();
            }
            match AudioPipeline::spawn(self.shared.clone(), key) {
                Ok(pipeline) => self.pipeline = Some(pipeline),
                Err(err) => log::error!("Failed to start PipeWire capture: {err:?}"),
            }
        }
    }

    pub fn bands(&self) -> [f32; GRAVITY_WELL_COUNT] {
        self.shared.latest_bands()
    }

    pub fn refresh_targets(&self) {
        match enumerate_targets() {
            Ok(list) => *self.targets.write() = list,
            Err(err) => log::warn!("audio target enumeration failed: {err:?}"),
        }
    }

    pub fn targets(&self) -> Vec<AudioTarget> {
        self.targets.read().clone()
    }
}

pub fn enumerate_targets() -> Result<Vec<AudioTarget>> {
    ensure_pipewire();
    let mainloop = pw::main_loop::MainLoopBox::new(None)?;
    let context = pw::context::ContextBox::new(mainloop.loop_(), None)?;
    let core = context.connect(None)?;
    let registry = core.get_registry()?;

    use std::sync::atomic::{AtomicBool, Ordering};
    let targets = Arc::new(Mutex::new(Vec::new()));
    let updated = Arc::new(AtomicBool::new(false));

    let targets_listener = targets.clone();
    let updated_listener = updated.clone();

    let listener = registry
        .add_listener_local()
        .global(move |global| {
            if global.type_ != pw::types::ObjectType::Node {
                return;
            }
            let Some(props) = global.props.as_ref() else {
                return;
            };
            let dict = props.as_ref();
            let media_class = dict.get("media.class").unwrap_or_default();
            if !media_class.starts_with("Audio/") {
                return;
            }
            let node_name = dict.get("node.name").unwrap_or_default();
            if node_name.is_empty() {
                return;
            }
            let description = dict
                .get("node.description")
                .or_else(|| dict.get("application.name"))
                .unwrap_or("");
            let target = AudioTarget {
                id: global.id,
                name: node_name.to_string(),
                description: description.to_string(),
                media_class: media_class.to_string(),
            };
            targets_listener.lock().push(target);
            updated_listener.store(true, Ordering::Relaxed);
        })
        .global_remove(|_| {})
        .register();

    // iterate a few times to gather events
    for _ in 0..20 {
        mainloop.loop_().iterate(Duration::from_millis(10));
        if updated.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
    }
    drop(listener);

    let mut list = targets.lock().clone();
    list.sort_by_key(|t| t.id);
    list.dedup_by_key(|t| t.id);
    list.sort_by(|a, b| a.label().cmp(&b.label()));
    Ok(list)
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        if let Some(mut pipeline) = self.pipeline.take() {
            pipeline.shutdown();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AudioPipelineKey {
    device: Option<String>,
    capture_sink: bool,
}

impl From<&AudioSettings> for AudioPipelineKey {
    fn from(settings: &AudioSettings) -> Self {
        Self {
            device: if settings.device.trim().is_empty() {
                None
            } else {
                Some(settings.device.trim().to_string())
            },
            capture_sink: settings.capture_sink,
        }
    }
}

struct AudioPipeline {
    key: AudioPipelineKey,
    quit: pw::channel::Sender<()>,
    thread: Option<thread::JoinHandle<()>>,
}

impl AudioPipeline {
    fn spawn(shared: Arc<AudioShared>, key: AudioPipelineKey) -> Result<Self> {
        let (quit_tx, quit_rx) = channel::<()>();
        let thread_key = key.clone();
        let thread_shared = shared.clone();
        let handle = thread::Builder::new()
            .name("dust-audio".into())
            .spawn(move || {
                if let Err(err) = audio_thread(thread_shared, thread_key, quit_rx) {
                    log::error!("Audio thread exited: {err:?}");
                }
            })
            .context("failed to spawn audio thread")?;

        Ok(Self {
            key,
            quit: quit_tx,
            thread: Some(handle),
        })
    }

    fn shutdown(&mut self) {
        let _ = self.quit.send(());
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AudioPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn audio_thread(
    shared: Arc<AudioShared>,
    key: AudioPipelineKey,
    quit_rx: pw::channel::Receiver<()>,
) -> Result<()> {
    ensure_pipewire();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let props = build_properties(&key);
    let stream = pw::stream::StreamBox::new(&core, "dust-audio", props)?;

    let data = AudioUserData::new(shared.clone());
    let _listener = stream
        .add_local_listener_with_user_data(data)
        .param_changed(|_, data, id, param| data.handle_param(id, param))
        .process(|stream, data| data.process(stream))
        .register()?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    let format_obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };

    let serialized = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(format_obj),
    )
    .context("failed to serialize PipeWire format")?
    .0
    .into_inner();
    let mut params = [pw::spa::pod::Pod::from_bytes(&serialized)
        .context("failed to parse PipeWire enum format pod")?];

    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    let loop_ref = mainloop.loop_();
    let mainloop_clone = mainloop.clone();
    let _quit = quit_rx.attach(loop_ref, move |_| {
        mainloop_clone.quit();
    });

    mainloop.run();
    Ok(())
}

fn ensure_pipewire() {
    PIPEWIRE_INIT.get_or_init(|| {
        pw::init();
    });
}

fn build_properties(key: &AudioPipelineKey) -> PropertiesBox {
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
    };
    if let Some(target) = &key.device {
        props.insert("target.object", target.clone());
    }
    if key.capture_sink {
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }
    props.to_owned()
}

struct AudioShared {
    config: RwLock<AudioRuntimeConfig>,
    bands: Mutex<[f32; GRAVITY_WELL_COUNT]>,
}

impl AudioShared {
    fn new() -> Self {
        Self {
            config: RwLock::new(AudioRuntimeConfig::default()),
            bands: Mutex::new([0.0; GRAVITY_WELL_COUNT]),
        }
    }

    fn update_config(&self, settings: &AudioSettings) {
        let mut cfg = self.config.write();
        cfg.gain = settings.gain.clamp(0.1, 100.0);
        cfg.gate = settings.gate.clamp(0.0, 1.0);
        cfg.smoothing = settings.smoothing.clamp(0.0, 0.995);
    }

    fn config(&self) -> AudioRuntimeConfig {
        *self.config.read()
    }

    fn publish(&self, values: [f32; GRAVITY_WELL_COUNT]) {
        *self.bands.lock() = values;
    }

    fn latest_bands(&self) -> [f32; GRAVITY_WELL_COUNT] {
        *self.bands.lock()
    }
}

#[derive(Clone, Copy)]
struct AudioRuntimeConfig {
    gain: f32,
    gate: f32,
    smoothing: f32,
}

impl Default for AudioRuntimeConfig {
    fn default() -> Self {
        Self {
            gain: 1.0,
            gate: 0.01,
            smoothing: 0.5,
        }
    }
}

struct AudioUserData {
    shared: Arc<AudioShared>,
    analyzer: AudioAnalyzer,
    format: spa::param::audio::AudioInfoRaw,
}

impl AudioUserData {
    fn new(shared: Arc<AudioShared>) -> Self {
        Self {
            shared,
            analyzer: AudioAnalyzer::new(),
            format: Default::default(),
        }
    }

    fn handle_param(&mut self, id: u32, param: Option<&pw::spa::pod::Pod>) {
        let Some(pod) = param else {
            return;
        };
        if id != pw::spa::param::ParamType::Format.as_raw() {
            return;
        }

        use spa::param::format::{MediaSubtype, MediaType};
        use spa::param::format_utils;

        let Ok((media_type, media_subtype)) = format_utils::parse_format(pod) else {
            return;
        };
        if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
            return;
        }

        if self.format.parse(pod).is_ok() {
            let rate = self.format.rate();
            self.analyzer.set_sample_rate(rate.max(1));
        }
    }

    fn process(&mut self, stream: &pw::stream::Stream) {
        let Some(mut buffer) = stream.dequeue_buffer() else {
            return;
        };
        let datas = buffer.datas_mut();
        if datas.is_empty() {
            return;
        }

        let data = &mut datas[0];
        let size = data.chunk().size() as usize;
        if size == 0 {
            return;
        }
        let Some(raw) = data.data() else {
            return;
        };
        if raw.len() < size {
            return;
        }

        let bytes = &raw[..size];
        let samples: &[f32] = cast_slice(bytes);
        let channels = self.format.channels().max(1) as usize;
        let config = self.shared.config();
        if let Some(result) = self.analyzer.consume(samples, channels, config) {
            self.shared.publish(result.values);
        }
    }
}

struct AudioAnalyzer {
    fft_size: usize,
    sample_rate: u32,
    window: Vec<f32>,
    buffer: Vec<f32>,
    cursor: usize,
    spectrum: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
    smoothed: [f32; GRAVITY_WELL_COUNT],
}

struct AudioFrameResult {
    values: [f32; GRAVITY_WELL_COUNT],
}

impl AudioAnalyzer {
    fn new() -> Self {
        let fft_size = 1024;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let mut analyzer = Self {
            fft_size,
            sample_rate: 48_000,
            window: vec![0.0; fft_size],
            buffer: vec![0.0; fft_size],
            cursor: 0,
            spectrum: vec![Complex32::default(); fft_size],
            fft,
            smoothed: [0.0; GRAVITY_WELL_COUNT],
        };
        analyzer.update_window();
        analyzer
    }

    fn set_sample_rate(&mut self, rate: u32) {
        self.sample_rate = rate.max(1);
    }

    fn consume(
        &mut self,
        samples: &[f32],
        channels: usize,
        config: AudioRuntimeConfig,
    ) -> Option<AudioFrameResult> {
        let stride = channels.max(1);
        let mut result = None;
        for frame in samples.chunks(stride) {
            let mono = frame.iter().copied().sum::<f32>() / frame.len() as f32;
            if let Some(res) = self.push_sample(mono, config) {
                result = Some(res);
            }
        }
        result
    }

    fn push_sample(&mut self, sample: f32, config: AudioRuntimeConfig) -> Option<AudioFrameResult> {
        self.buffer[self.cursor] = sample;
        self.cursor += 1;
        if self.cursor >= self.fft_size {
            self.cursor = 0;
            Some(self.compute_frame(config))
        } else {
            None
        }
    }

    fn compute_frame(&mut self, config: AudioRuntimeConfig) -> AudioFrameResult {
        for i in 0..self.fft_size {
            self.spectrum[i].re = self.buffer[i] * self.window[i];
            self.spectrum[i].im = 0.0;
        }
        self.fft.process(&mut self.spectrum);

        const BANDS: [(f32, f32); GRAVITY_WELL_COUNT] = [
            (20.0, 200.0),
            (200.0, 800.0),
            (800.0, 3_000.0),
            (3_000.0, 12_000.0),
        ];

        let mut accum = [0.0; GRAVITY_WELL_COUNT];
        let norm = 1.0 / (self.fft_size as f32).powi(2);
        for i in 1..self.fft_size / 2 {
            let freq = i as f32 * (self.sample_rate as f32 / self.fft_size as f32);
            let energy = self.spectrum[i].norm_sqr() * norm;
            for (band_idx, range) in BANDS.iter().enumerate() {
                if freq >= range.0 && freq < range.1 {
                    accum[band_idx] += energy;
                    break;
                }
            }
        }

        let mut values = [0.0; GRAVITY_WELL_COUNT];
        for (dst, src) in values.iter_mut().zip(accum) {
            *dst = (src * config.gain).sqrt().clamp(0.0, 10.0);
        }
        let avg = values.iter().copied().sum::<f32>() / values.len() as f32;
        if avg < config.gate {
            values = [0.0; GRAVITY_WELL_COUNT];
        }

        let smooth = config.smoothing;
        for i in 0..GRAVITY_WELL_COUNT {
            self.smoothed[i] = self.smoothed[i] * smooth + values[i] * (1.0 - smooth);
        }

        let normalized = self.smoothed.map(|v| (v / 5.0).clamp(0.0, 1.0));
        AudioFrameResult { values: normalized }
    }

    fn update_window(&mut self) {
        let n = self.fft_size as f32;
        for i in 0..self.fft_size {
            self.window[i] = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1.0)).cos();
        }
    }
}
