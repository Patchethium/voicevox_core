//! cargo run -p voicevox_core --example pitch_noise --features load-onnxruntime --
//!   ORT VVM STYLE KANA SIGMA SEED OUTPUT.wav
use anyhow::{Context as _, ensure};
use voicevox_core::{
    AccelerationMode, PitchNoiseOptions, StyleId,
    blocking::{Onnxruntime, Synthesizer, VoiceModelFile},
};

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 7,
        "usage: pitch_noise ORT VVM STYLE KANA SIGMA SEED OUTPUT.wav"
    );
    let rt = Onnxruntime::load_once().filename(&args[0]).perform()?;
    let synth = Synthesizer::builder(rt)
        .acceleration_mode(AccelerationMode::Cpu)
        .cpu_num_threads(1)
        .build()?;
    synth
        .load_voice_model(&VoiceModelFile::open(&args[1])?)
        .perform()?;
    let style = StyleId::new(args[2].parse().context("STYLE must be a u32")?);
    let options = PitchNoiseOptions {
        sigma: args[4].parse().context("SIGMA must be a float")?,
        seed: args[5].parse().context("SEED must be a u32")?,
    };
    let mut query = synth.create_audio_query_from_kana(&args[3], style)?;
    query.accent_phrases =
        synth.replace_mora_pitch_with_noise(&query.accent_phrases, style, options)?;
    let wav = synth.synthesis(&query, style).perform()?;
    std::fs::write(&args[6], wav)?;
    eprintln!(
        "saved {} (sigma={}, seed={})",
        args[6], options.sigma, options.seed
    );
    Ok(())
}
