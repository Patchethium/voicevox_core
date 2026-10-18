use super::*;
use crate::synthesizer::MARGIN;
use std::time::Instant;

type Session = Arc<async_lock::Mutex<ort::session::Session>>;

/// `pad_decoder_feature::<PADDING_FRAME_LENGTH>` in `synthesizer.rs`.
const PADDING: usize = 38;
const HOP: usize = 256;

fn session(rt: &blocking::Onnxruntime, bytes: Vec<u8>, threads: u16) -> anyhow::Result<Session> {
    let options = InferenceSessionOptions::new(threads, DeviceSpec::Cpu);
    Ok(Arc::new(
        rt.new_session(&ModelBytes::Onnx(bytes), options)?.0,
    ))
}

fn signature(session: &Session) -> (Vec<String>, Vec<String>) {
    let session = session.lock_blocking();
    let names =
        |outlets: &[ort::value::Outlet]| outlets.iter().map(|o| o.name().to_owned()).collect();
    (names(session.inputs()), names(session.outputs()))
}

fn float_output(ctx: OnnxruntimeRunContext) -> anyhow::Result<ndarray::ArrayD<f32>> {
    let OutputTensor::Float32(output) = blocking::Onnxruntime::run_blocking(ctx)?.remove(0) else {
        bail!("nonfloat decoder output")
    };
    Ok(output)
}

/// Padded decoder input as built by `Synthesizer::decode`: voiced stretches with
/// varying pitch, phoneme changes, and silence padding on both ends.
fn features(length: usize) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
    let total = length + 2 * PADDING;
    let f0 = ndarray::Array2::from_shape_fn((total, 1), |(i, _)| {
        let voiced = (PADDING..PADDING + length).contains(&i) && (i / 17) % 5 != 0;
        if voiced {
            5.5 + 0.2 * (i as f32 / 20.0).sin()
        } else {
            0.0
        }
    });
    let phoneme = ndarray::Array2::from_shape_fn((total, 45), |(i, p)| {
        let id = if (PADDING..PADDING + length).contains(&i) {
            (i / 9) % 44 + 1
        } else {
            0
        };
        f32::from(u8::from(p == id))
    });
    (f0, phoneme)
}

fn run_decoder_inputs(
    session: &Session,
    (f0, phoneme): &(ndarray::Array2<f32>, ndarray::Array2<f32>),
    speaker: i64,
) -> anyhow::Result<ndarray::ArrayD<f32>> {
    let mut ctx = OnnxruntimeRunContext::from(session.clone());
    ctx.push_input("f0", f0.clone())?;
    ctx.push_input("phoneme", phoneme.clone())?;
    ctx.push_input("speaker_id", ndarray::arr1(&[speaker]))?;
    float_output(ctx)
}

fn run_acoustic(
    session: &Session,
    features: &(ndarray::Array2<f32>, ndarray::Array2<f32>),
    speaker: i64,
) -> anyhow::Result<ndarray::Array2<f32>> {
    Ok(run_decoder_inputs(session, features, speaker)?.into_dimensionality()?)
}

/// `Synthesizer::decode`: the padding is trimmed from the waveform.
fn run_monolithic(
    session: &Session,
    features: &(ndarray::Array2<f32>, ndarray::Array2<f32>),
    speaker: i64,
) -> anyhow::Result<Vec<f32>> {
    let wave = run_decoder_inputs(session, features, speaker)?
        .into_raw_vec_and_offset()
        .0;
    Ok(wave[PADDING * HOP..wave.len() - PADDING * HOP].to_vec())
}

fn run_vocoder(session: &Session, spec: ndarray::ArrayView2<'_, f32>) -> anyhow::Result<Vec<f32>> {
    let mut ctx = OnnxruntimeRunContext::from(session.clone());
    ctx.push_input("spec", spec.to_owned())?;
    let wave = float_output(ctx)?;
    ensure!(wave.len() == spec.nrows() * HOP, "unexpected vocoder hop");
    Ok(wave.into_iter().collect())
}

/// `Synthesizer::generate_full_intermediate` + `render`: keep `MARGIN` frames of context.
fn with_margin(spec_with_padding: &ndarray::Array2<f32>) -> ndarray::ArrayView2<'_, f32> {
    let rows = spec_with_padding.nrows();
    spec_with_padding.slice(ndarray::s![PADDING - MARGIN..rows - PADDING + MARGIN, ..])
}

fn render(
    vocoder: &Session,
    spec: ndarray::ArrayView2<'_, f32>,
    chunk: usize,
) -> anyhow::Result<Vec<f32>> {
    let length = spec.nrows() - 2 * MARGIN;
    let mut wave = Vec::with_capacity(length * HOP);
    for start in (0..length).step_by(chunk) {
        let end = (start + chunk).min(length);
        let segment = run_vocoder(
            vocoder,
            spec.slice(ndarray::s![start..end + 2 * MARGIN, ..]),
        )?;
        wave.extend_from_slice(&segment[MARGIN * HOP..segment.len() - MARGIN * HOP]);
    }
    Ok(wave)
}

fn max_error(a: &[f32], b: &[f32]) -> anyhow::Result<f32> {
    ensure!(
        a.len() == b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    ensure!(a.iter().chain(b).all(|x| x.is_finite()), "nonfinite output");
    Ok(a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max))
}

struct Report {
    split: f32,
    chunked: Option<f32>,
}

/// Compares the monolithic decoder against the split pair and, if `chunked`,
/// streaming-style chunked rendering against whole-utterance rendering.
fn compare(
    monolithic: &Session,
    SplitParts { acoustic, vocoder }: &SplitParts,
    speaker: i64,
    chunked: bool,
) -> anyhow::Result<Report> {
    ensure!(
        signature(acoustic)
            == (
                vec!["f0".into(), "phoneme".into(), "speaker_id".into()],
                vec!["spec".into()]
            ),
        "unexpected spectrogram signature: {:?}",
        signature(acoustic)
    );
    ensure!(
        signature(vocoder) == (vec!["spec".into()], vec!["wave".into()]),
        "unexpected vocoder signature: {:?}",
        signature(vocoder)
    );
    let features = features(200);
    let expected = run_monolithic(monolithic, &features, speaker)?;
    let spec = run_acoustic(acoustic, &features, speaker)?;
    ensure!(spec.ncols() == 80, "unexpected mel width {}", spec.ncols());
    let whole = render(vocoder, with_margin(&spec), usize::MAX)?;
    let split = max_error(&expected, &whole)?;
    let chunked = chunked
        .then(|| {
            [25, 50, 64].into_iter().try_fold(0.0_f32, |max, chunk| {
                let wave = render(vocoder, with_margin(&spec), chunk)?;
                anyhow::Ok(max.max(max_error(&whole, &wave)?))
            })
        })
        .transpose()?;
    Ok(Report { split, chunked })
}

struct SplitParts {
    acoustic: Session,
    vocoder: Session,
}

fn load_split(
    rt: &blocking::Onnxruntime,
    model: &ModelBytes,
    threads: u16,
) -> anyhow::Result<SplitParts> {
    let super::super::decoder_graph::SplitDecoder {
        spectrogram,
        vocoder,
    } = export_and_split_decoder(model)?;
    Ok(SplitParts {
        acoustic: session(rt, spectrogram, threads)?,
        vocoder: session(rt, vocoder, threads)?,
    })
}

fn sample(name: &str) -> anyhow::Result<Vec<u8>> {
    let path = format!(
        "{}/../../model/sample.vvm/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).with_context(|| path)
}

// With `MARGIN` frames of context, chunk boundaries see the same receptive field as
// whole-utterance rendering (50-frame chunks are bitwise equal on production models).
// Other segment lengths can select different ORT Conv kernels: up to 6e-4 was observed
// mid-chunk, about 20 int16 steps. A missing margin is 10-100x larger (5e-3 with 8
// frames, ~0.1 with none), so the bound separates kernel noise from lost context.
const CHUNKED_TOLERANCE: f32 = 2e-3;

#[test]
fn sample_split_matches_monolithic_and_streams_with_margin() -> anyhow::Result<()> {
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let decode = ModelBytes::Onnx(sample("decode.onnx")?);
    let monolithic = Arc::new(
        rt.new_session(&decode, InferenceSessionOptions::new(1, DeviceSpec::Cpu))?
            .0,
    );
    for speaker in 0..4 {
        let report = compare(&monolithic, &load_split(rt, &decode, 1)?, speaker, true)?;
        // Level1 may fuse kernels differently across ORT builds; observed results are bitwise.
        ensure!(
            report.split <= 1e-6,
            "speaker {speaker}: split mismatch {}",
            report.split
        );
        let chunked = report.chunked.unwrap();
        ensure!(
            chunked <= CHUNKED_TOLERANCE,
            "speaker {speaker}: chunked mismatch {chunked}"
        );
    }
    Ok(())
}

/// `sample.vvm` also ships upstream's own split of the same decoder.
#[test]
fn sample_split_matches_official_split_models() -> anyhow::Result<()> {
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let ours = load_split(rt, &ModelBytes::Onnx(sample("decode.onnx")?), 1)?;
    let official = SplitParts {
        acoustic: session(rt, sample("predict_spectrogram.onnx")?, 1)?,
        vocoder: session(rt, sample("vocoder.onnx")?, 1)?,
    };
    let features = features(120);
    for speaker in 0..4 {
        let spec = run_acoustic(&ours.acoustic, &features, speaker)?;
        let official_spec = run_acoustic(&official.acoustic, &features, speaker)?;
        let error = max_error(spec.as_slice().unwrap(), official_spec.as_slice().unwrap())?;
        ensure!(
            error <= 1e-6,
            "speaker {speaker}: spectrogram mismatch {error}"
        );
        let error = max_error(
            &run_vocoder(&ours.vocoder, spec.view())?,
            &run_vocoder(&official.vocoder, spec.view())?,
        )?;
        ensure!(error <= 1e-6, "speaker {speaker}: vocoder mismatch {error}");
    }
    Ok(())
}

#[test]
fn margin_is_required_for_chunked_rendering() -> anyhow::Result<()> {
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let SplitParts { acoustic, vocoder } =
        load_split(rt, &ModelBytes::Onnx(sample("decode.onnx")?), 1)?;
    let spec = run_acoustic(&acoustic, &features(200), 0)?;
    let core = spec.slice(ndarray::s![PADDING..spec.nrows() - PADDING, ..]);
    let whole = run_vocoder(&vocoder, core)?;
    let mut naive = vec![];
    for start in (0..core.nrows()).step_by(25) {
        let end = (start + 25).min(core.nrows());
        naive.extend(run_vocoder(
            &vocoder,
            core.slice(ndarray::s![start..end, ..]),
        )?);
    }
    let error = max_error(&whole, &naive)?;
    ensure!(
        error > 5.0 * CHUNKED_TOLERANCE,
        "zero-margin chunking unexpectedly exact: {error}"
    );
    Ok(())
}

#[test]
fn rejects_non_decoder_models() -> anyhow::Result<()> {
    blocking::Onnxruntime::from_test_util_data()?;
    let error = export_and_split_decoder(&ModelBytes::Onnx(sample("predict_intonation.onnx")?))
        .unwrap_err();
    ensure!(error.to_string().contains("mel bottleneck"), "{error:#}");
    Ok(())
}

/// Explicit local diagnostic; production models are never fixtures or exported persistently.
#[test]
#[ignore = "set AZALEA_DECODER_ORT and AZALEA_DECODER_VVM_DIR; run in a separate test process"]
fn installed_decoder_diagnostic() -> anyhow::Result<()> {
    use crate::{
        blocking::VoiceModelFile, core::infer::domains::TalkOperation, future::FutureExt as _,
    };
    let rt = blocking::Onnxruntime::load_once()
        .filename(std::env::var("AZALEA_DECODER_ORT")?)
        .perform()?;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(16) as u16);
    eprintln!("using {threads} inference threads");
    let mut failures = Vec::new();
    let mut checked = 0;
    let mut paths = std::fs::read_dir(std::env::var("AZALEA_DECODER_VVM_DIR")?)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    for path in paths
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "vvm"))
    {
        let model = VoiceModelFile::open(&path)?;
        let contents = model.inner().read_inference_models().block_on()?;
        let Some((ids, models)) = &contents.talk else {
            continue;
        };
        let decode = &models[TalkOperation::Decode];
        let result = (|| {
            let options = InferenceSessionOptions::new(threads, DeviceSpec::Cpu);
            let monolithic = Arc::new(rt.new_session(decode, options)?.0);
            let started = Instant::now();
            let split = load_split(rt, decode, threads)?;
            let elapsed = started.elapsed();
            let mut reports = vec![];
            for (i, style) in model
                .metas()
                .iter()
                .flat_map(|m| &m.styles)
                .filter(|s| s.r#type == crate::StyleType::Talk)
                .enumerate()
            {
                let speaker = ids.get(&style.id).map_or(style.id.0, |id| id.raw_id());
                // The vocoder is shared by every style, so chunking is checked once per VVM.
                let report = compare(&monolithic, &split, speaker.into(), i == 0)?;
                ensure!(
                    report.split <= 1e-6,
                    "style {}: split mismatch {}",
                    style.id,
                    report.split
                );
                if let Some(chunked) = report.chunked {
                    ensure!(
                        chunked <= CHUNKED_TOLERANCE,
                        "style {}: chunked mismatch {chunked}",
                        style.id
                    );
                }
                reports.push(report);
            }
            anyhow::Ok((elapsed, reports))
        })();
        match result {
            Ok((elapsed, reports)) => {
                checked += reports.len();
                eprintln!(
                    "{} export+split={elapsed:?} styles={} max split={:e} chunked={:e}",
                    path.display(),
                    reports.len(),
                    reports.iter().map(|r| r.split).fold(0.0, f32::max),
                    reports[0].chunked.unwrap(),
                );
            }
            Err(error) => {
                eprintln!("{} FAIL: {error:#}", path.display());
                failures.push(format!("{}: {error:#}", path.display()));
            }
        }
    }
    ensure!(checked > 0, "no talk decoders found");
    ensure!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("verified {checked} decoder/style combinations");
    Ok(())
}
