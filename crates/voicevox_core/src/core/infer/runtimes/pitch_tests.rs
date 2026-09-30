use super::*;
use crate::{
    AccelerationMode, PitchNoiseOptions, StyleId,
    blocking::{Synthesizer, VoiceModelFile},
    future::FutureExt as _,
};
use std::time::Instant;

fn prediction(
    session: Arc<async_lock::Mutex<ort::session::Session>>,
    speaker: i64,
    noise: Option<&[f32]>,
) -> anyhow::Result<Vec<f32>> {
    let mut ctx = OnnxruntimeRunContext::from(session);
    ctx.push_input("length", ndarray::arr0(5_i64))?;
    for (name, data) in [
        ("vowel_phoneme_list", [0_i64, 14, 6, 30, 0]),
        ("consonant_phoneme_list", [-1, 37, 35, 37, -1]),
        ("start_accent_list", [0, 1, 0, 0, 0]),
        ("end_accent_list", [0, 1, 0, 0, 0]),
        ("start_accent_phrase_list", [0, 1, 0, 0, 0]),
        ("end_accent_phrase_list", [0, 0, 0, 1, 0]),
    ] {
        ctx.push_input(name, ndarray::arr1(&data))?;
    }
    ctx.push_input("speaker_id", ndarray::arr1(&[speaker]))?;
    if let Some(noise) = noise {
        ctx.push_input("azalea_pitch_noise", ndarray::arr1(noise))?;
    }
    let output = blocking::Onnxruntime::run_blocking(ctx)?.remove(0);
    let OutputTensor::Float32(output) = output else {
        bail!("nonfloat pitch output")
    };
    Ok(output.into_iter().collect())
}

fn compare(rt: &blocking::Onnxruntime, model: &ModelBytes, speaker: i64) -> anyhow::Result<()> {
    let options = InferenceSessionOptions::new(1, DeviceSpec::Cpu);
    let base = Arc::new(rt.new_session(model, options)?.0);
    let started = Instant::now();
    let bytes = export_and_transform(model)?;
    let transformation = started.elapsed();
    let transformed = Arc::new(rt.new_session(&ModelBytes::Onnx(bytes), options)?.0);
    let original = prediction(base, speaker, None)?;
    let zero = prediction(transformed.clone(), speaker, Some(&[0.0; 5]))?;
    // Level1 can fuse the new Add differently across ORT builds/CPU kernels.
    let max_error = original
        .iter()
        .zip(&zero)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    ensure!(
        original.iter().chain(&zero).all(|x| x.is_finite()),
        "nonfinite output"
    );
    ensure!(max_error <= 1e-6, "zero-noise mismatch: {max_error}");
    let impulse = prediction(
        transformed.clone(),
        speaker,
        Some(&[0.0, 0.05, 0.0, 0.0, 0.0]),
    )?;
    ensure!(impulse[0] == zero[0], "impulse changed earlier output");
    ensure!(
        (impulse[1] - zero[1] - 0.05).abs() < 1e-6,
        "missing direct impulse"
    );
    ensure!(
        impulse[2..]
            .iter()
            .zip(&zero[2..])
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "no autoregressive propagation"
    );
    let started = Instant::now();
    for _ in 0..20 {
        ensure!(
            prediction(transformed.clone(), speaker, Some(&[0.0; 5]))? == zero,
            "nondeterministic legacy inference"
        );
    }
    eprintln!(
        "transform={transformation:?}, inference_mean={:?}, zero_max_error={max_error}, exact={}",
        started.elapsed() / 20,
        original == zero
    );
    Ok(())
}

#[test]
fn sample_zero_noise_and_impulse_propagation() -> anyhow::Result<()> {
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let bytes = include_bytes!("../../../../../../model/sample.vvm/predict_intonation.onnx");
    compare(rt, &ModelBytes::Onnx(bytes.to_vec()), 0)
}

#[test]
fn pitch_api_parity_preservation_and_legacy_determinism() -> anyhow::Result<()> {
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let synth = Synthesizer::builder(rt)
        .acceleration_mode(AccelerationMode::Cpu)
        .cpu_num_threads(1)
        .build()?;
    let model = VoiceModelFile::open(test_util::SAMPLE_VOICE_MODEL_FILE_PATH)?;
    synth.load_voice_model(&model).perform()?;
    let style = StyleId::new(302);
    let query = synth.create_audio_query_from_kana("テ'_スト、コンニチワ'", style)?;
    let phrases = &query.accent_phrases;
    let options = PitchNoiseOptions {
        sigma: 0.05,
        seed: 42,
    };
    let noisy = synth.replace_mora_pitch_with_noise(phrases, style, options)?;
    assert_eq!(
        synth.replace_mora_pitch_with_noise(phrases, style, options)?,
        noisy
    );
    assert_ne!(
        noisy,
        synth.replace_mora_pitch_with_noise(
            phrases,
            style,
            PitchNoiseOptions {
                seed: 43,
                ..options
            }
        )?
    );
    assert_eq!(synth.replace_mora_pitch(phrases, style)?, *phrases);
    assert_eq!(
        synth.replace_mora_pitch_with_noise(phrases, style, PitchNoiseOptions::default())?,
        *phrases
    );
    let mut restored = noisy.clone();
    for (new, old) in restored.iter_mut().zip(phrases) {
        for (new, old) in new
            .moras
            .iter_mut()
            .chain(new.pause_mora.iter_mut())
            .zip(old.moras.iter().chain(old.pause_mora.iter()))
        {
            if old.pitch == 0.0 {
                assert_eq!(new.pitch, 0.0);
            }
            new.pitch = old.pitch;
        }
    }
    assert_eq!(restored, *phrases);
    let nonblocking =
        crate::nonblocking::Synthesizer::builder(crate::nonblocking::Onnxruntime::get().unwrap())
            .acceleration_mode(AccelerationMode::Cpu)
            .cpu_num_threads(1)
            .build()?;
    let async_model =
        crate::nonblocking::VoiceModelFile::open(test_util::SAMPLE_VOICE_MODEL_FILE_PATH)
            .block_on()?;
    nonblocking
        .load_voice_model(&async_model)
        .perform()
        .block_on()?;
    assert_eq!(
        nonblocking
            .replace_mora_pitch_with_noise(phrases, style, options)
            .block_on()?,
        noisy
    );
    for result in futures_util::future::join_all(
        (0..8).map(|_| nonblocking.replace_mora_pitch_with_noise(phrases, style, options)),
    )
    .block_on()
    {
        assert_eq!(result?, noisy);
    }
    for sigma in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(
            nonblocking
                .replace_mora_pitch_with_noise(
                    phrases,
                    StyleId::new(u32::MAX),
                    PitchNoiseOptions { sigma, seed: 0 }
                )
                .block_on()
                .unwrap_err()
                .to_string()
                .contains("sigma")
        );
    }
    Ok(())
}

#[test]
fn unsupported_pitch_does_not_register_partial_model() -> anyhow::Result<()> {
    use crate::core::{
        infer::domains::{InferenceDomainMap, TalkOperation},
        status::Status,
    };
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let options = InferenceSessionOptions::new(1, DeviceSpec::Cpu);
    let status = Status::new(
        rt,
        InferenceDomainMap {
            talk: enum_map::enum_map!(_ => options),
            experimental_talk: enum_map::enum_map!(_ => options),
            singing_teacher: enum_map::enum_map!(_ => options),
            frame_decode: enum_map::enum_map!(_ => options),
        },
    );
    let model = VoiceModelFile::open(test_util::SAMPLE_VOICE_MODEL_FILE_PATH)?;
    let mut contents = model.inner().read_inference_models().block_on()?;
    contents.talk.as_mut().unwrap().1[TalkOperation::PredictIntonation] = ModelBytes::Onnx(
        include_bytes!("../../../../../../model/sample.vvm/predict_duration.onnx").to_vec(),
    );
    let error = status
        .insert_model(model.inner().header(), &contents, Default::default())
        .unwrap_err();
    let error = format!("{:#}", anyhow::Error::from(error));
    assert!(
        error.contains("TalkDomain") && error.contains("pitch") && error.contains("sample.vvm"),
        "{error}"
    );
    assert!(!status.is_loaded_model(model.id()));
    assert!(status.metas().is_empty());
    let mut valid = model.inner().read_inference_models().block_on()?;
    valid.talk = None;
    status.insert_model(model.inner().header(), &valid, Default::default())?;
    let metas = status.metas();
    assert!(
        status
            .insert_model(
                model.inner().header(),
                &contents,
                crate::OnExistingVoiceModelId::Reload
            )
            .is_err()
    );
    assert!(status.is_loaded_model(model.id()));
    assert_eq!(
        serde_json::to_value(status.metas())?,
        serde_json::to_value(metas)?
    );
    // Exercise the experimental-only input signature after successful registration.
    let output = status
        .run_session::<crate::asyncs::SingleTasked, _>(
            model.id(),
            crate::core::infer::domains::experimental_talk::PredictIntonationInput {
                length: ndarray::arr0(5),
                vowel_phoneme_list: ndarray::arr1(&[0, 14, 6, 30, 0]),
                consonant_phoneme_list: ndarray::arr1(&[-1, 37, 35, 37, -1]),
                start_accent_list: ndarray::arr1(&[0, 1, 0, 0, 0]),
                end_accent_list: ndarray::arr1(&[0, 1, 0, 0, 0]),
                start_accent_phrase_list: ndarray::arr1(&[0, 1, 0, 0, 0]),
                end_accent_phrase_list: ndarray::arr1(&[0, 0, 0, 1, 0]),
                speaker_id: ndarray::arr1(&[0]),
                azalea_pitch_noise: ndarray::arr1(&[0.0, 0.05, 0.0, 0.0, 0.0]),
            },
            (),
        )
        .block_on()?;
    assert_eq!(output.f0_list.len(), 5);
    assert!(output.f0_list.iter().all(|x| x.is_finite()));
    Ok(())
}

/// Explicit local diagnostic; production models are never fixtures or exported persistently.
#[test]
#[ignore = "set AZALEA_PITCH_ORT and AZALEA_PITCH_VVM_DIR; run in a separate test process"]
fn installed_pitch_diagnostic() -> anyhow::Result<()> {
    use crate::core::infer::domains::{ExperimentalTalkOperation, TalkOperation};
    let rt = blocking::Onnxruntime::load_once()
        .filename(std::env::var("AZALEA_PITCH_ORT")?)
        .perform()?;
    let mut failures = Vec::new();
    let mut checked = 0;
    let mut paths = std::fs::read_dir(std::env::var("AZALEA_PITCH_VVM_DIR")?)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    for path in paths
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "vvm"))
    {
        let model = VoiceModelFile::open(&path)?;
        let contents = model.inner().read_inference_models().block_on()?;
        for (domain, pair) in [
            (
                "talk",
                contents
                    .talk
                    .as_ref()
                    .map(|(ids, models)| (ids, &models[TalkOperation::PredictIntonation])),
            ),
            (
                "experimental_talk",
                contents.experimental_talk.as_ref().map(|(ids, models)| {
                    (ids, &models[ExperimentalTalkOperation::PredictIntonation])
                }),
            ),
        ] {
            let Some((ids, bytes)) = pair else { continue };
            for style in model
                .metas()
                .iter()
                .flat_map(|m| &m.styles)
                .filter(|s| s.r#type == crate::StyleType::Talk)
            {
                let speaker = ids.get(&style.id).map_or(style.id.0, |id| id.raw_id());
                eprint!("{} {domain} style={} ", path.display(), style.id);
                checked += 1;
                if let Err(error) = compare(rt, bytes, speaker.into()) {
                    eprintln!("FAIL: {error:#}");
                    failures.push(format!(
                        "{} {domain} {}: {error:#}",
                        path.display(),
                        style.id
                    ));
                }
            }
        }
        // Also exercise ordinary, atomic VVM loading with all domains.
        let synth = Synthesizer::builder(rt)
            .acceleration_mode(AccelerationMode::Cpu)
            .cpu_num_threads(1)
            .build()?;
        if let Err(error) = synth.load_voice_model(&model).perform() {
            failures.push(format!("{} ordinary load: {error}", path.display()));
        }
    }
    ensure!(checked > 0, "no talk predictors found");
    ensure!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!("verified {checked} predictor/style combinations");
    Ok(())
}

#[test]
fn nested_selection_runs_both_branches() -> anyhow::Result<()> {
    use onnx_protobuf::{
        AttributeProto, Message as _, ModelProto, NodeProto, TensorProto,
        attribute_proto::AttributeType,
    };
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let mut model = ModelProto::parse_from_bytes(include_bytes!(
        "../../../../../../model/sample.vvm/predict_intonation.onnx"
    ))?;
    let graph = model.graph.as_mut().unwrap();
    let mut branch = graph.clone();
    branch.input.clear();
    for _ in 0..2 {
        branch.node = vec![NodeProto {
            op_type: "If".into(),
            input: vec!["choose_model".into()],
            output: vec!["f0_list".into()],
            attribute: ["then_branch", "else_branch"]
                .into_iter()
                .map(|name| AttributeProto {
                    name: name.into(),
                    type_: AttributeType::GRAPH.into(),
                    g: Some(branch.clone()).into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }];
        branch.initializer.clear();
        branch.value_info.clear();
    }
    graph.node = vec![NodeProto {
        op_type: "Equal".into(),
        input: vec!["speaker_id".into(), "zero_speaker".into()],
        output: vec!["choose_model".into()],
        ..Default::default()
    }];
    graph.node.extend(branch.node);
    graph.initializer = vec![TensorProto {
        name: "zero_speaker".into(),
        data_type: 7,
        dims: vec![1],
        int64_data: vec![0],
        ..Default::default()
    }];
    graph.value_info.clear();
    let model = ModelBytes::Onnx(model.write_to_bytes()?);
    compare(rt, &model, 0)?;
    compare(rt, &model, 1)
}

#[test]
fn sequence_collection_feedback_and_rejection() -> anyhow::Result<()> {
    use onnx_protobuf::{
        AttributeProto, Message as _, ModelProto, NodeProto, TypeProto, ValueInfoProto,
        attribute_proto::AttributeType, type_proto,
    };
    let rt = blocking::Onnxruntime::from_test_util_data()?;
    let mut model = ModelProto::parse_from_bytes(include_bytes!(
        "../../../../../../model/sample.vvm/predict_intonation.onnx"
    ))?;
    let graph = model.graph.as_mut().unwrap();
    let loop_index = graph.node.iter().position(|n| n.op_type == "Loop").unwrap();
    let node = &mut graph.node[loop_index];
    let body = node.attribute[0].g.as_mut().unwrap();
    let mut sequence_type = TypeProto::new();
    sequence_type.set_sequence_type(type_proto::Sequence {
        elem_type: body.output[3].type_.clone(),
        ..Default::default()
    });
    let sequence_input = ValueInfoProto {
        name: "collected_pitches".into(),
        type_: Some(sequence_type.clone()).into(),
        ..Default::default()
    };
    let sequence_output = ValueInfoProto {
        name: "next_pitches".into(),
        type_: Some(sequence_type).into(),
        ..Default::default()
    };
    let scanned = body.output.pop().unwrap();
    body.node.push(NodeProto {
        op_type: "SequenceInsert".into(),
        input: vec![sequence_input.name.clone(), scanned.name],
        output: vec![sequence_output.name.clone()],
        ..Default::default()
    });
    body.input.insert(2, sequence_input);
    body.output.insert(1, sequence_output);
    // Newer exporters leave the batch/time axes symbolic on the Conv output.
    let pitch_type = body.output[2].type_.as_mut().unwrap().mut_tensor_type();
    pitch_type.shape.as_mut().unwrap().dim[0].set_dim_param("batch".into());
    pitch_type.shape.as_mut().unwrap().dim[2].set_dim_param("time".into());
    node.input.insert(2, "empty_pitches".into());
    let scan_output = node.output.pop().unwrap();
    node.output.insert(0, "pitch_sequence".into());
    graph.node.insert(
        loop_index,
        NodeProto {
            op_type: "SequenceEmpty".into(),
            output: vec!["empty_pitches".into()],
            attribute: vec![AttributeProto {
                name: "dtype".into(),
                type_: AttributeType::INT.into(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    graph.node.insert(
        loop_index + 2,
        NodeProto {
            op_type: "ConcatFromSequence".into(),
            input: vec!["pitch_sequence".into()],
            output: vec![scan_output],
            attribute: [("axis", 0), ("new_axis", 1)]
                .into_iter()
                .map(|(name, i)| AttributeProto {
                    name: name.into(),
                    type_: AttributeType::INT.into(),
                    i,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        },
    );
    // Loop-13 adds sequence carries. Upgrade axes encoding in the public opset-12 fixture.
    model
        .opset_import
        .iter_mut()
        .find(|op| op.domain.is_empty())
        .unwrap()
        .version = 13;
    fn axes_inputs(graph: &mut onnx_protobuf::GraphProto) {
        for (index, node) in graph.node.iter_mut().enumerate() {
            if matches!(node.op_type.as_str(), "Squeeze" | "Unsqueeze" | "ReduceSum") {
                if let Some(at) = node.attribute.iter().position(|a| a.name == "axes") {
                    let axes = node.attribute.remove(at).ints;
                    let name = format!("fixture_axes_{index}");
                    graph.initializer.push(onnx_protobuf::TensorProto {
                        name: name.clone(),
                        data_type: 7,
                        dims: vec![axes.len() as i64],
                        int64_data: axes,
                        ..Default::default()
                    });
                    node.input.push(name);
                }
            }
            for attribute in &mut node.attribute {
                if let Some(graph) = attribute.g.as_mut() {
                    axes_inputs(graph);
                }
            }
        }
    }
    axes_inputs(model.graph.as_mut().unwrap());
    compare(rt, &ModelBytes::Onnx(model.write_to_bytes()?), 0)?;
    let body = model.graph.as_mut().unwrap().node[loop_index + 1].attribute[0]
        .g
        .as_mut()
        .unwrap();
    body.node.last_mut().unwrap().input[0] = "wrong_sequence".into();
    assert!(super::super::pitch_graph::transform(&model.write_to_bytes()?).is_err());
    Ok(())
}
