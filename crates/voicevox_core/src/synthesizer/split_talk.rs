//! Inference through [`SplitTalkDomain`] (`azalea` fork), mirroring the experimental-talk paths.
use crate::{
    Result, StyleId,
    core::{
        Array1ExtForPostProcess as _, ensure_minimum_phoneme_length,
        infer::{
            self, InferenceRuntime,
            domains::{SplitTalkDomain, split_talk},
        },
        pad_decoder_feature,
        status::Status,
    },
};

use super::{MARGIN, PADDING_FRAME_LENGTH, finite_pitch};

impl<R: InferenceRuntime> Status<R> {
    pub(super) async fn split_talk_predict_duration<A: infer::AsyncExt>(
        &self,
        phoneme_vector: ndarray::Array1<i64>,
        style_id: StyleId,
    ) -> Result<Vec<f32>> {
        let (model_id, inner_voice_id) = self.ids_for::<SplitTalkDomain>(style_id)?;
        let split_talk::PredictDurationOutput {
            phoneme_length: output,
        } = self
            .run_session::<A, _>(
                model_id,
                split_talk::PredictDurationInput {
                    phoneme_list: phoneme_vector,
                    speaker_id: ndarray::arr1(&[inner_voice_id.raw_id().into()]),
                },
                A::LIGHT_INFERENCE_CANCELLABLE,
            )
            .await?;
        Ok(ensure_minimum_phoneme_length(output.into_vec()))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "same arguments as `Status::predict_intonation`"
    )]
    pub(super) async fn split_talk_predict_intonation<A: infer::AsyncExt>(
        &self,
        length: usize,
        vowel_phoneme_vector: ndarray::Array1<i64>,
        consonant_phoneme_vector: ndarray::Array1<i64>,
        start_accent_vector: ndarray::Array1<i64>,
        end_accent_vector: ndarray::Array1<i64>,
        start_accent_phrase_vector: ndarray::Array1<i64>,
        end_accent_phrase_vector: ndarray::Array1<i64>,
        style_id: StyleId,
        noise: ndarray::Array1<f32>,
        prefix: ndarray::Array1<f32>,
        prefix_mask: ndarray::Array1<i64>,
    ) -> Result<Vec<f32>> {
        let (model_id, inner_voice_id) = self.ids_for::<SplitTalkDomain>(style_id)?;
        let split_talk::PredictIntonationOutput { f0_list: output } = self
            .run_session::<A, _>(
                model_id,
                split_talk::PredictIntonationInput {
                    length: ndarray::arr0(length as i64),
                    vowel_phoneme_list: vowel_phoneme_vector,
                    consonant_phoneme_list: consonant_phoneme_vector,
                    start_accent_list: start_accent_vector,
                    end_accent_list: end_accent_vector,
                    start_accent_phrase_list: start_accent_phrase_vector,
                    end_accent_phrase_list: end_accent_phrase_vector,
                    speaker_id: ndarray::arr1(&[inner_voice_id.raw_id().into()]),
                    azalea_pitch_noise: noise,
                    azalea_pitch_prefix: prefix,
                    azalea_pitch_prefix_mask: prefix_mask,
                },
                A::LIGHT_INFERENCE_CANCELLABLE,
            )
            .await?;
        finite_pitch(output.into_vec(), length)
    }

    /// Same contract as `Status::generate_full_intermediate`: the returned features keep
    /// `MARGIN` frames of context on both ends.
    pub(super) async fn split_talk_generate_full_intermediate<A: infer::AsyncExt>(
        &self,
        length: usize,
        phoneme_size: usize,
        f0: ndarray::Array1<f32>,
        phoneme_vector: ndarray::Array1<f32>,
        style_id: StyleId,
    ) -> Result<ndarray::Array2<f32>> {
        let (model_id, inner_voice_id) = self.ids_for::<SplitTalkDomain>(style_id)?;
        let (length_with_padding, f0_with_padding, phoneme_with_padding) =
            pad_decoder_feature::<PADDING_FRAME_LENGTH>(
                f0,
                phoneme_vector
                    .into_shape_with_order([length, phoneme_size])
                    .unwrap(),
            );
        let split_talk::GenerateFullIntermediateOutput {
            spec: spec_with_padding,
        } = self
            .run_session::<A, _>(
                model_id,
                split_talk::GenerateFullIntermediateInput {
                    f0: f0_with_padding
                        .into_shape_with_order([length_with_padding, 1])
                        .unwrap(),
                    phoneme: phoneme_with_padding,
                    speaker_id: ndarray::arr1(&[inner_voice_id.raw_id().into()]),
                },
                A::LIGHT_INFERENCE_CANCELLABLE,
            )
            .await?;
        const _: () = assert!(MARGIN <= PADDING_FRAME_LENGTH);
        Ok(spec_with_padding
            .slice(ndarray::s![
                PADDING_FRAME_LENGTH - MARGIN
                    ..spec_with_padding.nrows() - PADDING_FRAME_LENGTH + MARGIN,
                ..
            ])
            .to_owned())
    }

    pub(super) async fn split_talk_render_audio_segment<A: infer::AsyncExt>(
        &self,
        spec: ndarray::Array2<f32>,
        style_id: StyleId,
    ) -> Result<ndarray::Array1<f32>> {
        let (model_id, _inner_voice_id) = self.ids_for::<SplitTalkDomain>(style_id)?;
        let split_talk::RenderAudioSegmentOutput { wave } = self
            .run_session::<A, _>(
                model_id,
                split_talk::RenderAudioSegmentInput { spec },
                A::DEFAULT_HEAVY_INFERENCE_CANCELLABLE,
            )
            .await?;
        Ok(wave)
    }
}

#[cfg(test)]
mod tests {
    use enum_map::enum_map;
    use onnx_protobuf::{Message as _, ModelProto, NodeProto};

    use crate::{
        StyleType,
        asyncs::SingleTasked,
        blocking::VoiceModelFile,
        core::{
            devices::DeviceSpec,
            infer::{
                InferenceSessionOptions,
                domains::{InferenceDomainMap, TalkDomain, TalkOperation},
            },
            voice_model::ModelBytes,
        },
        engine::PhonemeCode,
        future::FutureExt as _,
    };

    use super::{super::super::blocking::Onnxruntime, *};

    fn status() -> anyhow::Result<Status<Onnxruntime>> {
        let options = InferenceSessionOptions::new(1, DeviceSpec::Cpu);
        Ok(Status::new(
            Onnxruntime::from_test_util_data()?,
            InferenceDomainMap {
                talk: enum_map! { _ => options },
                experimental_talk: enum_map! { _ => options },
                singing_teacher: enum_map! { _ => options },
                frame_decode: enum_map! { _ => options },
            },
        ))
    }

    fn talk_style(model: &VoiceModelFile) -> StyleId {
        model
            .metas()
            .iter()
            .flat_map(|c| &c.styles)
            .find(|s| s.r#type == StyleType::Talk)
            .unwrap()
            .id
    }

    /// Unpadded decoder features, as produced from an audio query.
    fn features(length: usize) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
        let f0 = ndarray::Array1::from_shape_fn(length, |i| {
            if (i / 17) % 5 == 0 {
                0.0
            } else {
                5.5 + 0.2 * (i as f32 / 20.0).sin()
            }
        });
        let size = PhonemeCode::num_phoneme();
        let phoneme = ndarray::Array1::from_shape_fn(length * size, |i| {
            f32::from(u8::from(i % size == (i / size / 9) % (size - 1) + 1))
        });
        (f0, phoneme)
    }

    fn decode(status: &Status<Onnxruntime>, style_id: StyleId, length: usize) -> Result<Vec<f32>> {
        let (f0, phoneme) = features(length);
        status
            .decode::<SingleTasked>(
                length,
                PhonemeCode::num_phoneme(),
                f0,
                phoneme,
                style_id,
                (),
            )
            .block_on()
    }

    fn max_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn talk_only_model_is_split_and_matches_monolithic_talk() -> anyhow::Result<()> {
        let model = VoiceModelFile::open(::test_util::SAMPLE_VOICE_MODEL_FILE_PATH)?;
        let style_id = talk_style(&model);

        // `talk` takes priority over the native split decoder, so this is the monolithic path.
        let reference = status()?;
        let contents = model.inner().read_inference_models().block_on()?;
        reference.insert_model(model.inner().header(), &contents, Default::default())?;
        assert!(reference.contains_domain::<TalkDomain>(style_id));
        assert!(!reference.contains_domain::<SplitTalkDomain>(style_id));

        let split = status()?;
        let mut contents = model.inner().read_inference_models().block_on()?;
        contents.experimental_talk = None;
        split.insert_model(model.inner().header(), &contents, Default::default())?;
        assert!(split.contains_domain::<SplitTalkDomain>(style_id));
        assert!(!split.contains_domain::<TalkDomain>(style_id));

        let phonemes = ndarray::arr1(&[0_i64, 37, 14, 35, 6, 37, 30, 0]);
        assert_eq!(
            reference
                .predict_duration::<SingleTasked>(phonemes.clone(), style_id)
                .block_on()?,
            split
                .predict_duration::<SingleTasked>(phonemes, style_id)
                .block_on()?,
        );

        let length = 150;
        let expected = decode(&reference, style_id, length)?;
        let actual = decode(&split, style_id, length)?;
        // Level1 may fuse kernels differently across ORT builds; observed results are bitwise.
        let error = max_error(&expected, &actual);
        assert!(error <= 1e-6, "split decode mismatch {error}");

        // Streaming: chunks rendered with `MARGIN` context reassemble the utterance.
        let (f0, phoneme) = features(length);
        let spec = split
            .generate_full_intermediate::<SingleTasked>(
                length,
                PhonemeCode::num_phoneme(),
                f0,
                phoneme,
                style_id,
            )
            .block_on()?;
        assert_eq!(spec.nrows(), length + 2 * MARGIN);
        let mut streamed = vec![];
        for start in (0..length).step_by(50) {
            let end = (start + 50).min(length);
            let segment = spec
                .slice(ndarray::s![start..end + 2 * MARGIN, ..])
                .to_owned();
            let wave = split
                .render_audio_segment::<SingleTasked>(segment, style_id)
                .block_on()?;
            streamed.extend_from_slice(
                &wave.as_slice().unwrap()[MARGIN * 256..wave.len() - MARGIN * 256],
            );
        }
        let error = max_error(&expected, &streamed);
        assert!(error <= 2e-3, "streamed mismatch {error}");
        Ok(())
    }

    /// An unsupported decoder keeps the ordinary talk domain instead of failing the load.
    #[test]
    fn unsplittable_decoder_falls_back_to_talk() -> anyhow::Result<()> {
        let model = VoiceModelFile::open(::test_util::SAMPLE_VOICE_MODEL_FILE_PATH)?;
        let style_id = talk_style(&model);
        let mut contents = model.inner().read_inference_models().block_on()?;
        contents.experimental_talk = None;
        let (_, models) = contents.talk.as_mut().unwrap();
        let ModelBytes::Onnx(bytes) = &models[TalkOperation::Decode] else {
            panic!("sample decoder should be ONNX");
        };
        // An `Identity` between the transpose and unsqueeze hides the bottleneck pattern
        // without changing what the decoder computes.
        let mut decoder = ModelProto::parse_from_bytes(bytes)?;
        let graph = decoder.graph.as_mut().unwrap();
        let unsqueeze = graph
            .node
            .iter()
            .position(|n| {
                n.op_type == "Unsqueeze"
                    && graph.node.iter().any(|c| {
                        c.input.first() == n.output.first()
                            && c.input
                                .iter()
                                .any(|i| i == "hifi_gan_forwarder.conv_pre.weight")
                    })
            })
            .unwrap();
        let input = std::mem::replace(&mut graph.node[unsqueeze].input[0], "hidden".into());
        graph.node.insert(
            unsqueeze,
            NodeProto {
                op_type: "Identity".into(),
                input: vec![input],
                output: vec!["hidden".into()],
                ..Default::default()
            },
        );
        models[TalkOperation::Decode] = ModelBytes::Onnx(decoder.write_to_bytes()?);

        let status = status()?;
        status.insert_model(model.inner().header(), &contents, Default::default())?;
        assert!(status.contains_domain::<TalkDomain>(style_id));
        assert!(!status.contains_domain::<SplitTalkDomain>(style_id));
        assert_eq!(decode(&status, style_id, 60)?.len(), 60 * 256);
        Ok(())
    }
}
