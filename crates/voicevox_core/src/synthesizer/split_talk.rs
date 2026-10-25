//! Inference through [`SplitTalkDomain`] (`azalea` fork), mirroring the streaming-talk paths,
//! and the fork's public streaming API on top of it.
use std::ops::Range;

use crate::{
    AudioQuery, Result, StyleId,
    core::{
        Array1ExtForPostProcess as _, ensure_minimum_phoneme_length,
        infer::{
            self, InferenceRuntime,
            domains::{SplitTalkDomain, StreamingTalkDomain, split_talk},
        },
        pad_decoder_feature,
        status::Status,
    },
    engine::wav_from_s16le,
    frame::FRAME_RATE,
};

use typed_floats::{NonNaNFinite, PositiveFinite};

use super::{AsInner as _, AudioFeature, SynthesisOptions, finish_intermediate, finite_pitch};

/// Frames per second of [`StreamingAudio`]: one frame is 256 samples at 24 kHz.
pub const STREAMING_FRAME_RATE: f64 = FRAME_RATE;

/// Default chunk size for [`StreamingAudio::chunk_ranges`]. Chunks of at least this many
/// frames have rendered bit-identically to whole-utterance rendering on production models.
pub const DEFAULT_STREAMING_CHUNK_FRAMES: usize = 50;

/// Options for preparing a [`StreamingAudio`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingOptions {
    /// Same as the synthesis option of the same name.
    pub enable_interrogative_upspeak: bool,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            enable_interrogative_upspeak: super::DEFAULT_ENABLE_INTERROGATIVE_UPSPEAK,
        }
    }
}

/// An utterance whose acoustic features are computed, ready to be rendered in frame ranges
/// (`azalea` fork).
///
/// Create it with `prepare_streaming` and render it with `render_streaming` on the
/// synthesizer that prepared it. Rendering every range of [`Self::chunk_ranges`] in order
/// and concatenating the PCM gives the same audio as `synthesis` (without the WAV header).
#[derive(Clone, PartialEq, Debug)]
pub struct StreamingAudio {
    feature: AudioFeature,
    output_sampling_rate: u32,
    output_stereo: bool,
}

impl StreamingAudio {
    /// Number of frames, at [`STREAMING_FRAME_RATE`].
    pub fn frame_length(&self) -> usize {
        self.feature.len()
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.frame_length() as f64 / STREAMING_FRAME_RATE
    }

    /// Style the features were computed with.
    pub fn style_id(&self) -> StyleId {
        self.feature.slice(..).style_id()
    }

    /// Output sampling rate of the rendered PCM, from the audio query.
    pub fn output_sampling_rate(&self) -> u32 {
        self.output_sampling_rate
    }

    /// Whether the rendered PCM is stereo, from the audio query.
    pub fn output_stereo(&self) -> bool {
        self.output_stereo
    }

    /// Consecutive frame ranges of at most `chunk_frames` frames covering the utterance.
    /// A `chunk_frames` of zero is treated as one.
    pub fn chunk_ranges(&self, chunk_frames: usize) -> impl Iterator<Item = Range<usize>> + use<> {
        let length = self.frame_length();
        let chunk_frames = chunk_frames.max(1);
        (0..length)
            .step_by(chunk_frames)
            .map(move |start| start..(start + chunk_frames).min(length))
    }

    /// Wraps rendered 16-bit little-endian PCM in a WAV header for this utterance's format.
    pub fn wav(&self, pcm: &[u8]) -> Vec<u8> {
        wav_from_s16le(pcm, self.output_sampling_rate, self.output_stereo)
    }

    /// `range` clamped to the utterance, so rendering never panics.
    fn clamp(&self, range: Range<usize>) -> Range<usize> {
        let end = range.end.min(self.frame_length());
        range.start.min(end)..end
    }
}

impl<T, A: super::AsyncExt> super::Inner<T, A> {
    fn supports_streaming(&self, style_id: StyleId) -> bool {
        let status = self.status();
        status.contains_domain::<SplitTalkDomain>(style_id)
            || status.contains_domain::<StreamingTalkDomain>(style_id)
    }

    async fn prepare_streaming(
        &self,
        audio_query: &AudioQuery,
        style_id: StyleId,
        options: StreamingOptions,
    ) -> Result<StreamingAudio> {
        let options = SynthesisOptions::<A> {
            enable_interrogative_upspeak: options.enable_interrogative_upspeak,
            ..Default::default()
        };
        let feature = self
            .create_audio_feature(audio_query, style_id, &options)
            .await?;
        Ok(StreamingAudio {
            feature,
            output_sampling_rate: audio_query.output_sampling_rate.get().get(),
            output_stereo: audio_query.output_stereo,
        })
    }

    async fn render_streaming(
        &self,
        audio: &StreamingAudio,
        range: Range<usize>,
    ) -> Result<Vec<u8>> {
        let range = audio.clamp(range);
        if range.is_empty() {
            return Ok(Vec::new());
        }
        self.render(audio.feature.slice(range)).await
    }
}

impl<T> super::blocking::Synthesizer<T> {
    /// Whether `style_id` can be rendered in segments: its model has a split decoder,
    /// either shipped natively or split while loading (`azalea` fork).
    pub fn supports_streaming(&self, style_id: StyleId) -> bool {
        self.0.supports_streaming(style_id)
    }

    /// Computes the acoustic features of a whole utterance for segmented rendering
    /// (`azalea` fork). Most of the synthesis time is left to [`Self::render_streaming`].
    ///
    /// Fails when the style's model has no split decoder; see [`Self::supports_streaming`].
    pub fn prepare_streaming(
        &self,
        audio_query: &AudioQuery,
        style_id: StyleId,
        options: StreamingOptions,
    ) -> Result<StreamingAudio> {
        use crate::future::FutureExt as _;
        self.0
            .prepare_streaming(audio_query, style_id, options)
            .block_on()
    }

    /// Renders the frames in `range` as 16-bit little-endian PCM (`azalea` fork).
    ///
    /// `range` is clamped to the utterance; an empty range renders nothing. Each segment is
    /// vocoded with surrounding context, so consecutive ranges join seamlessly.
    pub fn render_streaming(&self, audio: &StreamingAudio, range: Range<usize>) -> Result<Vec<u8>> {
        use crate::future::FutureExt as _;
        self.0.render_streaming(audio, range).block_on()
    }

    /// Renders `audio` lazily in chunks of `chunk_frames` frames (`azalea` fork).
    ///
    /// See [`DEFAULT_STREAMING_CHUNK_FRAMES`] for a chunk size that keeps the output
    /// identical to whole-utterance rendering.
    pub fn render_streaming_chunks<'a>(
        &'a self,
        audio: &'a StreamingAudio,
        chunk_frames: usize,
    ) -> impl Iterator<Item = Result<Vec<u8>>> + 'a {
        audio
            .chunk_ranges(chunk_frames)
            .map(move |range| self.render_streaming(audio, range))
    }
}

impl<T> super::nonblocking::Synthesizer<T> {
    /// Whether `style_id` can be rendered in segments: its model has a split decoder,
    /// either shipped natively or split while loading (`azalea` fork).
    pub fn supports_streaming(&self, style_id: StyleId) -> bool {
        self.0.supports_streaming(style_id)
    }

    /// Computes the acoustic features of a whole utterance for segmented rendering
    /// (`azalea` fork). Inference runs on the blocking thread pool.
    ///
    /// Fails when the style's model has no split decoder; see [`Self::supports_streaming`].
    pub async fn prepare_streaming(
        &self,
        audio_query: &AudioQuery,
        style_id: StyleId,
        options: StreamingOptions,
    ) -> Result<StreamingAudio> {
        self.0
            .prepare_streaming(audio_query, style_id, options)
            .await
    }

    /// Renders the frames in `range` as 16-bit little-endian PCM (`azalea` fork).
    /// Inference runs on the blocking thread pool; drop the future to stop streaming.
    ///
    /// `range` is clamped to the utterance; an empty range renders nothing. Each segment is
    /// vocoded with surrounding context, so consecutive ranges join seamlessly.
    pub async fn render_streaming(
        &self,
        audio: &StreamingAudio,
        range: Range<usize>,
    ) -> Result<Vec<u8>> {
        self.0.render_streaming(audio, range).await
    }
}

impl<R: InferenceRuntime> Status<R> {
    pub(super) async fn split_talk_predict_duration<A: infer::AsyncExt>(
        &self,
        phoneme_vector: ndarray::Array1<i64>,
        style_id: StyleId,
    ) -> Result<Vec<PositiveFinite<f32>>> {
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
        let output = crate::core::ensure_non_nan_finite(&output.into_vec(), |invalid| {
            anyhow::anyhow!("`predict_duration` returned an array that contains: {invalid}")
        })?;
        Ok(ensure_minimum_phoneme_length(output))
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
    ) -> Result<Vec<NonNaNFinite<f32>>> {
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
    ) -> Result<Box<[[f32; crate::dim::AUDIO_FEATURE_BINS]]>> {
        let (model_id, inner_voice_id) = self.ids_for::<SplitTalkDomain>(style_id)?;
        let (length_with_padding, f0_with_padding, phoneme_with_padding) = pad_decoder_feature(
            f0,
            phoneme_vector
                .into_shape_with_order([length, phoneme_size])
                .unwrap(),
        );
        let f0_with_padding_len = f0_with_padding.len();
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
        finish_intermediate(spec_with_padding, f0_with_padding_len)
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
    use crate::core::adjust::MARGIN;
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

    use super::{super::super::blocking::Onnxruntime, super::AsInner as _, *};

    fn status() -> anyhow::Result<Status<Onnxruntime>> {
        let options = InferenceSessionOptions::new(1, DeviceSpec::Cpu);
        Ok(Status::new(
            Onnxruntime::from_test_util_data()?,
            InferenceDomainMap {
                talk: enum_map! { _ => options },
                streaming_talk: enum_map! { _ => options },
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
            .find(|s| s.id == StyleId(303) && s.r#type == StyleType::Talk)
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
        contents.streaming_talk = None;
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
        assert_eq!(spec.len(), length + 2 * MARGIN);
        let mut streamed = vec![];
        for start in (0..length).step_by(50) {
            let end = (start + 50).min(length);
            let segment = ndarray::ArrayView2::from(&spec[start..end + 2 * MARGIN]).to_owned();
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
        contents.streaming_talk = None;
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

    /// The sample VVM without its native split decoder, so loading splits `talk.decode`.
    fn talk_only_sample() -> anyhow::Result<(tempfile::TempDir, std::path::PathBuf)> {
        use std::io::{Read as _, Write as _};
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("talk_only.vvm");
        let source = std::fs::File::open(::test_util::SAMPLE_VOICE_MODEL_FILE_PATH)?;
        let mut source = zip::ZipArchive::new(source)?;
        let mut target = zip::ZipWriter::new(std::fs::File::create(&path)?);
        for i in 0..source.len() {
            let mut entry = source.by_index(i)?;
            let name = entry.name().to_owned();
            let mut bytes = vec![];
            entry.read_to_end(&mut bytes)?;
            if name == "manifest.json" {
                let mut manifest = serde_json::from_slice::<serde_json::Value>(&bytes)?;
                manifest
                    .as_object_mut()
                    .unwrap()
                    .remove("streaming_talk")
                    .unwrap();
                bytes = serde_json::to_vec(&manifest)?;
            }
            if name == "metas.json" {
                let mut metas = serde_json::from_slice::<serde_json::Value>(&bytes)?;
                for character in metas.as_array_mut().unwrap() {
                    for style in character["styles"].as_array_mut().unwrap() {
                        if style["type"] == "streaming_talk" {
                            style["type"] = "talk".into();
                        }
                    }
                }
                bytes = serde_json::to_vec(&metas)?;
            }
            target.start_file(name, zip::write::FileOptions::default())?;
            target.write_all(&bytes)?;
        }
        target.finish()?;
        Ok((dir, path))
    }

    fn samples(pcm: &[u8]) -> Vec<i16> {
        pcm.chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    #[test]
    fn streaming_api_matches_synthesis_for_split_talk() -> anyhow::Result<()> {
        use crate::{AccelerationMode, blocking::Synthesizer};

        let (_dir, path) = talk_only_sample()?;
        let synthesizer = Synthesizer::builder(Onnxruntime::from_test_util_data()?)
            .acceleration_mode(AccelerationMode::Cpu)
            .cpu_num_threads(1)
            .build()?;
        let model = VoiceModelFile::open(&path)?;
        let style_id = talk_style(&model);
        assert!(!synthesizer.supports_streaming(style_id));
        assert!(
            synthesizer
                .prepare_streaming(&Default::default(), style_id, Default::default())
                .is_err()
        );
        synthesizer.load_voice_model(&model).perform()?;
        assert!(
            synthesizer
                .0
                .status()
                .contains_domain::<SplitTalkDomain>(style_id)
        );
        assert!(synthesizer.supports_streaming(style_id));

        let mut query =
            synthesizer.create_audio_query_from_kana("テ'_スト、コンニチワ'", style_id)?;
        // Per-sample PCM options must survive chunking.
        query.output_sampling_rate = crate::SamplingRate::new(48000)?;
        query.output_stereo = true;
        query.volume_scale = 0.8.try_into().unwrap();
        let wav = synthesizer.synthesis(&query, style_id).perform()?;

        let audio = synthesizer.prepare_streaming(&query, style_id, Default::default())?;
        assert_eq!(audio.style_id(), style_id);
        assert_eq!(audio.output_sampling_rate(), 48000);
        assert!(audio.output_stereo());
        let pcm = synthesizer
            .render_streaming_chunks(&audio, crate::DEFAULT_STREAMING_CHUNK_FRAMES)
            .collect::<Result<Vec<_>>>()?
            .concat();
        // Upstream's lazy API and audio-feature slices must use the same split domain.
        let upstream_wav = synthesizer
            .streaming_synthesis(&query, style_id)
            .segment_length(crate::DEFAULT_STREAMING_CHUNK_FRAMES as f64 / FRAME_RATE)
            .perform()?
            .collect::<Result<Vec<_>>>()?
            .concat();
        assert_eq!(upstream_wav, audio.wav(&pcm));
        let feature = synthesizer
            .create_audio_feature(&query, style_id)
            .perform()?;
        assert_eq!(feature.len(), audio.frame_length());
        assert_eq!(
            synthesizer.render(feature.slice(..))?,
            synthesizer.render_streaming(&audio, 0..audio.frame_length())?
        );
        let streamed = audio.wav(&pcm);
        assert_eq!(streamed.len(), wav.len());
        assert_eq!(streamed[..44], wav[..44], "WAV header");
        // Up to one int16 step of rounding for kernel differences between segment lengths.
        let difference = samples(&streamed[44..])
            .into_iter()
            .zip(samples(&wav[44..]))
            .map(|(a, b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap();
        assert!(difference <= 1, "streamed PCM differs by {difference}");
        let bytes_per_frame = 256 * 2 * 2 * 2;
        assert_eq!(pcm.len(), audio.frame_length() * bytes_per_frame);
        assert!((audio.duration_seconds() - audio.frame_length() as f64 / 93.75).abs() < 1e-9);

        // Ranges are clamped instead of panicking.
        let length = audio.frame_length();
        let tail = synthesizer.render_streaming(&audio, length - 10..length + 100)?;
        assert_eq!(tail, pcm[(length - 10) * bytes_per_frame..]);
        let inverted = Range { start: 5, end: 3 };
        assert!(synthesizer.render_streaming(&audio, inverted)?.is_empty());
        assert!(
            synthesizer
                .render_streaming(&audio, length + 5..length + 9)?
                .is_empty()
        );

        let ranges = audio.chunk_ranges(7).collect::<Vec<_>>();
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, length);
        assert!(ranges.windows(2).all(|w| w[0].end == w[1].start));
        assert!(ranges.iter().all(|r| (1..=7).contains(&r.len())));
        assert_eq!(audio.chunk_ranges(0).count(), length);

        // The nonblocking synthesizer renders the same chunks.
        let nonblocking = crate::nonblocking::Synthesizer::builder(
            crate::nonblocking::Onnxruntime::get().unwrap(),
        )
        .acceleration_mode(AccelerationMode::Cpu)
        .cpu_num_threads(1)
        .build()?;
        let model = crate::nonblocking::VoiceModelFile::open(&path).block_on()?;
        nonblocking.load_voice_model(&model).perform().block_on()?;
        assert!(nonblocking.supports_streaming(style_id));
        let audio = nonblocking
            .prepare_streaming(&query, style_id, Default::default())
            .block_on()?;
        let mut nonblocking_pcm = vec![];
        for range in audio.chunk_ranges(crate::DEFAULT_STREAMING_CHUNK_FRAMES) {
            nonblocking_pcm.extend(nonblocking.render_streaming(&audio, range).block_on()?);
        }
        assert_eq!(nonblocking_pcm, pcm);
        use futures_util::TryStreamExt as _;
        let upstream_wav = nonblocking
            .streaming_synthesis(&query, style_id)
            .segment_length(crate::DEFAULT_STREAMING_CHUNK_FRAMES as f64 / FRAME_RATE)
            .perform()
            .block_on()?
            .try_collect::<Vec<_>>()
            .block_on()?
            .concat();
        assert_eq!(upstream_wav, audio.wav(&pcm));
        Ok(())
    }

    /// Models with a native split decoder stream through it as before.
    #[test]
    fn streaming_api_uses_native_split_decoder() -> anyhow::Result<()> {
        use crate::{AccelerationMode, blocking::Synthesizer};

        let synthesizer = Synthesizer::builder(Onnxruntime::from_test_util_data()?)
            .acceleration_mode(AccelerationMode::Cpu)
            .cpu_num_threads(1)
            .build()?;
        let model = VoiceModelFile::open(::test_util::SAMPLE_VOICE_MODEL_FILE_PATH)?;
        let style_id = StyleId(302);
        synthesizer.load_voice_model(&model).perform()?;
        assert!(
            !synthesizer
                .0
                .status()
                .contains_domain::<SplitTalkDomain>(style_id)
        );
        assert!(synthesizer.supports_streaming(style_id));
        let query = synthesizer.create_audio_query_from_kana("コンニチワ'", style_id)?;
        let audio = synthesizer.prepare_streaming(&query, style_id, Default::default())?;
        let pcm = synthesizer
            .render_streaming_chunks(&audio, crate::DEFAULT_STREAMING_CHUNK_FRAMES)
            .collect::<Result<Vec<_>>>()?
            .concat();
        assert_eq!(pcm.len(), audio.frame_length() * 256 * 2);
        Ok(())
    }
}
