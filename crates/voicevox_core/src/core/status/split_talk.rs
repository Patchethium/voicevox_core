//! Loads talk-only VVMs into [`SplitTalkDomain`] by splitting their decoder (`azalea` fork).
use enum_map::{EnumMap, enum_map};
use tracing::warn;

use super::{
    super::{
        infer::{
            InferenceRuntime,
            domains::{InferenceDomainMap, SplitTalkDomain, SplitTalkOperation, TalkOperation},
            session_set::InferenceSessionSet,
        },
        manifest::StyleIdToInnerVoiceId,
        voice_model::{ModelBytes, ModelBytesWithInnerVoiceIdsByDomain},
    },
    InferenceDomainExt, LoadedModel, SessionOptionsByDomain,
};

pub(super) type SessionSetWithInnerVoiceIds<R> = Option<(
    StyleIdToInnerVoiceId,
    InferenceSessionSet<R, SplitTalkDomain>,
)>;

impl InferenceDomainExt for SplitTalkDomain {
    fn visit<R: InferenceRuntime>(
        model: &LoadedModel<R>,
    ) -> Option<&(StyleIdToInnerVoiceId, InferenceSessionSet<R, Self>)> {
        model.split_talk.as_ref()
    }
}

/// Splits the talk decoder of a VVM without a native split decoder.
///
/// `None` keeps the ordinary talk domain: either there is nothing to split, the runtime
/// cannot split, or the decoder is not the supported architecture. Streaming is then
/// unavailable for the model, but it still loads.
pub(super) fn create_session_set<R: InferenceRuntime>(
    rt: &R,
    contents: &InferenceDomainMap<ModelBytesWithInnerVoiceIdsByDomain>,
    session_options: &InferenceDomainMap<SessionOptionsByDomain>,
) -> SessionSetWithInnerVoiceIds<R> {
    if contents.experimental_talk.is_some() {
        return None;
    }
    let (inner_voice_ids, models) = contents.talk.as_ref()?;
    let session_set = (|| {
        let Some([spectrogram, vocoder]) = rt.split_talk_decoder(&models[TalkOperation::Decode])?
        else {
            return Ok(None);
        };
        // In `SplitTalkOperation` order.
        let models = EnumMap::from_array([
            clone(&models[TalkOperation::PredictDuration]),
            clone(&models[TalkOperation::PredictIntonation]),
            spectrogram,
            vocoder,
        ]);
        // Same devices as the native split domain: only the vocoder is heavy.
        let talk = &session_options.talk;
        let options = enum_map! {
            SplitTalkOperation::PredictDuration
            | SplitTalkOperation::GenerateFullIntermediate => talk[TalkOperation::PredictDuration],
            SplitTalkOperation::PredictIntonation => talk[TalkOperation::PredictIntonation],
            SplitTalkOperation::RenderAudioSegment => talk[TalkOperation::Decode],
        };
        InferenceSessionSet::new(rt, &models, &options).map(Some)
    })();
    match session_set {
        Ok(session_set) => session_set.map(|session_set| (inner_voice_ids.clone(), session_set)),
        Err(error) => {
            warn!("could not split the talk decoder, so streaming is unavailable: {error:#}");
            None
        }
    }
}

fn clone(model: &ModelBytes) -> ModelBytes {
    match model {
        ModelBytes::Onnx(bytes) => ModelBytes::Onnx(bytes.clone()),
        ModelBytes::VvBin(bytes) => ModelBytes::VvBin(bytes.clone()),
    }
}
