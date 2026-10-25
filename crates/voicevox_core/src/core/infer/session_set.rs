use std::{collections::HashMap, fmt::Display, marker::PhantomData, sync::Arc};

use anyhow::{Context as _, bail};
use enum_map::{Enum as _, EnumMap};
use itertools::Itertools as _;

use crate::error::ErrorRepr;

use super::{
    super::voice_model::ModelBytes, InferenceDomain, InferenceInputSignature, InferenceOperation,
    InferenceRuntime, InferenceSessionOptions, InferenceSignature, PITCH_CONTROL_INPUTS, ParamInfo,
};

pub(crate) struct InferenceSessionSet<R: InferenceRuntime, D: InferenceDomain>(
    EnumMap<D::Operation, Arc<R::Session>>,
);

impl<R: InferenceRuntime, D: InferenceDomain> InferenceSessionSet<R, D> {
    pub(in super::super) fn new(
        rt: &R,
        model_bytes: &EnumMap<D::Operation, ModelBytes>,
        options: &EnumMap<D::Operation, InferenceSessionOptions>,
    ) -> anyhow::Result<Self> {
        let mut sessions = model_bytes
            .iter()
            .map(|(op, model_bytes)| {
                let (expected_input_param_infos, expected_output_param_infos) =
                    <D::Operation as InferenceOperation>::PARAM_INFOS[op];

                let (sess, actual_input_param_infos, actual_output_param_infos) = rt
                    .new_session(
                        model_bytes,
                        InferenceSessionOptions {
                            pitch_noise: D::needs_pitch_noise(op),
                            ..options[op]
                        },
                    )
                    .with_context(|| {
                        format!(
                            "{} operation {}",
                            std::any::type_name::<D>(),
                            op.into_usize()
                        )
                    })?;

                let expected_input_param_infos = expected_inputs(
                    D::needs_pitch_noise(op),
                    expected_input_param_infos,
                    &actual_input_param_infos,
                );
                check_param_infos(&expected_input_param_infos, &actual_input_param_infos)?;
                check_param_infos(
                    &expected_output_param_infos.iter().collect::<Vec<_>>(),
                    &actual_output_param_infos,
                )?;

                Ok((op.into_usize(), sess.into()))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        return Ok(Self(EnumMap::<D::Operation, _>::from_fn(|k| {
            sessions.remove(&k.into_usize()).expect("should exist")
        })));

        fn check_param_infos<D: PartialEq + Display>(
            expected: &[&ParamInfo<D>],
            actual: &[ParamInfo<D>],
        ) -> anyhow::Result<()> {
            if !(expected.len() == actual.len()
                && itertools::zip_eq(expected, actual)
                    .all(|(expected, actual)| expected.accepts(actual)))
            {
                let expected = display_param_infos(expected.iter().copied());
                let actual = display_param_infos(actual);
                bail!("expected {{{expected}}}, got {{{actual}}}")
            }
            Ok(())
        }

        fn display_param_infos<'a, D: Display + 'a>(
            infos: impl IntoIterator<Item = &'a ParamInfo<D>>,
        ) -> impl Display {
            infos
                .into_iter()
                .map(|ParamInfo { name, dt, ndim }| {
                    let brackets = match *ndim {
                        Some(ndim) => &"[]".repeat(ndim),
                        None => "[]...",
                    };
                    format!("{name}: {dt}{brackets}")
                })
                .join(", ")
        }
    }
}

/// The inputs a session must declare: a pitch predictor the runtime could not transform lacks
/// the pitch-control inputs (`azalea` fork).
fn expected_inputs<'a, K>(
    needs_pitch_noise: bool,
    expected: &'a [ParamInfo<K>],
    actual: &[ParamInfo<K>],
) -> Vec<&'a ParamInfo<K>> {
    let is_pitch_control = |info: &ParamInfo<K>| PITCH_CONTROL_INPUTS.contains(&&*info.name);
    let untransformed = needs_pitch_noise && !actual.iter().any(is_pitch_control);
    expected
        .iter()
        .filter(|info| !(untransformed && is_pitch_control(info)))
        .collect()
}

impl<R: InferenceRuntime, D: InferenceDomain> InferenceSessionSet<R, D> {
    pub(in super::super) fn get<I>(&self) -> InferenceSessionCell<R, I>
    where
        I: InferenceInputSignature<Signature: InferenceSignature<Domain = D>>,
    {
        InferenceSessionCell {
            inner: self.0[I::Signature::OPERATION].clone(),
            marker: PhantomData,
        }
    }
}

pub(in super::super) struct InferenceSessionCell<R: InferenceRuntime, I> {
    inner: Arc<R::Session>,
    marker: PhantomData<fn(I)>,
}

impl<R: InferenceRuntime, I: InferenceInputSignature> InferenceSessionCell<R, I> {
    pub(in super::super) async fn run<A: super::AsyncExt>(
        self,
        input: I,
        cancellable: A::Cancellable,
    ) -> crate::Result<<I::Signature as InferenceSignature>::Output> {
        async {
            let ctx = input.make_run_context::<R>(self.inner.clone())?;
            A::run_session::<R>(ctx, cancellable).await?.try_into()
        }
        .await
        .map_err(|source| ErrorRepr::RunModel { note: None, source })
        .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::InputScalarKind, ParamInfo, expected_inputs};

    fn infos(names: &[&'static str]) -> Vec<ParamInfo<InputScalarKind>> {
        names
            .iter()
            .map(|&name| ParamInfo {
                name: name.into(),
                dt: InputScalarKind::Float32,
                ndim: Some(1),
            })
            .collect()
    }

    fn names(infos: Vec<&ParamInfo<InputScalarKind>>) -> Vec<&str> {
        infos.into_iter().map(|info| &*info.name).collect()
    }

    const TRANSFORMED: &[&str] = &[
        "length",
        "azalea_pitch_noise",
        "azalea_pitch_prefix",
        "azalea_pitch_prefix_mask",
    ];

    #[test]
    fn untransformed_pitch_predictors_need_no_pitch_control_inputs() {
        let expected = infos(TRANSFORMED);
        let stock = infos(&["length"]);
        assert_eq!(["length"], *names(expected_inputs(true, &expected, &stock)));
    }

    #[test]
    fn transformed_pitch_predictors_and_other_operations_keep_every_input() {
        let expected = infos(TRANSFORMED);
        let transformed = infos(TRANSFORMED);
        assert_eq!(
            TRANSFORMED,
            names(expected_inputs(true, &expected, &transformed))
        );
        let stock = infos(&["length"]);
        assert_eq!(
            TRANSFORMED,
            names(expected_inputs(false, &expected, &stock))
        );
    }
}
