use rand::SeedableRng as _;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as _, StandardNormal};
use serde::{Deserialize, Serialize};

use crate::{Result, error::InvalidQueryError};

/// Gaussian noise added to the autoregressive talk pitch predictor.
///
/// Reproducible for the same model, inputs, seed and implementation version.
/// Noise is sampled only for voiced moras; silence does not reset feedback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct PitchNoiseOptions {
    /// Standard deviation in the predictor's natural-log pitch units.
    /// Must be finite and nonnegative. Zero bypasses random sampling.
    pub sigma: f32,
    /// Request-local ChaCha8 seed.
    pub seed: u32,
}

impl PitchNoiseOptions {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(InvalidQueryError {
                what: "pitch noise sigma must be finite and nonnegative",
                value: Some(Box::new(self.sigma)),
                source: None,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn sample(self, voiced: impl IntoIterator<Item = bool>) -> Result<Vec<f32>> {
        self.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.into());
        voiced
            .into_iter()
            .map(|voiced| {
                let noise = if voiced && self.sigma != 0.0 {
                    let normal: f32 = StandardNormal.sample(&mut rng);
                    normal * self.sigma
                } else {
                    0.0
                };
                if !noise.is_finite() {
                    return Err(InvalidQueryError {
                        what: "generated pitch noise is nonfinite",
                        value: Some(Box::new(noise)),
                        source: None,
                    }
                    .into());
                }
                Ok(noise)
            })
            .collect()
    }
}

/// Build padded teacher-forcing inputs, leaving silence/unvoiced feedback unchanged.
/// The public prefix excludes boundary silence, but includes pause moras.
pub(crate) fn prefix_inputs(prefix: &[f32], voiced: &[bool]) -> Result<(Vec<f32>, Vec<i64>)> {
    let invalid = |what| InvalidQueryError {
        what,
        value: None,
        source: None,
    };
    if prefix.len() > voiced.len().saturating_sub(2) {
        return Err(invalid("pitch prefix is longer than the mora sequence").into());
    }
    let mut values = vec![0.0; voiced.len()];
    let mut mask = vec![0; voiced.len()];
    for (i, &pitch) in prefix.iter().enumerate() {
        if !pitch.is_finite() || pitch < 0.0 {
            return Err(invalid("pitch prefix values must be finite and nonnegative").into());
        }
        if !voiced[i + 1] && pitch != 0.0 {
            return Err(invalid("unvoiced and pause pitch prefix values must be zero").into());
        }
        values[i + 1] = pitch;
        mask[i + 1] = i64::from(voiced[i + 1]);
    }
    Ok((values, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_placement_and_request_local_repeatability() {
        for sigma in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(PitchNoiseOptions { sigma, seed: 0 }.sample([true]).is_err());
        }
        assert_eq!(
            PitchNoiseOptions::default().sample([true, false]).unwrap(),
            [0.0; 2]
        );
        let options = PitchNoiseOptions {
            sigma: 0.1,
            seed: 42,
        };
        let expected = options.sample([false, true, false, true, false]).unwrap();
        assert_eq!([expected[0], expected[2], expected[4]], [0.0; 3]);
        assert_eq!(
            [expected[1], expected[3]],
            options.sample([true, true]).unwrap()[..]
        );
        assert_ne!(
            expected,
            PitchNoiseOptions {
                seed: 43,
                ..options
            }
            .sample([false, true, false, true, false])
            .unwrap()
        );
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    options.sample([false, true, false, true, false]).unwrap()
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), expected);
        }
        assert!(
            PitchNoiseOptions {
                sigma: f32::MAX,
                seed: 0
            }
            .sample([true; 100])
            .is_err()
        );
    }
}
