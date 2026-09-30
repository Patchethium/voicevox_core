# Seeded autoregressive pitch noise (`specta` fork)

Both Rust synthesizers expose `replace_mora_pitch_with_noise(phrases, style_id,
PitchNoiseOptions { sigma, seed })`. Create an ordinary query, replace its
`accent_phrases`, then synthesize it. Duration, text, accents and other fields
are preserved. `AudioQuery`, VVM manifests, existing method signatures, and
other language bindings are unchanged.

`sigma` is a finite, nonnegative standard deviation in the predictor's natural-log
pitch units. The default is zero, with seed zero. Zero bypasses sampling.
A request-local `ChaCha8Rng::seed_from_u64(u64::from(seed))` samples float32
`rand_distr::StandardNormal` values in voiced-mora order. Boundary silence,
pauses and unvoiced moras receive zero noise and consume no samples. Existing
post-inference unvoiced masking remains in effect. Zero noise at a later step
does not reset earlier autoregressive effects. Nonfinite noise and predictions
are errors; samples are never clamped.

Reproducibility applies to the same model, inputs, seed and implementation
version, including RNG dependencies. Bitwise inference equivalence across
hardware or ONNX Runtime versions is not promised. Legacy query creation, TTS
and low-level pitch inference always supply zeros.

## Automatic loading and supported graphs

Every talk and experimental-talk intonation session is transformed during
ordinary VVM loading, even when only deterministic APIs are used. A temporary
CPU session with optimizations disabled exports a self-contained ONNX model.
Only binary input uses `session.use_vv_bin`. `tempfile` owns the private export
directory; the export session is released before parsing, and temporary files
are removed on both success and failure. Loading requires a writable system
temporary directory and enough disk space for each exported predictor. Binary
VVMs require VOICEVOX ONNX Runtime. No persistent conversion cache, Python
runtime, installation edits, or separate conversion step is involved.

The transformer supports standard ONNX opsets 12–20 and recognizes the reference
GRU/Conv predictor by connectivity rather than internal names:

- The `length` input bounds a Loop with one singleton float32 pitch carry and
  one GRU hidden-state carry.
- Previous pitch enters a Concat → Transpose → GRU path. The GRU output flows
  through Squeeze → Transpose → Conv to the next pitch carry.
- Two Gather nodes collect that same pitch either as a Loop scan output or by
  appending it to a third, sequence carry (`SequenceInsert`, opset 13 or later).
- Nested model-selection If branches each contain exactly one supported predictor.

The new float32 `[length]` input, `azalea_pitch_noise`, is gathered at the Loop
iteration index and added immediately after the Conv. Both feedback and output
consume the sum. Weights, unrelated graph content and metadata are preserved by
ONNX protobuf parsing/serialization. Final sessions use the normal inference
settings and strict input/output signature checks.

Missing, ambiguous or unsupported structures intentionally fail loading; there
is no deterministic fallback that silently disables noise. Reserved tensor-name
collisions and already-transformed models are rejected. Metadata key
`azalea.pitch_noise.version=1` identifies this rewrite; unknown versions fail.
Model-load errors retain the VVM path and domain context, and no partial model
is registered. A failed reload preserves the previous model.

## Listening example

From the repository root, using downloaded runtime/model paths:

```sh
cargo run -p voicevox_core --example pitch_noise --features load-onnxruntime -- \
  /path/to/libvoicevox_onnxruntime.so.1.17.3 /path/to/0.vvm \
  2 "コンニチワ'" 0.05 42 noisy.wav
```

Repeat with sigma `0` for a baseline, or change the seed for another trajectory.
The example uses kana so it does not require an Open JTalk dictionary argument.

## Verification and local diagnostics

```sh
cargo test -p voicevox_core --features load-onnxruntime,specta --locked
cargo test -p voicevox_core --features load-onnxruntime --locked
cargo check -p voicevox_core --all-targets --features load-onnxruntime,specta --locked
cargo check -p voicevox_core --all-targets --features load-onnxruntime --locked
```

Automated tests use the public sample model, including constructed nested
selection and sequence-collection variants. They check graph rejection and
duplicate markers, seeded and concurrent requests, finite-value validation,
voiced-only placement, API parity, non-pitch preservation, experimental-talk
inference and atomic registration/reload failures. Original and transformed
sessions use identical Level1 CPU inference settings. Zero-noise comparisons
allow at most `1e-6` absolute error in log-pitch units for runtime/kernel fusion
differences; observed results below were bitwise identical. A fixed `0.05`
impulse at step 1 must leave step 0 unchanged and alter at least one later step
with zero direct noise.

The installed-model diagnostic is explicitly opt-in and runs in a separate
process because the runtime is a singleton:

```sh
AZALEA_PITCH_ORT=/path/to/libvoicevox_onnxruntime.so.1.17.3 \
AZALEA_PITCH_VVM_DIR=/path/to/vvms \
cargo test -p voicevox_core --features load-onnxruntime \
  installed_pitch_diagnostic -- --ignored --nocapture
```

It checks every talk style in both talk domains, reports each comparison and
performs ordinary VVM loading, including non-talk domains. Exported production
models are never retained or committed. All failures are collected and make the
diagnostic fail.

Local validation with VOICEVOX ONNX Runtime 1.17.3 on Linux passed all 113
predictor/style combinations across 22 installed talk VVMs; ordinary loading
also passed for the singing VVM. Every zero-noise comparison was bitwise equal,
and every impulse propagated. With one inference thread and five timesteps,
temporary export plus graph transformation took 2.65–44.75 ms (median 8.84 ms).
Steady-state inference, including input/output construction, averaged
205–550 µs per request (median 246 µs across styles, 20 repetitions each).
Final-session creation and other VVM sessions are excluded from transformation
time. These are local debug-build measurements, not a performance guarantee.
