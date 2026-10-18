//! Split of the reference monolithic talk decoder at its 80-dim mel bottleneck.
//!
//! The acoustic part (`f0`, `phoneme`, `speaker_id` → `spec`) and the HiFi-GAN vocoder
//! (`spec` → `wave`) are extracted by backward reachability, so nested model-selection
//! `If` branches and constant subgraphs are carried along with whichever part uses them.
use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{Context as _, bail, ensure};
use onnx_protobuf::{
    GraphProto, Message as _, ModelProto, NodeProto, StringStringEntryProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, tensor_shape_proto, type_proto,
};

pub(super) const SPEC: &str = "spec";
const MEL_CHANNELS: i64 = 80;
const MARKER: &str = "azalea.decoder_split.version";
const PART: &str = "azalea.decoder_split.part";
const VERSION: &str = "1";

#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "wired up by the split-talk domain")
)]
pub(super) struct SplitDecoder {
    /// `f0: f32[length, 1]`, `phoneme: f32[length, 45]`, `speaker_id: i64[1]` → `spec: f32[length, 80]`
    pub(super) spectrogram: Vec<u8>,
    /// `spec: f32[length, 80]` → `wave: f32[outlength]`
    pub(super) vocoder: Vec<u8>,
}

pub(super) fn split(bytes: &[u8]) -> anyhow::Result<SplitDecoder> {
    let mut model = ModelProto::parse_from_bytes(bytes).context("parse exported decoder ONNX")?;
    for marker in &model.metadata_props {
        if marker.key == MARKER {
            ensure!(
                marker.value == VERSION,
                "unsupported decoder split version: {}",
                marker.value
            );
            bail!("decoder is already split (version {VERSION})");
        }
    }
    ensure!(
        model
            .opset_import
            .iter()
            .any(|op| op.domain.is_empty() && (12..=20).contains(&op.version)),
        "unsupported decoder ONNX opset"
    );
    let mut graph = model.graph.take().context("missing decoder graph")?;
    ensure!(
        graph.sparse_initializer.is_empty(),
        "sparse initializers are unsupported"
    );
    let plan = Plan::new(&graph)?;

    let Plan {
        cut,
        acoustic_nodes,
        vocoder_nodes,
        acoustic_names,
        vocoder_names,
    } = plan;
    let mut acoustic = GraphProto {
        name: format!("{}_spectrogram", graph.name),
        ..Default::default()
    };
    let mut vocoder = GraphProto {
        name: format!("{}_vocoder", graph.name),
        ..Default::default()
    };
    for (i, node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
        match (acoustic_nodes.contains(&i), vocoder_nodes.contains(&i)) {
            (true, true) => {
                acoustic.node.push(node.clone());
                vocoder.node.push(node);
            }
            (true, false) => acoustic.node.push(node),
            (false, true) => vocoder.node.push(node),
            (false, false) => {}
        }
    }
    // Initializers dominate the model size, so they are moved and only cloned when shared.
    for tensor in std::mem::take(&mut graph.initializer) {
        match (
            acoustic_names.contains(&tensor.name),
            vocoder_names.contains(&tensor.name),
        ) {
            (true, true) => {
                acoustic.initializer.push(tensor.clone());
                vocoder.initializer.push(tensor);
            }
            (true, false) => acoustic.initializer.push(tensor),
            (false, true) => vocoder.initializer.push(tensor),
            (false, false) => {}
        }
    }
    let initializers = acoustic
        .initializer
        .iter()
        .chain(&vocoder.initializer)
        .map(|t| t.name.clone())
        .collect::<HashSet<_>>();
    acoustic.input = graph
        .input
        .iter()
        .filter(|v| !initializers.contains(&v.name))
        .cloned()
        .collect();
    acoustic.output = vec![spec_info()];
    vocoder.input = vec![spec_info()];
    vocoder.output = std::mem::take(&mut graph.output);
    for info in graph.value_info {
        if info.name == cut {
            continue;
        }
        if acoustic_names.contains(&info.name) {
            acoustic.value_info.push(info.clone());
        }
        if vocoder_names.contains(&info.name) {
            vocoder.value_info.push(info);
        }
    }
    if cut != SPEC {
        rename(&mut acoustic, &cut, SPEC);
        rename(&mut vocoder, &cut, SPEC);
    }

    let finish = |graph: GraphProto, part: &str| {
        let mut model = model.clone();
        model.graph = Some(graph).into();
        model.metadata_props.extend([
            StringStringEntryProto {
                key: MARKER.into(),
                value: VERSION.into(),
                ..Default::default()
            },
            StringStringEntryProto {
                key: PART.into(),
                value: part.into(),
                ..Default::default()
            },
        ]);
        model
            .write_to_bytes()
            .with_context(|| format!("serialize decoder {part} ONNX"))
    };
    Ok(SplitDecoder {
        spectrogram: finish(acoustic, "spectrogram")?,
        vocoder: finish(vocoder, "vocoder")?,
    })
}

/// Node indices and tensor names used by each part of the top-level graph.
struct Plan {
    cut: String,
    acoustic_nodes: BTreeSet<usize>,
    vocoder_nodes: BTreeSet<usize>,
    acoustic_names: HashSet<String>,
    vocoder_names: HashSet<String>,
}

impl Plan {
    fn new(graph: &GraphProto) -> anyhow::Result<Self> {
        let initializers = graph
            .initializer
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect::<HashMap<_, _>>();
        let inputs = graph
            .input
            .iter()
            .map(|v| v.name.as_str())
            .filter(|name| !initializers.contains_key(name))
            .collect::<HashSet<_>>();
        ensure!(
            graph.output.len() == 1,
            "expected one decoder output, found {}",
            graph.output.len()
        );
        let output = graph.output[0].name.as_str();
        let mut producers = HashMap::new();
        for (i, node) in graph.node.iter().enumerate() {
            for name in node.output.iter().filter(|s| !s.is_empty()) {
                ensure!(
                    producers.insert(name.as_str(), i).is_none(),
                    "ambiguous tensor producer: {name}"
                );
            }
        }
        let dependencies = graph.node.iter().map(dependencies).collect::<Vec<_>>();
        let graph = Graph {
            graph,
            producers,
            dependencies,
        };

        let mut cuts = BTreeSet::new();
        for (i, node) in graph.graph.node.iter().enumerate() {
            if let Some(cut) = graph.mel_cut(node, &initializers) {
                cuts.insert((cut, i));
            }
        }
        let mut found = Vec::new();
        for (cut, conv) in cuts {
            let (nodes, leaves) = graph.closure(output, Some(cut));
            // The vocoder must depend on the graph inputs only through the cut.
            if leaves.contains(cut)
                && nodes.contains(&conv)
                && leaves
                    .iter()
                    .all(|name| *name == cut || initializers.contains_key(name))
            {
                found.push((cut, nodes, leaves));
            }
        }
        let [(cut, vocoder_nodes, vocoder_leaves)] =
            <[_; 1]>::try_from(found).map_err(|found| {
                anyhow::anyhow!(
                    "expected one mel bottleneck feeding the vocoder, found {}",
                    found.len()
                )
            })?;
        ensure!(
            cut != output && !inputs.contains(cut),
            "mel bottleneck must be an intermediate tensor"
        );
        if cut != SPEC {
            ensure!(
                !all_names(graph.graph).contains(SPEC),
                "reserved decoder tensor name: {SPEC}"
            );
        }
        let (acoustic_nodes, acoustic_leaves) = graph.closure(cut, None);
        for leaf in &acoustic_leaves {
            ensure!(
                inputs.contains(leaf) || initializers.contains_key(leaf),
                "undefined tensor in acoustic part: {leaf}"
            );
        }
        let names = |nodes: &BTreeSet<usize>, leaves: HashSet<&str>| {
            nodes
                .iter()
                .flat_map(|&i| graph.graph.node[i].output.iter().map(String::as_str))
                .chain(leaves)
                .map(str::to_owned)
                .collect::<HashSet<_>>()
        };
        Ok(Self {
            cut: cut.to_owned(),
            acoustic_names: names(&acoustic_nodes, acoustic_leaves),
            vocoder_names: names(&vocoder_nodes, vocoder_leaves),
            acoustic_nodes,
            vocoder_nodes,
        })
    }
}

struct Graph<'a> {
    graph: &'a GraphProto,
    producers: HashMap<&'a str, usize>,
    dependencies: Vec<Vec<&'a str>>,
}

impl<'a> Graph<'a> {
    /// `T[length, 80] → Transpose(perm=[1, 0]) → Unsqueeze(axes=[0]) → Conv(W[_, 80, _])`
    fn mel_cut(
        &self,
        conv: &'a NodeProto,
        initializers: &HashMap<&str, &TensorProto>,
    ) -> Option<&'a str> {
        if !(conv.op_type == "Conv" && conv.domain.is_empty() && matches!(conv.input.len(), 2 | 3))
        {
            return None;
        }
        let weight = initializers.get(conv.input[1].as_str())?;
        if !(weight.dims.len() == 3 && weight.dims[1] == MEL_CHANNELS) {
            return None;
        }
        let unsqueeze = self.producer(&conv.input[0])?;
        let axes = match unsqueeze.input.len() {
            1 => unsqueeze
                .attribute
                .iter()
                .find(|a| a.name == "axes")
                .map(|a| a.ints.clone()),
            2 => self.constant_ints(&unsqueeze.input[1], initializers),
            _ => None,
        };
        if !(unsqueeze.op_type == "Unsqueeze" && axes.as_deref() == Some(&[0])) {
            return None;
        }
        let transpose = self.producer(&unsqueeze.input[0])?;
        let perm = transpose.attribute.iter().find(|a| a.name == "perm")?;
        (transpose.op_type == "Transpose" && transpose.input.len() == 1 && perm.ints == [1, 0])
            .then_some(transpose.input[0].as_str())
    }

    fn producer(&self, name: &str) -> Option<&'a NodeProto> {
        let node = &self.graph.node[*self.producers.get(name)?];
        (node.domain.is_empty() && node.output.len() == 1).then_some(node)
    }

    fn constant_ints(
        &self,
        name: &str,
        initializers: &HashMap<&str, &TensorProto>,
    ) -> Option<Vec<i64>> {
        let tensor = match initializers.get(name) {
            Some(tensor) => *tensor,
            None => {
                let node = self.producer(name)?;
                if node.op_type != "Constant" {
                    return None;
                }
                node.attribute
                    .iter()
                    .find(|a| a.name == "value")?
                    .t
                    .as_ref()?
            }
        };
        if tensor.data_type != 7 {
            return None;
        }
        if !tensor.int64_data.is_empty() {
            return Some(tensor.int64_data.clone());
        }
        let chunks = tensor.raw_data.chunks_exact(8);
        chunks.remainder().is_empty().then(|| {
            chunks
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
    }

    /// Nodes needed to compute `target`, and the tensors they read that no node produces.
    fn closure(&self, target: &'a str, stop: Option<&str>) -> (BTreeSet<usize>, HashSet<&'a str>) {
        let mut nodes = BTreeSet::new();
        let mut leaves = HashSet::new();
        let mut seen = HashSet::new();
        let mut stack = vec![target];
        while let Some(name) = stack.pop() {
            if !seen.insert(name) {
                continue;
            }
            match self.producers.get(name) {
                Some(&i) if Some(name) != stop => {
                    if nodes.insert(i) {
                        stack.extend(&self.dependencies[i]);
                    }
                }
                _ => {
                    leaves.insert(name);
                }
            }
        }
        (nodes, leaves)
    }
}

/// Explicit inputs plus outer-scope tensors captured by subgraph attributes.
fn dependencies(node: &NodeProto) -> Vec<&str> {
    let mut names = node
        .input
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    for attribute in &node.attribute {
        for graph in attribute.g.as_ref().into_iter().chain(&attribute.graphs) {
            names.extend(captured(graph));
        }
    }
    names
}

fn captured(graph: &GraphProto) -> BTreeSet<&str> {
    let defined = graph
        .input
        .iter()
        .map(|v| v.name.as_str())
        .chain(graph.initializer.iter().map(|t| t.name.as_str()))
        .chain(
            graph
                .node
                .iter()
                .flat_map(|n| n.output.iter().map(String::as_str)),
        )
        .collect::<HashSet<_>>();
    graph
        .node
        .iter()
        .flat_map(dependencies)
        .filter(|name| !defined.contains(name))
        .collect()
}

fn all_names(graph: &GraphProto) -> HashSet<&str> {
    let mut names = graph
        .input
        .iter()
        .chain(&graph.output)
        .chain(&graph.value_info)
        .map(|v| v.name.as_str())
        .chain(graph.initializer.iter().map(|t| t.name.as_str()))
        .chain(
            graph
                .node
                .iter()
                .flat_map(|n| n.input.iter().chain(&n.output).map(String::as_str)),
        )
        .collect::<HashSet<_>>();
    for node in &graph.node {
        for attribute in &node.attribute {
            for graph in attribute.g.as_ref().into_iter().chain(&attribute.graphs) {
                names.extend(all_names(graph));
            }
        }
    }
    names
}

fn rename(graph: &mut GraphProto, from: &str, to: &str) {
    for name in graph
        .node
        .iter_mut()
        .flat_map(|n| n.input.iter_mut().chain(&mut n.output))
    {
        if name == from {
            *name = to.into();
        }
    }
    for node in &mut graph.node {
        for attribute in &mut node.attribute {
            for graph in attribute
                .g
                .as_mut()
                .into_iter()
                .chain(&mut attribute.graphs)
            {
                rename(graph, from, to);
            }
        }
    }
}

fn spec_info() -> ValueInfoProto {
    let mut length = tensor_shape_proto::Dimension::new();
    length.set_dim_param("length".into());
    let mut channels = tensor_shape_proto::Dimension::new();
    channels.set_dim_value(MEL_CHANNELS);
    let mut ty = TypeProto::new();
    ty.set_tensor_type(type_proto::Tensor {
        elem_type: 1,
        shape: Some(TensorShapeProto {
            dim: vec![length, channels],
            ..Default::default()
        })
        .into(),
        ..Default::default()
    });
    ValueInfoProto {
        name: SPEC.into(),
        type_: Some(ty).into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_protobuf::{AttributeProto, attribute_proto::AttributeType};
    use std::sync::LazyLock;

    // Same reference architecture as production 0.16 `talk.decode` graphs (family A).
    static SAMPLE: LazyLock<Vec<u8>> = LazyLock::new(|| {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../model/sample.vvm/decode.onnx"
        ))
        .unwrap()
    });

    fn sample() -> ModelProto {
        ModelProto::parse_from_bytes(&SAMPLE).unwrap()
    }
    fn apply(model: &ModelProto) -> anyhow::Result<(ModelProto, ModelProto)> {
        let SplitDecoder {
            spectrogram,
            vocoder,
        } = split(&model.write_to_bytes().unwrap())?;
        Ok((
            ModelProto::parse_from_bytes(&spectrogram)?,
            ModelProto::parse_from_bytes(&vocoder)?,
        ))
    }
    fn names(values: &[ValueInfoProto]) -> Vec<&str> {
        values.iter().map(|v| v.name.as_str()).collect()
    }
    fn initializer_bytes(graph: &GraphProto) -> usize {
        graph.initializer.iter().map(|t| t.raw_data.len()).sum()
    }
    fn uses(graph: &GraphProto, name: &str) -> bool {
        all_names(graph).contains(name)
    }

    #[test]
    fn splits_reference_decoder_and_preserves_content() {
        let before = sample();
        let (acoustic, vocoder) = apply(&before).unwrap();
        let (a, v, b) = (&acoustic.graph, &vocoder.graph, &before.graph);
        assert_eq!(names(&a.input), ["f0", "phoneme", "speaker_id"]);
        assert_eq!(names(&a.output), [SPEC]);
        assert_eq!(names(&v.input), [SPEC]);
        assert_eq!(names(&v.output), ["wave"]);
        let dims = v.input[0].type_.tensor_type().shape.dim.clone();
        assert_eq!(dims[0].dim_param(), "length");
        assert_eq!(dims[1].dim_value(), MEL_CHANNELS);
        assert_eq!(a.node.len() + v.node.len(), b.node.len());
        assert_eq!(
            a.initializer.len() + v.initializer.len(),
            b.initializer.len()
        );
        assert_eq!(
            initializer_bytes(a) + initializer_bytes(v),
            initializer_bytes(b)
        );
        assert!(
            v.initializer
                .iter()
                .any(|t| t.name == "hifi_gan_forwarder.conv_pre.weight")
        );
        assert!(!uses(v, "f0") && !uses(v, "speaker_id") && !uses(v, "1161"));
        assert!(!uses(a, "1161") && !uses(a, "wave"));
        assert_eq!(before.opset_import, acoustic.opset_import);
        assert_eq!(before.opset_import, vocoder.opset_import);
        for (model, part) in [(&acoustic, "spectrogram"), (&vocoder, "vocoder")] {
            let props = &model.metadata_props;
            assert!(props.iter().any(|p| p.key == MARKER && p.value == VERSION));
            assert!(props.iter().any(|p| p.key == PART && p.value == part));
            let error = split(&model.write_to_bytes().unwrap()).unwrap_err();
            assert!(error.to_string().contains("already"), "{error}");
        }
        let mut future = acoustic;
        future.metadata_props[0].value = "999".into();
        let error = split(&future.write_to_bytes().unwrap()).unwrap_err();
        assert!(error.to_string().contains("version"), "{error}");
    }

    fn rename_all(graph: &mut GraphProto) {
        let keep = ["f0", "phoneme", "speaker_id", "wave", ""];
        let rename = |name: &mut String| {
            if !keep.contains(&name.as_str()) {
                *name = format!("renamed_{name}");
            }
        };
        for v in graph
            .input
            .iter_mut()
            .chain(&mut graph.output)
            .chain(&mut graph.value_info)
        {
            rename(&mut v.name);
        }
        for t in &mut graph.initializer {
            rename(&mut t.name);
        }
        for n in &mut graph.node {
            for v in n.input.iter_mut().chain(&mut n.output) {
                rename(v);
            }
            for a in &mut n.attribute {
                if let Some(g) = a.g.as_mut() {
                    rename_all(g);
                }
            }
        }
    }

    #[test]
    fn renamed_tensors_and_already_named_spec() {
        let mut model = sample();
        rename_all(model.graph.as_mut().unwrap());
        let (_, vocoder) = apply(&model).unwrap();
        assert_eq!(names(&vocoder.graph.input), [SPEC]);

        // Newer production graphs (families B and C) already name the bottleneck `spec`.
        let mut model = sample();
        rename(model.graph.as_mut().unwrap(), "1161", SPEC);
        let (acoustic, vocoder) = apply(&model).unwrap();
        assert_eq!(names(&acoustic.graph.output), [SPEC]);
        assert_eq!(names(&vocoder.graph.input), [SPEC]);
    }

    fn selection(output: &str, branches: [GraphProto; 2]) -> NodeProto {
        NodeProto {
            op_type: "If".into(),
            input: vec!["choose_model".into()],
            output: vec![output.into()],
            attribute: ["then_branch", "else_branch"]
                .into_iter()
                .zip(branches)
                .map(|(name, g)| AttributeProto {
                    name: name.into(),
                    type_: AttributeType::GRAPH.into(),
                    g: Some(g).into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    /// Family C: the acoustic models live in speaker-selected `If` branches that capture
    /// outer-scope inputs and initializers; the shared vocoder stays at the top level.
    #[test]
    fn nested_model_selection_captures_outer_scope() {
        let before = sample();
        let mut model = before.clone();
        let graph = model.graph.as_mut().unwrap();
        let plan = Plan::new(graph).unwrap();
        let (mut acoustic, mut top) = (vec![], vec![]);
        for (i, node) in std::mem::take(&mut graph.node).into_iter().enumerate() {
            if plan.acoustic_nodes.contains(&i) {
                acoustic.push(node);
            } else {
                top.push(node);
            }
        }
        let output = |name: &str| ValueInfoProto {
            name: name.into(),
            ..spec_info()
        };
        let mut leaf = GraphProto {
            node: acoustic,
            output: vec![output("selected")],
            ..Default::default()
        };
        rename(&mut leaf, &plan.cut, "selected");
        let inner = GraphProto {
            node: vec![selection("selected", [leaf.clone(), leaf.clone()])],
            output: vec![output("selected")],
            ..Default::default()
        };
        top.splice(
            0..0,
            [
                NodeProto {
                    op_type: "Less".into(),
                    input: vec!["speaker_id".into(), "offset".into()],
                    output: vec!["choose_model".into()],
                    ..Default::default()
                },
                selection(&plan.cut, [leaf, inner]),
            ],
        );
        graph.node = top;
        graph.initializer.push(TensorProto {
            name: "offset".into(),
            data_type: 7,
            dims: vec![1],
            int64_data: vec![2],
            ..Default::default()
        });
        graph.value_info.clear();

        let (acoustic, vocoder) = apply(&model).unwrap();
        let (a, v) = (&acoustic.graph, &vocoder.graph);
        assert_eq!(names(&a.input), ["f0", "phoneme", "speaker_id"]);
        assert_eq!(a.node.len(), 2);
        assert_eq!(a.node[1].op_type, "If");
        assert_eq!(a.node[1].output, [SPEC]);
        assert_eq!(v.node.len(), plan.vocoder_nodes.len());
        // Branch weights are captured from the top level and must follow the acoustic part.
        assert!(
            a.initializer
                .iter()
                .any(|t| t.name == "yukarin_sosoa_forwarder.post.bias")
        );
        assert!(a.initializer.iter().any(|t| t.name == "offset"));
        assert!(v.initializer.iter().all(|t| t.name != "offset"));
        assert_eq!(
            initializer_bytes(a) + initializer_bytes(v),
            initializer_bytes(&before.graph)
        );
    }

    #[test]
    fn constant_nodes_follow_their_consumers() {
        let mut model = sample();
        let graph = model.graph.as_mut().unwrap();
        let index = graph
            .initializer
            .iter()
            .position(|t| t.name == "hifi_gan_forwarder.conv_pre.bias")
            .unwrap();
        let bias = graph.initializer.remove(index);
        graph.node.insert(
            0,
            NodeProto {
                op_type: "Constant".into(),
                output: vec![bias.name.clone()],
                attribute: vec![AttributeProto {
                    name: "value".into(),
                    type_: AttributeType::TENSOR.into(),
                    t: Some(bias).into(),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        let (acoustic, vocoder) = apply(&model).unwrap();
        assert!(vocoder.graph.node.iter().any(|n| n.op_type == "Constant"));
        assert!(acoustic.graph.node.iter().all(|n| n.op_type != "Constant"));
    }

    #[test]
    fn rejects_missing_ambiguous_dependent_and_colliding_graphs() {
        assert!(split(b"not protobuf").is_err());
        let conv_pre = |g: &GraphProto| {
            g.node
                .iter()
                .position(|n| {
                    n.input
                        .iter()
                        .any(|i| i == "hifi_gan_forwarder.conv_pre.weight")
                })
                .unwrap()
        };
        for mutation in 0..8 {
            let mut m = sample();
            let g = m.graph.as_mut().unwrap();
            match mutation {
                // No bottleneck pattern.
                0 => {
                    let i = conv_pre(g);
                    g.node[i].op_type = "ConvTranspose".into();
                }
                // Vocoder reads a graph input directly.
                1 => {
                    let i = conv_pre(g);
                    let out = g.node[i].output[0].clone();
                    rename(g, &out, "mixed");
                    g.node[i].output[0] = out.clone();
                    g.node.insert(
                        i + 1,
                        NodeProto {
                            op_type: "Add".into(),
                            input: vec![out, "f0".into()],
                            output: vec!["mixed".into()],
                            ..Default::default()
                        },
                    );
                }
                // Wrong transpose.
                2 => {
                    let i = conv_pre(g);
                    let unsqueeze = g.node[i].input[0].clone();
                    let transpose = g
                        .node
                        .iter()
                        .find(|n| n.output == [unsqueeze.clone()])
                        .unwrap()
                        .input[0]
                        .clone();
                    let node = g.node.iter_mut().find(|n| n.output == [transpose.clone()]);
                    let perm = node
                        .unwrap()
                        .attribute
                        .iter_mut()
                        .find(|a| a.name == "perm");
                    perm.unwrap().ints = vec![0, 1];
                }
                // Reserved name collision.
                3 => {
                    let i = conv_pre(g);
                    let out = g.node[i].output[0].clone();
                    rename(g, &out, SPEC);
                }
                // Two outputs.
                4 => {
                    let extra = g.output[0].clone();
                    g.output.push(ValueInfoProto {
                        name: "1161".into(),
                        ..extra
                    });
                }
                // Duplicate producer.
                5 => {
                    let node = g.node[conv_pre(g)].clone();
                    g.node.push(node);
                }
                // Undefined acoustic tensor.
                6 => {
                    let i = g
                        .node
                        .iter()
                        .position(|n| n.input.iter().any(|x| x == "f0"))
                        .unwrap();
                    for x in &mut g.node[i].input {
                        if x == "f0" {
                            *x = "missing".into();
                        }
                    }
                }
                // Unsupported opset.
                7 => m.opset_import.retain(|o| !o.domain.is_empty()),
                _ => unreachable!(),
            }
            assert!(apply(&m).is_err(), "mutation {mutation}");
        }
    }
}
