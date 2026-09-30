//! Conservative rewrite of the reference autoregressive GRU/Conv pitch predictor.
use anyhow::{Context as _, bail, ensure};
use onnx_protobuf::{
    GraphProto, Message as _, ModelProto, NodeProto, StringStringEntryProto, TensorShapeProto,
    TypeProto, ValueInfoProto, tensor_shape_proto, type_proto,
};

pub(super) const NOISE_INPUT: &str = "azalea_pitch_noise";
pub(super) const PREFIX_INPUT: &str = "azalea_pitch_prefix";
pub(super) const PREFIX_MASK_INPUT: &str = "azalea_pitch_prefix_mask";
const MARKER: &str = "azalea.pitch_noise.version";
const VERSION: &str = "2";

pub(super) fn transform(bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut model = ModelProto::parse_from_bytes(bytes).context("parse exported pitch ONNX")?;
    for marker in &model.metadata_props {
        if marker.key == MARKER {
            ensure!(
                marker.value == VERSION,
                "unsupported pitch noise transformation version: {}",
                marker.value
            );
            bail!("pitch predictor already has pitch noise transformation version {VERSION}");
        }
    }
    ensure!(
        model
            .opset_import
            .iter()
            .any(|op| op.domain.is_empty() && (12..=20).contains(&op.version)),
        "unsupported pitch predictor ONNX opset"
    );
    let graph = model
        .graph
        .as_mut()
        .context("missing pitch predictor graph")?;
    check_names(graph)?;
    let mut count = 0;
    rewrite_selection(graph, &mut count).context("unsupported autoregressive pitch graph")?;
    let mut dimension = tensor_shape_proto::Dimension::new();
    dimension.set_dim_param("length".into());
    let mut ty = TypeProto::new();
    ty.set_tensor_type(type_proto::Tensor {
        elem_type: 1,
        shape: Some(TensorShapeProto {
            dim: vec![dimension],
            ..Default::default()
        })
        .into(),
        ..Default::default()
    });
    for (name, elem_type) in [(NOISE_INPUT, 1), (PREFIX_INPUT, 1), (PREFIX_MASK_INPUT, 7)] {
        let mut ty = ty.clone();
        ty.mut_tensor_type().elem_type = elem_type;
        graph.input.push(ValueInfoProto {
            name: name.into(),
            type_: Some(ty).into(),
            ..Default::default()
        });
    }
    model.metadata_props.push(StringStringEntryProto {
        key: MARKER.into(),
        value: VERSION.into(),
        ..Default::default()
    });
    model
        .write_to_bytes()
        .context("serialize transformed pitch ONNX")
}

fn check_names(graph: &GraphProto) -> anyhow::Result<()> {
    for name in graph
        .input
        .iter()
        .chain(&graph.output)
        .chain(&graph.value_info)
        .map(|v| &v.name)
        .chain(graph.initializer.iter().map(|v| &v.name))
        .chain(
            graph
                .node
                .iter()
                .flat_map(|n| n.input.iter().chain(&n.output)),
        )
    {
        ensure!(
            !name.starts_with("azalea_pitch_"),
            "reserved pitch noise tensor name: {name}"
        );
    }
    for node in &graph.node {
        for attribute in &node.attribute {
            if let Some(graph) = attribute.g.as_ref() {
                check_names(graph)?;
            }
            for graph in &attribute.graphs {
                check_names(graph)?;
            }
        }
    }
    Ok(())
}

// Each selected branch must contain exactly one predictor. Arbitrary control flow
// and multiple sequential pitch loops are deliberately rejected.
fn rewrite_selection(graph: &mut GraphProto, count: &mut usize) -> anyhow::Result<()> {
    let mut predictors = 0;
    for node in &mut graph.node {
        match node.op_type.as_str() {
            "Loop" if node.domain.is_empty() => {
                ensure!(
                    matches!(node.input.len(), 4 | 5)
                        && node.output.len() == 3
                        && node.input[0] == "length",
                    "unsupported pitch Loop signature/trip count: inputs={:?}, outputs={:?}",
                    node.input,
                    node.output
                );
                ensure!(
                    node.attribute.len() == 1 && node.attribute[0].name == "body",
                    "unsupported Loop attributes"
                );
                let body = node.attribute[0].g.as_mut().context("missing Loop body")?;
                rewrite_loop(body, *count)?;
                *count += 1;
                predictors += 1;
            }
            "If" if node.domain.is_empty() => {
                ensure!(
                    node.attribute.len() == 2
                        && node.attribute.iter().any(|a| a.name == "then_branch")
                        && node.attribute.iter().any(|a| a.name == "else_branch"),
                    "unsupported model selection"
                );
                for branch in &mut node.attribute {
                    rewrite_selection(
                        branch.g.as_mut().context("missing selection branch")?,
                        count,
                    )?;
                }
                predictors += 1;
            }
            _ => ensure!(
                node.attribute
                    .iter()
                    .all(|a| a.g.is_none() && a.graphs.is_empty()),
                "unsupported nested pitch graph"
            ),
        }
    }
    ensure!(
        predictors == 1,
        "expected one pitch predictor per selection branch, found {predictors}"
    );
    Ok(())
}

fn producer<'a>(graph: &'a GraphProto, value: &str) -> anyhow::Result<&'a NodeProto> {
    let mut producers = graph
        .node
        .iter()
        .filter(|n| n.output.iter().any(|s| s == value));
    let node = producers
        .next()
        .with_context(|| format!("missing producer for {value}"))?;
    ensure!(producers.next().is_none(), "ambiguous tensor producer");
    ensure!(node.domain.is_empty(), "nonstandard pitch operator");
    Ok(node)
}

fn through<'a>(graph: &'a GraphProto, value: &str, ops: &[&str]) -> anyhow::Result<&'a NodeProto> {
    let mut node = producer(graph, value)?;
    for op in ops {
        ensure!(
            node.op_type == *op && !node.input.is_empty() && node.output.len() == 1,
            "unsupported pitch path: expected {op}, got {}",
            node.op_type
        );
        node = producer(graph, &node.input[0])?;
    }
    Ok(node)
}

fn pitch_shape(value: &ValueInfoProto, concrete: bool) -> bool {
    let Some(ty) = value.type_.as_ref() else {
        return false;
    };
    let tensor = ty.tensor_type();
    tensor.elem_type == 1
        && tensor.shape.as_ref().is_some_and(|shape| {
            shape.dim.len() == 3
                && shape.dim.iter().all(|d| {
                    (d.has_dim_value() && d.dim_value() == 1) || (!concrete && d.has_dim_param())
                })
        })
}

fn rewrite_loop(body: &mut GraphProto, index: usize) -> anyhow::Result<()> {
    ensure!(
        matches!(body.input.len(), 4 | 5) && body.output.len() == 4,
        "unsupported Loop carried/scan arity"
    );
    ensure!(
        body.input[0].type_.tensor_type().elem_type == 7
            && body.input[0].type_.tensor_type().shape.dim.is_empty(),
        "Loop index must be scalar int64"
    );
    ensure!(
        body.node.iter().all(|n| n
            .attribute
            .iter()
            .all(|a| a.g.is_none() && a.graphs.is_empty())),
        "nested loop body is unsupported"
    );
    let candidates: Vec<_> = (2..body.input.len())
        .filter(|&i| pitch_shape(&body.input[i], true) && pitch_shape(&body.output[i - 1], false))
        .collect();
    ensure!(
        candidates.len() == 1,
        "missing or ambiguous singleton pitch feedback"
    );
    let carry = candidates[0];
    let pitch = &body.output[carry - 1].name;
    let head = producer(body, pitch)?;
    ensure!(
        head.op_type == "Conv" && head.input.len() == 3 && head.output.len() == 1,
        "unsupported pitch head"
    );
    let gru = through(body, &head.input[0], &["Transpose", "Squeeze"])?;
    ensure!(
        gru.op_type == "GRU" && gru.input.len() == 6 && gru.output.len() == 2,
        "unsupported pitch GRU"
    );
    let hidden = (2..body.input.len())
        .find(|&i| body.input[i].name == gru.input[5])
        .context("missing GRU hidden carry")?;
    ensure!(
        hidden != carry && gru.output[1] == body.output[hidden - 1].name,
        "GRU hidden feedback mismatch"
    );
    let concat = through(body, &gru.input[0], &["Transpose"])?;
    ensure!(
        concat.op_type == "Concat"
            && concat.input.len() == 2
            && concat.input[1] == body.input[carry].name,
        "previous pitch does not feed GRU input"
    );
    let collected = if body.input.len() == 4 {
        &body.output[3].name
    } else {
        let sequence = (2..body.input.len())
            .find(|&i| i != carry && i != hidden)
            .unwrap();
        ensure!(
            body.input[sequence].type_.has_sequence_type()
                && body.output[sequence - 1].type_.has_sequence_type(),
            "unsupported pitch collection carry"
        );
        let insert = producer(body, &body.output[sequence - 1].name)?;
        ensure!(
            insert.op_type == "SequenceInsert"
                && insert.input.len() == 2
                && insert.input[0] == body.input[sequence].name,
            "unsupported pitch sequence collection"
        );
        &insert.input[1]
    };
    let scan_head = through(body, collected, &["Gather", "Gather"])?;
    ensure!(
        std::ptr::eq(scan_head, head),
        "pitch scan and feedback have different heads"
    );
    ensure!(
        body.node.iter().filter(|n| n.op_type == "GRU").count() == 1,
        "ambiguous GRU"
    );
    let head_index = body
        .node
        .iter()
        .position(|n| std::ptr::eq(n, head))
        .unwrap();
    let pitch = pitch.clone();
    let mean = format!("azalea_pitch_mean_{index}");
    let noise = format!("azalea_pitch_step_{index}");
    let noisy = format!("azalea_pitch_noisy_{index}");
    let prefix = format!("azalea_pitch_prefix_step_{index}");
    let mask = format!("azalea_pitch_prefix_mask_step_{index}");
    let mask_value = format!("azalea_pitch_prefix_mask_value_{index}");
    body.node[head_index].output[0] = mean.clone();
    body.node.splice(
        head_index + 1..head_index + 1,
        [
            NodeProto {
                op_type: "Gather".into(),
                input: vec![NOISE_INPUT.into(), body.input[0].name.clone()],
                output: vec![noise.clone()],
                ..Default::default()
            },
            NodeProto {
                op_type: "Add".into(),
                input: vec![mean, noise],
                output: vec![noisy.clone()],
                ..Default::default()
            },
            NodeProto {
                op_type: "Gather".into(),
                input: vec![PREFIX_INPUT.into(), body.input[0].name.clone()],
                output: vec![prefix.clone()],
                ..Default::default()
            },
            NodeProto {
                op_type: "Gather".into(),
                input: vec![PREFIX_MASK_INPUT.into(), body.input[0].name.clone()],
                output: vec![mask_value.clone()],
                ..Default::default()
            },
            NodeProto {
                op_type: "Cast".into(),
                input: vec![mask_value],
                output: vec![mask.clone()],
                attribute: vec![onnx_protobuf::AttributeProto {
                    name: "to".into(),
                    type_: onnx_protobuf::attribute_proto::AttributeType::INT.into(),
                    i: 9,
                    ..Default::default()
                }],
                ..Default::default()
            },
            NodeProto {
                op_type: "Where".into(),
                input: vec![mask, prefix, noisy],
                output: vec![pitch],
                ..Default::default()
            },
        ],
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_protobuf::{AttributeProto, attribute_proto::AttributeType};
    const SAMPLE: &[u8] =
        include_bytes!("../../../../../../model/sample.vvm/predict_intonation.onnx");

    fn sample() -> ModelProto {
        ModelProto::parse_from_bytes(SAMPLE).unwrap()
    }
    fn body(model: &mut ModelProto) -> &mut GraphProto {
        model
            .graph
            .as_mut()
            .unwrap()
            .node
            .iter_mut()
            .find(|n| n.op_type == "Loop")
            .unwrap()
            .attribute[0]
            .g
            .as_mut()
            .unwrap()
    }
    fn apply(model: &ModelProto) -> anyhow::Result<Vec<u8>> {
        transform(&model.write_to_bytes().unwrap())
    }

    #[test]
    fn feedback_and_scan_consume_same_perturbed_value_and_preserve_content() {
        let before = sample();
        let mut after = ModelProto::parse_from_bytes(&apply(&before).unwrap()).unwrap();
        assert_eq!(before.graph.initializer, after.graph.initializer);
        assert_eq!(before.opset_import, after.opset_import);
        assert_eq!(before.producer_name, after.producer_name);
        assert_eq!(after.graph.input.last().unwrap().name, PREFIX_MASK_INPUT);
        let b = body(&mut after);
        let selected = producer(b, &b.output[1].name).unwrap();
        assert_eq!(selected.op_type, "Where");
        assert_eq!(
            through(b, &selected.input[0], &["Cast"]).unwrap().input,
            [PREFIX_MASK_INPUT, &b.input[0].name]
        );
        assert_eq!(
            producer(b, &selected.input[1]).unwrap().input,
            [PREFIX_INPUT, &b.input[0].name]
        );
        let add = producer(b, &selected.input[2]).unwrap();
        assert_eq!(add.op_type, "Add");
        assert_eq!(producer(b, &add.input[0]).unwrap().op_type, "Conv");
        assert_eq!(
            through(b, &b.output[3].name, &["Gather", "Gather"]).unwrap(),
            selected
        );
        let gather = producer(b, &add.input[1]).unwrap();
        assert_eq!(gather.input, [NOISE_INPUT, &b.input[0].name]);
        assert!(apply(&after).unwrap_err().to_string().contains("already"));
        after.metadata_props.last_mut().unwrap().value = "999".into();
        assert!(apply(&after).unwrap_err().to_string().contains("version"));
    }

    fn rename(graph: &mut GraphProto) {
        let rename = |name: &mut String| {
            if !name.is_empty() && name != "length" {
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
        for v in &mut graph.initializer {
            rename(&mut v.name);
        }
        for n in &mut graph.node {
            for v in n.input.iter_mut().chain(&mut n.output) {
                rename(v);
            }
            for a in &mut n.attribute {
                if let Some(g) = a.g.as_mut() {
                    self::rename(g);
                }
            }
        }
    }

    #[test]
    fn renamed_tensors_and_nested_model_selection() {
        let mut model = sample();
        rename(model.graph.as_mut().unwrap());
        assert!(apply(&model).is_ok());
        let graph = model.graph.as_mut().unwrap();
        let loop_index = graph.node.iter().position(|n| n.op_type == "Loop").unwrap();
        let loop_node = graph.node.remove(loop_index);
        let mut branch = GraphProto {
            node: vec![loop_node.clone()],
            ..Default::default()
        };
        for _ in 0..2 {
            branch = GraphProto {
                node: vec![NodeProto {
                    op_type: "If".into(),
                    input: vec!["selection_condition".into()],
                    output: loop_node.output.clone(),
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
                }],
                ..Default::default()
            };
        }
        graph.node.splice(loop_index..loop_index, branch.node);
        assert!(apply(&model).is_ok());
    }

    #[test]
    fn rejects_missing_ambiguous_malformed_and_unsupported_feedback() {
        assert!(transform(b"not protobuf").is_err());
        for mutation in 0..7 {
            let mut m = sample();
            let b = body(&mut m);
            match mutation {
                0 => b.node.retain(|n| n.op_type != "Conv"),
                1 => {
                    let head = b.node.iter().find(|n| n.op_type == "Conv").unwrap().clone();
                    b.node.push(head);
                }
                2 => {
                    b.node
                        .iter_mut()
                        .find(|n| n.op_type == "Concat")
                        .unwrap()
                        .input[1] = "other".into()
                }
                3 => b.output[3].name = b.output[2].name.clone(),
                4 => {
                    b.node
                        .iter_mut()
                        .find(|n| n.op_type == "Conv")
                        .unwrap()
                        .op_type = "Gemm".into()
                }
                5 => b.input.clear(),
                6 => b.input[2].name = NOISE_INPUT.into(),
                _ => unreachable!(),
            }
            assert!(apply(&m).is_err(), "mutation {mutation}");
        }
        let mut m = sample();
        m.graph
            .as_mut()
            .unwrap()
            .node
            .retain(|n| n.op_type != "Loop");
        assert!(apply(&m).is_err());
        let mut m = sample();
        let n = m
            .graph
            .node
            .iter()
            .find(|n| n.op_type == "Loop")
            .unwrap()
            .clone();
        m.graph.as_mut().unwrap().node.push(n);
        assert!(apply(&m).is_err());
    }
}
