#!/usr/bin/env python3
"""
Convert a Hugging Face TrOCR checkpoint to MLX `.npz` weights.

Example:
    python convert_to_mlx.py \
        --model microsoft/trocr-base-printed \
        --output weights/trocr-base-printed.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import mlx.core as mx
import torch
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from mlx_ocr import MLXTrOCRModel, config_from_dict


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def transpose_conv(weight):
    return to_numpy(weight).transpose(0, 2, 3, 1)


def collect_encoder_weights(
    state_dict: Dict[str, torch.Tensor], config_dict: dict
) -> Iterable[Tuple[str, mx.array]]:
    weights = []

    weights.append(
        ("encoder.cls_token", mx.array(to_numpy(state_dict["encoder.embeddings.cls_token"])))
    )
    weights.append(
        (
            "encoder.pos_embedding",
            mx.array(to_numpy(state_dict["encoder.embeddings.position_embeddings"])),
        )
    )
    weights.append(
        (
            "encoder.patch_embed.weight",
            mx.array(transpose_conv(state_dict["encoder.embeddings.patch_embeddings.projection.weight"])),
        )
    )
    bias_key = "encoder.embeddings.patch_embeddings.projection.bias"
    if bias_key in state_dict:
        weights.append(
            ("encoder.patch_embed.bias", mx.array(to_numpy(state_dict[bias_key])))
        )

    num_hidden_layers = config_dict["encoder"]["num_hidden_layers"]
    for idx in range(num_hidden_layers):
        prefix = f"encoder.encoder.layer.{idx}"
        target = f"encoder.transformer.layers.{idx}"

        weights.extend(
            [
                (
                    f"{target}.attention.query_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.attention.query.weight"])),
                ),
                (
                    f"{target}.attention.query_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.attention.query.bias"])),
                ),
                (
                    f"{target}.attention.key_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.attention.key.weight"])),
                ),
                (
                    f"{target}.attention.key_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.attention.key.bias"])),
                ),
                (
                    f"{target}.attention.value_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.attention.value.weight"])),
                ),
                (
                    f"{target}.attention.value_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.attention.value.bias"])),
                ),
                (
                    f"{target}.attention.out_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.output.dense.weight"])),
                ),
                (
                    f"{target}.attention.out_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.attention.output.dense.bias"])),
                ),
                (
                    f"{target}.ln1.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.layernorm_before.weight"])),
                ),
                (
                    f"{target}.ln1.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.layernorm_before.bias"])),
                ),
                (
                    f"{target}.ln2.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.layernorm_after.weight"])),
                ),
                (
                    f"{target}.ln2.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.layernorm_after.bias"])),
                ),
                (
                    f"{target}.linear1.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.intermediate.dense.weight"])),
                ),
                (
                    f"{target}.linear1.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.intermediate.dense.bias"])),
                ),
                (
                    f"{target}.linear2.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.output.dense.weight"])),
                ),
                (
                    f"{target}.linear2.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.output.dense.bias"])),
                ),
            ]
        )

    weights.extend(
        [
            (
                "encoder.transformer.ln.weight",
                mx.array(to_numpy(state_dict["encoder.layernorm.weight"])),
            ),
            (
                "encoder.transformer.ln.bias",
                mx.array(to_numpy(state_dict["encoder.layernorm.bias"])),
            ),
        ]
    )

    return weights


def collect_decoder_weights(
    state_dict: Dict[str, torch.Tensor], config_dict: dict
) -> Iterable[Tuple[str, mx.array]]:
    weights = []
    decoder_prefix = "decoder.model.decoder"

    weights.append(
        (
            "decoder.embed_tokens.weight",
            mx.array(to_numpy(state_dict[f"{decoder_prefix}.embed_tokens.weight"])),
        )
    )
    weights.append(
        (
            "decoder.embed_positions.weight",
            mx.array(to_numpy(state_dict[f"{decoder_prefix}.embed_positions.weight"])),
        )
    )

    layernorm_embedding_w = f"{decoder_prefix}.layernorm_embedding.weight"
    if layernorm_embedding_w in state_dict:
        weights.append(
            (
                "decoder.layernorm_embedding.weight",
                mx.array(to_numpy(state_dict[layernorm_embedding_w])),
            )
        )
    layernorm_embedding_b = f"{decoder_prefix}.layernorm_embedding.bias"
    if layernorm_embedding_b in state_dict:
        weights.append(
            (
                "decoder.layernorm_embedding.bias",
                mx.array(to_numpy(state_dict[layernorm_embedding_b])),
            )
        )

    num_layers = config_dict["decoder"]["decoder_layers"]
    for idx in range(num_layers):
        prefix = f"{decoder_prefix}.layers.{idx}"
        target = f"decoder.layers.{idx}"

        weights.extend(
            [
                (
                    f"{target}.self_attn.q_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.q_proj.weight"])),
                ),
                (
                    f"{target}.self_attn.q_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.q_proj.bias"])),
                ),
                (
                    f"{target}.self_attn.k_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.k_proj.weight"])),
                ),
                (
                    f"{target}.self_attn.k_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.k_proj.bias"])),
                ),
                (
                    f"{target}.self_attn.v_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.v_proj.weight"])),
                ),
                (
                    f"{target}.self_attn.v_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.v_proj.bias"])),
                ),
                (
                    f"{target}.self_attn.out_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.out_proj.weight"])),
                ),
                (
                    f"{target}.self_attn.out_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn.out_proj.bias"])),
                ),
                (
                    f"{target}.self_attn_layer_norm.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn_layer_norm.weight"])),
                ),
                (
                    f"{target}.self_attn_layer_norm.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.self_attn_layer_norm.bias"])),
                ),
                (
                    f"{target}.encoder_attn.q_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.q_proj.weight"])),
                ),
                (
                    f"{target}.encoder_attn.q_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.q_proj.bias"])),
                ),
                (
                    f"{target}.encoder_attn.k_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.k_proj.weight"])),
                ),
                (
                    f"{target}.encoder_attn.k_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.k_proj.bias"])),
                ),
                (
                    f"{target}.encoder_attn.v_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.v_proj.weight"])),
                ),
                (
                    f"{target}.encoder_attn.v_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.v_proj.bias"])),
                ),
                (
                    f"{target}.encoder_attn.out_proj.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.out_proj.weight"])),
                ),
                (
                    f"{target}.encoder_attn.out_proj.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn.out_proj.bias"])),
                ),
                (
                    f"{target}.encoder_attn_layer_norm.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn_layer_norm.weight"])),
                ),
                (
                    f"{target}.encoder_attn_layer_norm.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.encoder_attn_layer_norm.bias"])),
                ),
                (
                    f"{target}.fc1.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.fc1.weight"])),
                ),
                (
                    f"{target}.fc1.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.fc1.bias"])),
                ),
                (
                    f"{target}.fc2.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.fc2.weight"])),
                ),
                (
                    f"{target}.fc2.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.fc2.bias"])),
                ),
                (
                    f"{target}.final_layer_norm.weight",
                    mx.array(to_numpy(state_dict[f"{prefix}.final_layer_norm.weight"])),
                ),
                (
                    f"{target}.final_layer_norm.bias",
                    mx.array(to_numpy(state_dict[f"{prefix}.final_layer_norm.bias"])),
                ),
            ]
        )

    weights.append(
        (
            "decoder.final_layer_norm.weight",
            mx.array(to_numpy(state_dict[f"{decoder_prefix}.final_layer_norm.weight"])),
        )
    )
    weights.append(
        (
            "decoder.final_layer_norm.bias",
            mx.array(to_numpy(state_dict[f"{decoder_prefix}.final_layer_norm.bias"])),
        )
    )

    weights.append(
        (
            "lm_head.weight",
            mx.array(to_numpy(state_dict["decoder.output_projection.weight"])),
        )
    )

    return weights


def convert(model_name: str, output_path: Path, config_output: Path | None = None):
    hf_config = VisionEncoderDecoderConfig.from_pretrained(model_name)
    config_dict = json.loads(hf_config.to_json_string())
    trocr_config = config_from_dict(config_dict)

    hf_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    hf_model.eval()
    state_dict = hf_model.state_dict()

    mlx_model = MLXTrOCRModel(trocr_config)

    weights = []
    weights.extend(collect_encoder_weights(state_dict, config_dict))
    weights.extend(collect_decoder_weights(state_dict, config_dict))

    mlx_model.load_weights(weights)
    mx.eval(mlx_model.parameters())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_model.save_weights(str(output_path))

    if config_output is not None:
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(json.dumps(config_dict, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Convert TrOCR weights for MLX.")
    parser.add_argument(
        "--model",
        default="microsoft/trocr-base-printed",
        help="Hugging Face model identifier or local path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the MLX weights (.npz).",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        help="Optional path to save the resolved configuration JSON.",
    )
    args = parser.parse_args()

    convert(args.model, args.output, args.config_output)
    print(f"Saved MLX weights to {args.output}")
    if args.config_output:
        print(f"Saved configuration to {args.config_output}")


if __name__ == "__main__":
    main()
