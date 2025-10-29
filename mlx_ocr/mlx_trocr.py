from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.transformer import MultiHeadAttention


@dataclass
class VisionEncoderConfig:
    image_size: int
    patch_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float
    hidden_dropout_prob: float
    attention_dropout_prob: float
    num_channels: int = 3
    qkv_bias: bool = False

    @property
    def num_patches(self) -> int:
        grid = self.image_size // self.patch_size
        return grid * grid


@dataclass
class DecoderConfig:
    hidden_size: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    decoder_layers: int
    dropout: float
    activation_dropout: float
    attention_dropout: float
    layer_norm_eps: float
    max_position_embeddings: int
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    cross_attention_hidden_size: int
    layernorm_embedding: bool = False
    scale_embedding: bool = False
    use_learned_position_embeddings: bool = True
    tie_word_embeddings: bool = False


@dataclass
class TrOCRConfig:
    vision: VisionEncoderConfig
    decoder: DecoderConfig
    max_length: int = 64

    @property
    def decoder_start_token_id(self) -> int:
        return self.decoder.decoder_start_token_id

    @property
    def pad_token_id(self) -> int:
        return self.decoder.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.decoder.eos_token_id


def config_from_dict(config_dict: dict) -> TrOCRConfig:
    encoder_cfg = config_dict["encoder"]
    decoder_cfg = config_dict["decoder"]

    vision = VisionEncoderConfig(
        image_size=encoder_cfg["image_size"],
        patch_size=encoder_cfg["patch_size"],
        hidden_size=encoder_cfg["hidden_size"],
        num_hidden_layers=encoder_cfg["num_hidden_layers"],
        num_attention_heads=encoder_cfg["num_attention_heads"],
        intermediate_size=encoder_cfg["intermediate_size"],
        layer_norm_eps=encoder_cfg.get("layer_norm_eps", 1e-5),
        hidden_dropout_prob=encoder_cfg.get("hidden_dropout_prob", 0.0),
        attention_dropout_prob=encoder_cfg.get("attention_probs_dropout_prob", 0.0),
        num_channels=encoder_cfg.get("num_channels", 3),
        qkv_bias=encoder_cfg.get("qkv_bias", False),
    )

    decoder = DecoderConfig(
        hidden_size=decoder_cfg["d_model"],
        decoder_attention_heads=decoder_cfg["decoder_attention_heads"],
        decoder_ffn_dim=decoder_cfg["decoder_ffn_dim"],
        decoder_layers=decoder_cfg["decoder_layers"],
        dropout=decoder_cfg.get("dropout", 0.0),
        activation_dropout=decoder_cfg.get("activation_dropout", 0.0),
        attention_dropout=decoder_cfg.get("attention_dropout", 0.0),
        layer_norm_eps=decoder_cfg.get("layer_norm_eps", 1e-5),
        max_position_embeddings=decoder_cfg["max_position_embeddings"],
        vocab_size=decoder_cfg["vocab_size"],
        pad_token_id=decoder_cfg["pad_token_id"],
        bos_token_id=decoder_cfg["bos_token_id"],
        eos_token_id=decoder_cfg["eos_token_id"],
        decoder_start_token_id=decoder_cfg["decoder_start_token_id"],
        cross_attention_hidden_size=decoder_cfg.get(
            "cross_attention_hidden_size", decoder_cfg["d_model"]
        ),
        layernorm_embedding=decoder_cfg.get("layernorm_embedding", False),
        scale_embedding=decoder_cfg.get("scale_embedding", False),
        use_learned_position_embeddings=decoder_cfg.get(
            "use_learned_position_embeddings", True
        ),
        tie_word_embeddings=decoder_cfg.get("tie_word_embeddings", False),
    )

    max_length = config_dict.get("max_length", decoder_cfg.get("max_length", 64))
    return TrOCRConfig(vision=vision, decoder=decoder, max_length=max_length)


def load_config_dict(config_path: str | Path) -> TrOCRConfig:
    config_dict = json.loads(Path(config_path).read_text())
    return config_from_dict(config_dict)


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.pos_embedding = mx.zeros((1, config.num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.transformer = nn.TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
        )

        # Override layer norms to match Hugging Face eps.
        for layer in self.transformer.layers:
            layer.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            layer.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if hasattr(layer, "attention"):
                layer.attention.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.transformer.ln = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        x = self.patch_embed(pixel_values)
        batch, height, width, channels = x.shape
        x = x.reshape(batch, height * width, channels)

        cls_tokens = mx.broadcast_to(
            self.cls_token, (batch, self.cls_token.shape[1], self.cls_token.shape[2])
        )
        x = mx.concatenate([cls_tokens, x], axis=1)

        pos_embedding = mx.broadcast_to(
            self.pos_embedding,
            (batch, self.pos_embedding.shape[1], self.pos_embedding.shape[2]),
        )
        x = x + pos_embedding
        x = self.dropout(x)

        return self.transformer(x, mask=None)


class TrOCRAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        kdim = kdim or embed_dim
        vdim = vdim or embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(kdim, embed_dim, bias=True)
        self.v_proj = nn.Linear(vdim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _shape(self, tensor: mx.array) -> mx.array:
        bsz, seq_len, _ = tensor.shape
        tensor = mx.unflatten(tensor, -1, (self.num_heads, self.head_dim))
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor, (bsz, seq_len)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        key_states_source = key_value_states if key_value_states is not None else hidden_states

        query = self.q_proj(hidden_states)
        key = self.k_proj(key_states_source)
        value = self.v_proj(key_states_source)

        query, (bsz, tgt_len) = self._shape(query)
        key, (_, src_len) = self._shape(key)
        value, _ = self._shape(value)

        mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                mask = mx.broadcast_to(
                    attention_mask,
                    (bsz, self.num_heads, attention_mask.shape[2], attention_mask.shape[3]),
                )
            else:
                mask = attention_mask

        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scaling, mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, tgt_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output


class TrOCRDecoderLayer(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.activation = nn.layers.activations.gelu

        self.self_attn = TrOCRAttention(
            embed_dim=config.hidden_size,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.encoder_attn = TrOCRAttention(
            embed_dim=config.hidden_size,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            kdim=config.cross_attention_hidden_size,
            vdim=config.cross_attention_hidden_size,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.fc1 = nn.Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        self_attn_mask: mx.array,
        encoder_hidden_states: Optional[mx.array],
        encoder_attention_mask: Optional[mx.array],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states, attention_mask=self_attn_mask
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class TrOCRDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = embed_scale
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm_embedding = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.layernorm_embedding
            else None
        )

        self.layers = [
            TrOCRDecoderLayer(config) for _ in range(config.decoder_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.padding_idx = config.pad_token_id

    def _create_causal_mask(self, seq_len: int, dtype: mx.Dtype) -> mx.array:
        mask = MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=dtype)
        mask = mask.reshape(1, 1, seq_len, seq_len)
        return mask

    def _expand_mask(
        self, mask: mx.array, dtype: mx.Dtype, tgt_len: Optional[int] = None
    ) -> mx.array:
        if mask is None:
            return None
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :]
        expanded_mask = expanded_mask.astype(dtype)
        expanded_mask = (1.0 - expanded_mask) * mx.finfo(dtype).min
        expanded_mask = mx.broadcast_to(
            expanded_mask, (bsz, 1, tgt_len, src_len)
        )
        return expanded_mask

    def _create_position_ids(self, input_ids: mx.array) -> mx.array:
        mask = (input_ids != self.padding_idx).astype(mx.int32)
        incremental = mx.cumsum(mask, axis=1)
        position_ids = incremental + self.padding_idx
        position_ids = position_ids * mask + self.padding_idx
        return position_ids

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: Optional[mx.array],
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        input_ids = input_ids.astype(mx.int32)
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        position_ids = self._create_position_ids(input_ids)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeddings

        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        attention_mask = (
            attention_mask if attention_mask is not None else mx.ones_like(input_ids)
        )
        dtype = hidden_states.dtype

        causal_mask = self._create_causal_mask(input_ids.shape[1], dtype)
        padding_mask = self._expand_mask(attention_mask, dtype)
        if padding_mask is not None:
            self_attn_mask = causal_mask + padding_mask
        else:
            self_attn_mask = causal_mask

        encoder_mask = None
        if encoder_attention_mask is not None:
            encoder_mask = self._expand_mask(
                encoder_attention_mask, dtype, tgt_len=input_ids.shape[1]
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                self_attn_mask=self_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_mask,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class MLXTrOCRModel(nn.Module):
    def __init__(self, config: TrOCRConfig):
        super().__init__()
        self.config = config
        self.encoder = VisionEncoder(config.vision)
        self.decoder = TrOCRDecoder(config.decoder)
        self.lm_head = nn.Linear(
            config.decoder.hidden_size, config.decoder.vocab_size, bias=False
        )

        if config.decoder.tie_word_embeddings:
            self.lm_head.weight = self.decoder.embed_tokens.weight

    def encode(
        self, pixel_values: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        del attention_mask
        return self.encoder(pixel_values)

    def decode(
        self,
        decoder_input_ids: mx.array,
        encoder_hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.decoder(
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.lm_head(hidden_states)

    def __call__(
        self,
        pixel_values: mx.array,
        decoder_input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        encoder_hidden_states = self.encode(pixel_values, attention_mask)
        logits = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return logits

    def generate(
        self,
        pixel_values: mx.array,
        attention_mask: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        max_length: Optional[int] = None,
    ) -> mx.array:
        max_length = max_length or self.config.max_length
        encoder_hidden_states = self.encode(pixel_values, attention_mask)

        batch_size = pixel_values.shape[0]
        start_token = mx.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=mx.int32,
        )
        generated = start_token
        finished = mx.zeros((batch_size,), dtype=mx.bool_)

        for _ in range(max_length):
            logits = self.decode(
                generated,
                encoder_hidden_states,
                attention_mask=None,
                encoder_attention_mask=encoder_attention_mask,
            )
            next_token_logits = logits[:, -1, :]
            next_tokens = mx.argmax(next_token_logits, axis=-1).astype(mx.int32)
            next_tokens = mx.where(
                finished,
                mx.full(next_tokens.shape, self.config.pad_token_id, dtype=mx.int32),
                next_tokens,
            )
            generated = mx.concatenate(
                [generated, next_tokens[:, None]], axis=1
            )
            finished = mx.logical_or(
                finished, next_tokens == self.config.eos_token_id
            )
            if mx.all(finished):
                break

        return generated
