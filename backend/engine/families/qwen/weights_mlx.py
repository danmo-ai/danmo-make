"""Qwen-Image 权重映射（MLX）— HF/diffusers → DanQing 嵌套/扁平键。

Registry 与 Pipeline 通过 ``weights.py`` 懒加载本模块，避免在非 ``*_mlx`` 路径顶层 ``import mlx``。
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx


@dataclass
class WeightTarget:
    to_pattern: str
    from_pattern: List[str]
    transform: Optional[Callable[[mx.array], mx.array]] = None
    required: bool = True
    max_blocks: Optional[int] = None


class WeightTransforms:
    @staticmethod
    def reshape_gamma_to_1d(tensor: mx.array) -> mx.array:
        if len(tensor.shape) > 1:
            return mx.reshape(tensor, (tensor.shape[0],))
        return tensor

    @staticmethod
    def transpose_patch_embed(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 5:
            return tensor.transpose(0, 2, 3, 4, 1)
        return tensor

    @staticmethod
    def transpose_conv3d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 5:
            return tensor.transpose(0, 2, 3, 4, 1)
        return tensor

    @staticmethod
    def transpose_conv2d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 4:
            return tensor.transpose(0, 2, 3, 1)
        return tensor

    @staticmethod
    def transpose_conv_transpose2d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 4:
            return tensor.transpose(1, 2, 3, 0)
        return tensor


class WeightMapper:
    @staticmethod
    def apply_mapping(
        hf_weights: Dict[str, mx.array],
        mapping: List[WeightTarget],
        num_blocks: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> Dict:
        # Auto-detect number of blocks if not provided
        if num_blocks is None:
            num_blocks = WeightMapper._detect_num_blocks(hf_weights)

        # Auto-detect number of layers if not provided
        if num_layers is None:
            num_layers = WeightMapper._detect_num_layers(hf_weights)

        # Build flat mapping: HF pattern -> [(MLX path, transform), ...] (supports one-to-many)
        flat_mapping = WeightMapper._build_flat_mapping(mapping, num_blocks, num_layers)

        # Map weights
        mapped_weights = {}
        mapped_count = 0
        skipped_count = 0

        for hf_key, hf_tensor in hf_weights.items():
            # Try to find matching mappings (can be multiple targets for one source)
            targets = flat_mapping.get(hf_key, [])

            if targets:
                for mlx_path, transform in targets:
                    # Apply transform if specified
                    tensor = hf_tensor
                    if transform:
                        tensor = transform(tensor)

                    # Build nested structure
                    WeightMapper._set_nested_value(mapped_weights, mlx_path, tensor)
                    mapped_count += 1
            else:
                # Weight not in mapping - might be intentionally skipped (e.g., lm_head)
                # or optional weight (e.g., conv_shortcut) - that's OK
                skipped_count += 1

        # MLX affine-quant companions mirror mapped ``*.weight`` paths (.scales / .biases).
        for hf_key, hf_tensor in hf_weights.items():
            suffix = hf_key.rsplit(".", 1)[-1]
            if suffix not in ("scales", "biases"):
                continue
            weight_hf = f"{hf_key.rsplit('.', 1)[0]}.weight"
            targets = flat_mapping.get(weight_hf, [])
            if not targets:
                continue
            for mlx_path, _ in targets:
                base_mlx = mlx_path.rsplit(".", 1)[0]
                companion = f"{base_mlx}.{suffix}"
                WeightMapper._set_nested_value(mapped_weights, companion, hf_tensor)
                mapped_count += 1

        # Optional: uncomment for debugging
        # print(f"✅ Mapped {mapped_count} weights, skipped {skipped_count}")

        return mapped_weights

    @staticmethod
    def _detect_num_blocks(hf_weights: Dict[str, mx.array]) -> int:
        block_numbers = set()
        for key in hf_weights.keys():
            # Match pattern: transformer_blocks.{number}.something
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))
                continue
            # Match pattern: single_transformer_blocks.{number}.something
            match = re.search(r"single_transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))

        if block_numbers:
            return max(block_numbers) + 1  # Blocks are 0-indexed
        return 0

    @staticmethod
    def _detect_num_layers(hf_weights: Dict[str, mx.array]) -> int:
        layer_numbers = set()
        for key in hf_weights.keys():
            # HF Qwen2.5-VL: model.layers.{n}.* ; mlx-community bundles: encoder.layers.{n}.*
            match = re.search(r"(?:model|encoder)\.layers\.(\d+)\.", key)
            if match:
                layer_numbers.add(int(match.group(1)))

        if layer_numbers:
            return max(layer_numbers) + 1  # Layers are 0-indexed
        return 28  # Default 28 layers for Qwen text encoder

    @staticmethod
    def _build_flat_mapping(
        mapping: List[WeightTarget], num_blocks: int = 0, num_layers: int = 28
    ) -> Dict[str, List[tuple[str, Optional[Callable[[mx.array], mx.array]]]]]:
        flat: Dict[str, List[tuple[str, Optional[Callable[[mx.array], mx.array]]]]] = {}

        def add_mapping(hf_key: str, mlx_path: str, transform: Optional[Callable[[mx.array], mx.array]]):
            if hf_key not in flat:
                flat[hf_key] = []
            flat[hf_key].append((mlx_path, transform))

        for target in mapping:
            # Expand placeholders for each pattern
            for hf_pattern in target.from_pattern:
                # Check which placeholders are present in BOTH patterns
                hf_has_block = "{block}" in hf_pattern
                to_has_block = "{block}" in target.to_pattern
                has_i = "{i}" in hf_pattern or "{i}" in target.to_pattern
                has_res = "{res}" in hf_pattern or "{res}" in target.to_pattern
                has_layer = "{layer}" in hf_pattern or "{layer}" in target.to_pattern

                # Handle multiple placeholders together
                if (hf_has_block or to_has_block) and has_res:
                    # Up blocks: expand both {block} and {res}
                    max_blocks = num_blocks if num_blocks > 0 else 4  # Default 4 for up_blocks
                    for block_num in range(max_blocks):
                        for res in range(3):  # 3 resnets per up_block
                            concrete_hf = hf_pattern.replace("{block}", str(block_num)).replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{block}", str(block_num)).replace(
                                "{res}", str(res)
                            )
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif hf_has_block and to_has_block:
                    # Both have {block} - standard one-to-one expansion
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    elif "visual.blocks" in hf_pattern or "visual.blocks" in target.to_pattern:
                        max_blocks = 32  # Visual blocks are always 32
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 4
                    for block_num in range(max_blocks):
                        concrete_hf = hf_pattern.replace("{block}", str(block_num))
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif to_has_block and not hf_has_block:
                    # One-to-many: single HF key maps to multiple MLX targets (e.g., relative_attention_bias)
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 24  # Default for T5
                    for block_num in range(max_blocks):
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(hf_pattern, concrete_mlx, target.transform)
                elif has_layer:
                    # Expand {layer} for text encoder layers or visual blocks
                    max_layers = num_layers if num_layers > 0 else 28  # Default 28 for text encoder
                    for layer_num in range(max_layers):
                        concrete_hf = hf_pattern.replace("{layer}", str(layer_num))
                        concrete_mlx = target.to_pattern.replace("{layer}", str(layer_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_i:
                    # Expand {i} only (for mid_block resnets)
                    for i in range(2):  # 2 resnets in mid_block
                        concrete_hf = hf_pattern.replace("{i}", str(i))
                        concrete_mlx = target.to_pattern.replace("{i}", str(i))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_res:
                    # This shouldn't happen for VAE (encoder down_blocks are explicit)
                    # But handle it just in case
                    if "up_block" in hf_pattern:
                        for res in range(3):
                            concrete_hf = hf_pattern.replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{res}", str(res))
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                else:
                    # No placeholder, use as-is
                    add_mapping(hf_pattern, target.to_pattern, target.transform)

        return flat

    @staticmethod
    def _set_nested_value(d: Dict, path: str, value: mx.array):
        parts = path.split(".")
        current = d
        i = 0

        while i < len(parts) - 1:
            part = parts[i]

            # Check if next part is a digit (list index)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                # This is a list, ensure it exists
                if part not in current:
                    current[part] = []
                # Ensure list is large enough
                idx = int(parts[i + 1])
                while len(current[part]) <= idx:
                    current[part].append({})
                current = current[part][idx]
                # Skip both the key and the index
                i += 2
            else:
                # Regular dict key
                if part not in current:
                    current[part] = {}
                current = current[part]
                i += 1

        # Set final value
        final_key = parts[-1]
        current[final_key] = value


class QwenWeightMapping:
    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="img_in.weight",
                from_pattern=["img_in.weight"],
            ),
            WeightTarget(
                to_pattern="img_in.bias",
                from_pattern=["img_in.bias"],
            ),
            WeightTarget(
                to_pattern="txt_norm.weight",
                from_pattern=["txt_norm.weight"],
            ),
            WeightTarget(
                to_pattern="txt_in.weight",
                from_pattern=["txt_in.weight"],
            ),
            WeightTarget(
                to_pattern="txt_in.bias",
                from_pattern=["txt_in.bias"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_1.weight",
                from_pattern=["time_text_embed.timestep_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_1.bias",
                from_pattern=["time_text_embed.timestep_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_2.weight",
                from_pattern=["time_text_embed.timestep_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_2.bias",
                from_pattern=["time_text_embed.timestep_embedder.linear_2.bias"],
            ),
            WeightTarget(
                to_pattern="norm_out.linear.weight",
                from_pattern=["norm_out.linear.weight"],
            ),
            WeightTarget(
                to_pattern="norm_out.linear.bias",
                from_pattern=["norm_out.linear.bias"],
            ),
            WeightTarget(
                to_pattern="proj_out.weight",
                from_pattern=["proj_out.weight"],
            ),
            WeightTarget(
                to_pattern="proj_out.bias",
                from_pattern=["proj_out.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_q.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_k.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_v.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_added_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_added_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_added_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_added_k.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.attn_to_out.0.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.attn_to_out.0.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.img_mod_linear.weight",
                from_pattern=["transformer_blocks.{block}.img_mod.1.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.img_mod_linear.bias",
                from_pattern=["transformer_blocks.{block}.img_mod.1.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.txt_mod_linear.weight",
                from_pattern=["transformer_blocks.{block}.txt_mod.1.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.txt_mod_linear.bias",
                from_pattern=["transformer_blocks.{block}.txt_mod.1.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.img_ff.mlp_in.weight",
                from_pattern=["transformer_blocks.{block}.img_mlp.net.0.proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.img_ff.mlp_in.bias",
                from_pattern=["transformer_blocks.{block}.img_mlp.net.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.img_ff.mlp_out.weight",
                from_pattern=["transformer_blocks.{block}.img_mlp.net.2.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.img_ff.mlp_out.bias",
                from_pattern=["transformer_blocks.{block}.img_mlp.net.2.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.txt_ff.mlp_in.weight",
                from_pattern=["transformer_blocks.{block}.txt_mlp.net.0.proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.txt_ff.mlp_in.bias",
                from_pattern=["transformer_blocks.{block}.txt_mlp.net.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.txt_ff.mlp_out.weight",
                from_pattern=["transformer_blocks.{block}.txt_mlp.net.2.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.txt_ff.mlp_out.bias",
                from_pattern=["transformer_blocks.{block}.txt_mlp.net.2.bias"],
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="decoder.conv_in.conv3d.weight",
                from_pattern=["decoder.conv_in.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.conv_in.conv3d.bias",
                from_pattern=["decoder.conv_in.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.conv_out.conv3d.weight",
                from_pattern=["decoder.conv_out.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.conv_out.conv3d.bias",
                from_pattern=["decoder.conv_out.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.norm_out.weight",
                from_pattern=["decoder.norm_out.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="post_quant_conv.conv3d.weight",
                from_pattern=["post_quant_conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="post_quant_conv.conv3d.bias",
                from_pattern=["post_quant_conv.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv1.conv3d.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv1.conv3d.bias",
                from_pattern=["decoder.mid_block.resnets.{i}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv2.conv3d.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv2.conv3d.bias",
                from_pattern=["decoder.mid_block.resnets.{i}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.norm1.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.norm2.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.norm.weight",
                from_pattern=["decoder.mid_block.attentions.0.norm.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_qkv.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_qkv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_qkv.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.proj.weight",
                from_pattern=["decoder.mid_block.attentions.0.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.proj.bias",
                from_pattern=["decoder.mid_block.attentions.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.conv1.conv3d.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.conv1.conv3d.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.conv2.conv3d.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.conv2.conv3d.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.norm1.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.norm2.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.skip_conv.conv3d.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.resnets.{res}.skip_conv.conv3d.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.upsamplers.0.resample_conv.weight",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.resample.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.upsamplers.0.resample_conv.bias",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.resample.1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.upsamplers.0.time_conv.conv3d.weight",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.time_conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.up_block{block}.upsamplers.0.time_conv.conv3d.bias",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.time_conv.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_in.conv3d.weight",
                from_pattern=["encoder.conv_in.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.conv_in.conv3d.bias",
                from_pattern=["encoder.conv_in.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_out.conv3d.weight",
                from_pattern=["encoder.conv_out.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.conv_out.conv3d.bias",
                from_pattern=["encoder.conv_out.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.norm_out.weight",
                from_pattern=["encoder.norm_out.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.norm.weight",
                from_pattern=["encoder.mid_block.attentions.0.norm.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_qkv.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_qkv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_qkv.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.proj.weight",
                from_pattern=["encoder.mid_block.attentions.0.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.proj.bias",
                from_pattern=["encoder.mid_block.attentions.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv1.conv3d.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv1.conv3d.bias",
                from_pattern=["encoder.mid_block.resnets.{i}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv2.conv3d.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv2.conv3d.bias",
                from_pattern=["encoder.mid_block.resnets.{i}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.norm1.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.norm2.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.0.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.0.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.0.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.0.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.0.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.0.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.0.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.0.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.0.norm1.weight",
                from_pattern=["encoder.down_blocks.0.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.0.norm2.weight",
                from_pattern=["encoder.down_blocks.0.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.1.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.1.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.1.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.1.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.1.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.1.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.1.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.1.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.1.norm1.weight",
                from_pattern=["encoder.down_blocks.1.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.resnets.1.norm2.weight",
                from_pattern=["encoder.down_blocks.1.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.downsamplers.0.resample_conv.weight",
                from_pattern=["encoder.down_blocks.2.resample.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.0.downsamplers.0.resample_conv.bias",
                from_pattern=["encoder.down_blocks.2.resample.1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.3.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.3.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.3.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.3.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.norm1.weight",
                from_pattern=["encoder.down_blocks.3.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.norm2.weight",
                from_pattern=["encoder.down_blocks.3.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.skip_conv.conv3d.weight",
                from_pattern=["encoder.down_blocks.3.conv_shortcut.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.0.skip_conv.conv3d.bias",
                from_pattern=["encoder.down_blocks.3.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.1.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.4.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.1.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.4.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.1.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.4.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.1.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.4.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.1.norm1.weight",
                from_pattern=["encoder.down_blocks.4.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.resnets.1.norm2.weight",
                from_pattern=["encoder.down_blocks.4.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.downsamplers.0.resample_conv.weight",
                from_pattern=["encoder.down_blocks.5.resample.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.1.downsamplers.0.resample_conv.bias",
                from_pattern=["encoder.down_blocks.5.resample.1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.6.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.6.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.6.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.6.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.norm1.weight",
                from_pattern=["encoder.down_blocks.6.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.norm2.weight",
                from_pattern=["encoder.down_blocks.6.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.skip_conv.conv3d.weight",
                from_pattern=["encoder.down_blocks.6.conv_shortcut.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.0.skip_conv.conv3d.bias",
                from_pattern=["encoder.down_blocks.6.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.1.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.7.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.1.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.7.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.1.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.7.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.1.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.7.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.1.norm1.weight",
                from_pattern=["encoder.down_blocks.7.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.resnets.1.norm2.weight",
                from_pattern=["encoder.down_blocks.7.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.downsamplers.0.resample_conv.weight",
                from_pattern=["encoder.down_blocks.8.resample.1.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.downsamplers.0.resample_conv.bias",
                from_pattern=["encoder.down_blocks.8.resample.1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.downsamplers.0.time_conv.conv3d.weight",
                from_pattern=["encoder.down_blocks.8.time_conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.2.downsamplers.0.time_conv.conv3d.bias",
                from_pattern=["encoder.down_blocks.8.time_conv.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.0.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.9.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.0.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.9.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.0.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.9.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.0.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.9.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.0.norm1.weight",
                from_pattern=["encoder.down_blocks.9.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.0.norm2.weight",
                from_pattern=["encoder.down_blocks.9.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.1.conv1.conv3d.weight",
                from_pattern=["encoder.down_blocks.10.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.1.conv1.conv3d.bias",
                from_pattern=["encoder.down_blocks.10.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.1.conv2.conv3d.weight",
                from_pattern=["encoder.down_blocks.10.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.1.conv2.conv3d.bias",
                from_pattern=["encoder.down_blocks.10.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.1.norm1.weight",
                from_pattern=["encoder.down_blocks.10.norm1.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.3.resnets.1.norm2.weight",
                from_pattern=["encoder.down_blocks.10.norm2.gamma"],
                transform=WeightTransforms.reshape_gamma_to_1d,
            ),
            WeightTarget(
                to_pattern="quant_conv.conv3d.weight",
                from_pattern=["quant_conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="quant_conv.conv3d.bias",
                from_pattern=["quant_conv.bias"],
            ),
        ]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="encoder.embed_tokens.weight",
                from_pattern=["model.embed_tokens.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.norm.weight",
                from_pattern=["model.norm.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.input_layernorm.weight",
                from_pattern=["model.layers.{layer}.input_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.post_attention_layernorm.weight",
                from_pattern=["model.layers.{layer}.post_attention_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.q_proj.weight",
                from_pattern=["model.layers.{layer}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.q_proj.bias",
                from_pattern=["model.layers.{layer}.self_attn.q_proj.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.k_proj.weight",
                from_pattern=["model.layers.{layer}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.k_proj.bias",
                from_pattern=["model.layers.{layer}.self_attn.k_proj.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.v_proj.weight",
                from_pattern=["model.layers.{layer}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.v_proj.bias",
                from_pattern=["model.layers.{layer}.self_attn.v_proj.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.self_attn.o_proj.weight",
                from_pattern=["model.layers.{layer}.self_attn.o_proj.weight"],
            ),
            # MLP
            WeightTarget(
                to_pattern="encoder.layers.{layer}.mlp.gate_proj.weight",
                from_pattern=["model.layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.mlp.up_proj.weight",
                from_pattern=["model.layers.{layer}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.layers.{layer}.mlp.down_proj.weight",
                from_pattern=["model.layers.{layer}.mlp.down_proj.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.visual.patch_embed.proj.weight",
                from_pattern=["visual.patch_embed.proj.weight"],
                transform=WeightTransforms.transpose_patch_embed,
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.attn.qkv.weight",
                from_pattern=["visual.blocks.{block}.attn.qkv.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.attn.qkv.bias",
                from_pattern=["visual.blocks.{block}.attn.qkv.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.attn.proj.weight",
                from_pattern=["visual.blocks.{block}.attn.proj.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.attn.proj.bias",
                from_pattern=["visual.blocks.{block}.attn.proj.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.mlp.gate_proj.weight",
                from_pattern=["visual.blocks.{block}.mlp.gate_proj.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.mlp.gate_proj.bias",
                from_pattern=["visual.blocks.{block}.mlp.gate_proj.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.mlp.up_proj.weight",
                from_pattern=["visual.blocks.{block}.mlp.up_proj.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.mlp.up_proj.bias",
                from_pattern=["visual.blocks.{block}.mlp.up_proj.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.mlp.down_proj.weight",
                from_pattern=["visual.blocks.{block}.mlp.down_proj.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.mlp.down_proj.bias",
                from_pattern=["visual.blocks.{block}.mlp.down_proj.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.norm1.weight",
                from_pattern=["visual.blocks.{block}.norm1.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.blocks.{block}.norm2.weight",
                from_pattern=["visual.blocks.{block}.norm2.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.merger.ln_q.weight",
                from_pattern=["visual.merger.ln_q.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.merger.mlp_0.weight",
                from_pattern=["visual.merger.mlp.0.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.merger.mlp_0.bias",
                from_pattern=["visual.merger.mlp.0.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.merger.mlp_1.weight",
                from_pattern=["visual.merger.mlp.2.weight"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.visual.merger.mlp_1.bias",
                from_pattern=["visual.merger.mlp.2.bias"],
                required=False,
            ),
        ]


def _flatten_nested_params(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Nested MLX tree → flat keys matching TransformerBase._param_map (e.g. dit.transformer_blocks.0...)."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten_nested_params(v, p))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            p = f"{prefix}.{i}"
            flat.update(_flatten_nested_params(item, p))
    else:
        flat[prefix] = obj
    return flat


_QWEN_PRECONVERTED_DIT_PREFIXES = (
    "transformer_blocks.",
    "img_in.",
    "txt_in.",
    "time_text_embed.",
    "norm_out.",
    "proj_out.",
)


def _pass_through_preconverted_qwen_dit_keys(flat: dict[str, Any], weights: dict) -> None:
    """mlx-community bundles may already use DanQing ``dit.*`` flat keys (skip diffusers remap)."""
    for key, tensor in weights.items():
        if key.startswith("dit."):
            flat.setdefault(key, tensor)
            continue
        if any(key.startswith(prefix) for prefix in _QWEN_PRECONVERTED_DIT_PREFIXES):
            flat.setdefault(f"dit.{key}", tensor)


def remap_qwen_transformer_weights(weights: dict) -> dict:
    """Diffusers-style flat checkpoint keys → DanQing flat keys under ``dit.*`` (Pipeline + remap_fn)."""
    nested = WeightMapper.apply_mapping(weights, QwenWeightMapping.get_transformer_mapping())
    flat = _flatten_nested_params(nested, "dit")
    _pass_through_preconverted_qwen_dit_keys(flat, weights)
    return flat


_QWEN_LORA_AB = re.compile(
    r"^(.+)\.lora_[AB](?:\.default)?\.(?:weight|bias)$",
    re.IGNORECASE,
)
_QWEN_LORA_DOWNUP = re.compile(
    r"^(.+)\.lora_(?:down|up)(?:\.default)?\.(?:weight|bias)$",
    re.IGNORECASE,
)


def _qwen_lora_key_to_hf_linear_stem(key: str) -> str | None:
    """Strip PEFT / diffusers LoRA suffix; return HF-style path under ``transformer_blocks.*`` (no ``.weight``)."""
    k = (key or "").strip()
    m = _QWEN_LORA_AB.match(k)
    if m:
        return m.group(1).rstrip(".")
    m = _QWEN_LORA_DOWNUP.match(k)
    if m:
        return m.group(1).rstrip(".")
    return None


def remap_qwen_lora_module_prefix(hf_stem: str) -> str:
    """HF/diffusers LoRA module path → DanQing ``dit`` subtree prefix (no ``dit.``, no ``.weight`` / ``.bias``)."""
    m = (hf_stem or "").strip().strip(".")
    while ".." in m:
        m = m.replace("..", ".")
    for pref in (
        "transformer.",
        "qwen_image_transformer.",
        "transformer.transformer.",
        "diffusion_model.",
        "pipe.transformer.",
        "model.",
    ):
        if m.startswith(pref):
            m = m[len(pref) :]
    m = m.replace(".base_model.model.", ".")
    m = re.sub(r"\.img_mlp\.net\.0\.proj$", ".img_ff.mlp_in", m)
    m = re.sub(r"\.img_mlp\.net\.2$", ".img_ff.mlp_out", m)
    m = re.sub(r"\.txt_mlp\.net\.0\.proj$", ".txt_ff.mlp_in", m)
    m = re.sub(r"\.txt_mlp\.net\.2$", ".txt_ff.mlp_out", m)
    m = re.sub(r"\.img_mod\.1$", ".img_mod_linear", m)
    m = re.sub(r"\.txt_mod\.1$", ".txt_mod_linear", m)
    # Accept diffusers to_out.0, legacy Studio to_out and direct MLX attn_to_out.0.
    # The anonymous list index is part of the model's actual parameter path.
    m = re.sub(r"\.attn\.(?:to_out|attn_to_out)(?:\.0)?$", ".attn.attn_to_out.0", m)
    if m.endswith(".weight"):
        m = m[: -len(".weight")].rstrip(".")
    elif m.endswith(".bias"):
        m = m[: -len(".bias")].rstrip(".")
    return m.strip(".").strip()


def remap_qwen_lora_keys(lora_weights: dict) -> dict[str, tuple[Any, Any, float]]:
    """Group LoRA tensors; map module names to DanQing ``QwenImageTransformer._param_map`` prefixes (no ``dit.``)."""
    # Safetensors alpha is optional; use alpha=rank when omitted, not an
    # unrelated legacy rank-8 value. Studio writes an explicit alpha on export.
    default_alpha: float | None = None
    alphas_by_tgt: dict[str, float] = {}
    for key, tensor in lora_weights.items():
        lk = key.lower()
        if "alpha" not in lk:
            continue
        if any(x in lk for x in ("lora_down", "lora_up", "lora_a", "lora_b")):
            continue
        if not lk.endswith(".alpha"):
            continue
        base = key[: -len(".alpha")] if key.lower().endswith(".alpha") else key
        tgt = remap_qwen_lora_module_prefix(base)
        if not tgt:
            continue
        try:
            val = tensor.item() if hasattr(tensor, "item") else float(tensor)
            alphas_by_tgt[tgt] = float(val)
        except (TypeError, ValueError):
            pass

    groups: dict[str, dict[str, Any]] = {}
    for key, tensor in lora_weights.items():
        lk = key.lower()
        if lk.endswith(".alpha") and "lora_" not in lk:
            continue
        stem = _qwen_lora_key_to_hf_linear_stem(key)
        if stem is None:
            continue
        if stem not in groups:
            groups[stem] = {}
        if re.search(r"\.lora_a(?:\.default)?\.(?:weight|bias)$", lk) or re.search(
            r"\.lora_down(?:\.default)?\.(?:weight|bias)$", lk
        ):
            groups[stem]["down"] = tensor
        elif re.search(r"\.lora_b(?:\.default)?\.(?:weight|bias)$", lk) or re.search(
            r"\.lora_up(?:\.default)?\.(?:weight|bias)$", lk
        ):
            groups[stem]["up"] = tensor

    out: dict[str, tuple[Any, Any, float]] = {}
    for hf_stem, parts in groups.items():
        if "down" not in parts or "up" not in parts:
            raise RuntimeError(
                f"Qwen LoRA module {hf_stem!r} is missing a down/up factor; "
                "refusing to apply a partial adapter."
            )
        tgt = remap_qwen_lora_module_prefix(hf_stem)
        if not tgt:
            raise RuntimeError(f"Qwen LoRA module {hf_stem!r} has no valid target prefix.")
        fallback_alpha = default_alpha if default_alpha is not None else float(parts["down"].shape[0])
        alpha = float(alphas_by_tgt.get(tgt, fallback_alpha))
        out[tgt] = (parts["down"], parts["up"], alpha)
    return out


def apply_qwen_transformer_weights(flat_dict: dict) -> dict:
    """Nested parameter tree for ``dit.update()`` (tests / tooling); Pipeline uses ``remap_qwen_transformer_weights``."""
    return WeightMapper.apply_mapping(flat_dict, QwenWeightMapping.get_transformer_mapping())


def apply_qwen_text_encoder_weights(flat_dict: dict) -> dict:
    has_encoder_prefix = any(k.startswith("encoder.") for k in flat_dict)
    has_model_prefix = any(k.startswith("model.") for k in flat_dict)
    if has_encoder_prefix and not has_model_prefix:
        # mlx-community / other pre-remapped bundles already use encoder.* flat keys.
        nested: dict = {}
        for key, value in flat_dict.items():
            if key.startswith("encoder."):
                WeightMapper._set_nested_value(nested, key, value)
        return nested
    return WeightMapper.apply_mapping(flat_dict, QwenWeightMapping.get_text_encoder_mapping())


def apply_qwen_vae_weights(flat_dict: dict) -> dict:
    return WeightMapper.apply_mapping(flat_dict, QwenWeightMapping.get_vae_mapping())


QWEN_IMAGE_LORA_COMPAT_MODEL_IDS = frozenset(
    {
        "qwen-image",
        "qwen-image-edit",
        "firered-image-edit-1.1",
    }
)


def qwen_image_lora_scope_key(value: str) -> str:
    """T2I / Edit / FireRed share DiT LoRA key layout — one merge scope."""
    mid = value.split(":", 1)[0].strip() if value else ""
    if mid in QWEN_IMAGE_LORA_COMPAT_MODEL_IDS:
        return "qwen_image"
    return mid


def qwen_image_lora_base_compatible(model_id: str, lora_base: str) -> bool:
    model_key = (model_id or "").split(":", 1)[0].strip()
    lora_key = (lora_base or "").split(":", 1)[0].strip()
    if not lora_key or lora_key == model_key:
        return True
    return model_key in QWEN_IMAGE_LORA_COMPAT_MODEL_IDS and lora_key in QWEN_IMAGE_LORA_COMPAT_MODEL_IDS

