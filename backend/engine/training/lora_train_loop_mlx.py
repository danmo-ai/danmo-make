"""Shared MLX DiT LoRA training loop (grad accum, resume, val loss, compile)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from backend.core.contracts import ExecutionContext
from backend.engine.training.training_log import training_log, training_progress
from backend.engine.training.lora_layers_mlx import add_grad_trees, scale_grad_tree
from backend.engine.training.lora_train_runtime_mlx import (
    LoraTrainRuntimeConfig,
    active_memory_gb,
    build_optimizer,
    clear_mlx_training_cache,
    configure_mlx_training_memory,
    load_training_checkpoint,
    peak_memory_gb,
    reset_peak_memory,
    save_training_checkpoint,
)


def run_dit_lora_train_loop(
    *,
    exec_ctx: ExecutionContext,
    model: Any,
    train_module: nn.Module,
    runtime: LoraTrainRuntimeConfig,
    work_dir: Path,
    adapter_dir: Path,
    base_model_id: str,
    n_samples: int,
    sample_batch: Callable[[list[int]], tuple[Any, ...]],
    train_indices: list[int],
    val_indices: list[int],
    loss_fn: Callable[..., mx.array],
    on_progress_preview: Callable[[int], None] | None = None,
    mlx_ctx: Any | None = None,
) -> tuple[list[dict[str, float]], Path | None]:
    configure_mlx_training_memory(mlx_ctx=mlx_ctx)
    clear_mlx_training_cache(mlx_ctx)
    reset_peak_memory()

    warmup = optim.linear_schedule(0, runtime.learning_rate, runtime.warmup_steps)
    cosine = optim.cosine_decay(
        runtime.learning_rate,
        max(1, runtime.iterations // runtime.grad_accumulate),
    )
    lr_schedule = optim.join_schedules([warmup, cosine], [runtime.warmup_steps])
    optimizer = build_optimizer(
        train_module,
        name=runtime.optimizer_name,
        learning_rate=lr_schedule,
        weight_decay=runtime.weight_decay,
    )

    start_iter = 0
    if runtime.resume_from:
        resume_path = Path(runtime.resume_from)
        start_iter = load_training_checkpoint(
            resume_path,
            train_module,
            optimizer,
            rank=runtime.lora_rank,
        )
        training_log(exec_ctx, "info", f"Resumed from {resume_path} at iteration {start_iter}")

    loss_history: list[dict[str, float]] = []
    loss_path = work_dir / "loss_history.json"
    if loss_path.is_file() and start_iter > 0:
        try:
            loss_history = json.loads(loss_path.read_text(encoding="utf-8"))
        except Exception:
            loss_history = []
    else:
        loss_path.write_text("[]", encoding="utf-8")

    loss_and_grad = nn.value_and_grad(train_module, loss_fn)

    state = [train_module.state, optimizer.state, mx.random.state]

    def _raw_step(batch_args: tuple[Any, ...], prev_grad: Any | None, do_update: bool) -> tuple[Any, Any | None]:
        loss, grads = loss_and_grad(*batch_args)
        if prev_grad is None:
            accum = grads
        else:
            accum = add_grad_trees(prev_grad, grads)
        if do_update:
            scaled = scale_grad_tree(accum, runtime.grad_accumulate)
            optimizer.update(train_module, scaled)
            accum = None
        return loss, accum

    step_fn: Callable[..., tuple[Any, Any | None]] = _raw_step
    if runtime.compile_step:
        try:
            step_fn = partial(mx.compile, inputs=state, outputs=state)(_raw_step)
        except Exception as e:
            training_log(exec_ctx, "warning", f"mx.compile disabled for training step: {e}")

    training_log(
        exec_ctx,
        "info",
        f"Training {runtime.iterations} iterations "
        f"(rank={runtime.lora_rank}, type={runtime.train_type}, qlora={runtime.qlora_bits}, "
        f"grad_ckpt={runtime.grad_checkpoint}, min_snr={runtime.min_snr_gamma}, "
        f"prior_w={runtime.prior_loss_weight}, opt={runtime.optimizer_name}) …",
    )

    accum_grads: Any | None = None
    losses: list[float] = []
    tic = time.time()
    log_every = max(1, min(10, runtime.iterations // 30 or 1))
    best_val: float | None = None
    best_path: Path | None = None
    stale_val = 0

    for i in range(start_iter, runtime.iterations):
        exec_ctx.cancel_token.raise_if_cancelled()
        idx = train_indices[int(mx.random.randint(0, len(train_indices), (1,)).item())]
        batch_args = sample_batch([idx])
        do_update = (i + 1) % runtime.grad_accumulate == 0
        loss, accum_grads = step_fn(batch_args, accum_grads, do_update)
        if do_update:
            mx.eval(train_module.parameters(), optimizer.state)
        mx.eval(loss)
        loss_val = float(loss.item())
        if not __import__("math").isfinite(loss_val):
            raise RuntimeError(
                f"LoRA training loss is non-finite at iteration {i + 1}: {loss_val!r}. "
                "Check learning_rate, prior_loss_weight/class_prompt, and dataset captions."
            )
        losses.append(loss_val)

        if (i + 1) == start_iter + 1 or (i + 1) % log_every == 0:
            avg = sum(losses) / len(losses)
            peak = peak_memory_gb()
            active = active_memory_gb()
            training_log(
                exec_ctx,
                "info",
                f"Iter {i + 1}/{runtime.iterations} loss={avg:.4f} "
                f"active_mem={active:.1f}GB peak_mem={peak:.1f}GB "
                f"it/s={len(losses) / max(time.time() - tic, 1e-6):.2f}",
            )
            row: dict[str, float] = {"step": float(i + 1), "loss": avg}
            if val_indices and runtime.val_every > 0 and (i + 1) % runtime.val_every == 0:
                val_losses: list[float] = []
                for vidx in val_indices:
                    vbatch = sample_batch([vidx])
                    vloss = loss_fn(*vbatch)
                    mx.eval(vloss)
                    val_losses.append(float(vloss.item()))
                    clear_mlx_training_cache(mlx_ctx)
                if val_losses:
                    val_avg = sum(val_losses) / len(val_losses)
                    row["val_loss"] = val_avg
                    training_log(exec_ctx, "info", f"Val loss={val_avg:.4f} ({len(val_indices)} samples)")
                    if best_val is None or val_avg < best_val:
                        best_val = val_avg
                        stale_val = 0
                        best_path = adapter_dir / "best_adapters.safetensors"
                        meta = {
                            "iteration": i + 1,
                            "lora_rank": runtime.lora_rank,
                            "base_model": base_model_id,
                            "val_loss": val_avg,
                            "train_type": runtime.train_type,
                        }
                        save_training_checkpoint(
                            best_path,
                            train_module,
                            optimizer,
                            rank=runtime.lora_rank,
                            meta=meta,
                        )
                        training_log(exec_ctx, "info", f"New best val loss; saved {best_path.name}")
                    else:
                        stale_val += 1
            loss_history.append(row)
            loss_path.write_text(json.dumps(loss_history), encoding="utf-8")
            losses = []
            tic = time.time()

            if (
                runtime.early_stop_patience > 0
                and val_indices
                and runtime.val_every > 0
                and stale_val >= runtime.early_stop_patience
                and best_val is not None
            ):
                training_log(
                    exec_ctx,
                    "info",
                    f"Early stop: val loss did not improve for {stale_val} eval cycles",
                )
                break

        training_progress(exec_ctx, step=i + 1, total=runtime.iterations, loss=loss_val)

        if on_progress_preview and (i + 1) % runtime.progress_every == 0:
            try:
                on_progress_preview(i + 1)
            except Exception as e:
                training_log(exec_ctx, "warning", f"Progress preview failed: {e}")

        if (i + 1) % runtime.checkpoint_every == 0:
            ckpt = adapter_dir / f"{i + 1:07d}_adapters.safetensors"
            meta = {
                "iteration": i + 1,
                "lora_rank": runtime.lora_rank,
                "base_model": base_model_id,
                "qlora_bits": runtime.qlora_bits,
                "train_type": runtime.train_type,
            }
            save_training_checkpoint(
                ckpt,
                train_module,
                optimizer,
                rank=runtime.lora_rank,
                meta=meta,
            )

    return loss_history, best_path
