"""Shared MLX DiT LoRA training loop (grad accum, resume, val loss, compile)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import contextmanager
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


_VAL_SEED_BASE = 0x5EED_0A11


@contextmanager
def preserve_training_random_streams(*, compiled_state: list[Any] | None = None):
    """Keep validation / previews from advancing either training RNG stream.

    ``mx.compile(inputs=state)`` retains the original ``_RandomState`` object;
    assigning a restored snapshot alone therefore leaves compiled training on a
    different stream. Update that retained slot to the restored object as well.
    """
    import random

    from backend.engine.training.lora_train_runtime_mlx import _snapshot_mlx_rng_state

    mlx_rng_state = _snapshot_mlx_rng_state()
    python_rng_state = random.getstate()
    try:
        yield
    finally:
        mx.random.state = mlx_rng_state
        if compiled_state is not None:
            if len(compiled_state) != 3:
                raise RuntimeError("Compiled training state must be [model, optimizer, MLX RNG].")
            compiled_state[2] = mx.random.state
        random.setstate(python_rng_state)


def _deterministic_val_loss(
    loss_fn: Callable[..., mx.array],
    sample_batch: Callable[[list[int]], tuple[Any, ...]],
    val_indices: list[int],
    *,
    mlx_ctx: Any | None,
    compiled_state: list[Any] | None = None,
) -> float:
    """Validation loss with fixed σ / ε per sample.

    ``loss_fn`` draws the noise level and noise from the global RNG; with a handful of val
    samples that variance dwarfs the training improvement, so picking ``best_adapters`` on it
    is a lottery (often an early, untrained checkpoint). Seed each sample and save/restore
    both MLX and Python RNG streams, including assistant toggles and caption dropout.
    """
    import random

    if compiled_state is not None and len(compiled_state) != 3:
        raise RuntimeError("Compiled training state must be [model, optimizer, MLX RNG].")
    from backend.engine.training.dit_training_loss_mlx import caption_dropout_disabled

    dropout_token = caption_dropout_disabled.set(True)
    try:
        losses: list[float] = []
        with preserve_training_random_streams(compiled_state=compiled_state):
            for k, vidx in enumerate(val_indices):
                val_seed = _VAL_SEED_BASE + 7919 * k + int(vidx)
                mx.random.seed(val_seed)
                random.seed(val_seed)
                vbatch = sample_batch([vidx])
                vloss = loss_fn(*vbatch)
                mx.eval(vloss)
                losses.append(float(vloss.item()))
                clear_mlx_training_cache(mlx_ctx)
    finally:
        caption_dropout_disabled.reset(dropout_token)
    return sum(losses) / max(1, len(losses))


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
    import random

    random.seed(runtime.seed)
    mx.random.seed(runtime.seed)
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

    best_val: float | None = None
    best_path = adapter_dir / "best_adapters.safetensors"
    best_meta_path = adapter_dir / "best_adapters.json"
    if start_iter > 0 and best_meta_path.is_file():
        try:
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            best_val = float(best_meta["val_loss"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read resume best-checkpoint metadata {best_meta_path}: {exc}") from exc
        if not best_path.is_file():
            raise RuntimeError(f"Best-checkpoint metadata exists but weights are missing: {best_path}")
    else:
        best_path = None

    loss_history: list[dict[str, float]] = []
    loss_path = work_dir / "loss_history.json"
    if loss_path.is_file() and start_iter > 0:
        try:
            loaded_history = json.loads(loss_path.read_text(encoding="utf-8"))
            loss_history = [row for row in loaded_history if int(row.get("step") or 0) <= start_iter]
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Cannot read resume loss history {loss_path}: {exc}") from exc
    loss_path.write_text(json.dumps(loss_history), encoding="utf-8")

    loss_and_grad = nn.value_and_grad(train_module, loss_fn)

    state = [train_module.state, optimizer.state, mx.random.state]

    def _raw_step(
        batch_args: tuple[Any, ...],
        prev_grad: Any | None,
        do_update: bool,
        accumulation_count: int,
    ) -> tuple[Any, Any | None]:
        loss, grads = loss_and_grad(*batch_args)
        if prev_grad is None:
            accum = grads
        else:
            accum = add_grad_trees(prev_grad, grads)
        if do_update:
            scaled = scale_grad_tree(accum, max(1, int(accumulation_count)))
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
        f"caption_dropout={runtime.caption_dropout:g}, "
        f"prior_w={runtime.prior_loss_weight}, opt={runtime.optimizer_name}) …",
    )

    accum_grads: Any | None = None
    losses: list[float] = []
    tic = time.time()
    log_every = max(1, min(10, runtime.iterations // 30 or 1))
    stale_val = 0
    stop_requested = False
    completed_iter = start_iter
    last_update_iter = start_iter

    for i in range(start_iter, runtime.iterations):
        exec_ctx.cancel_token.raise_if_cancelled()
        idx = train_indices[int(mx.random.randint(0, len(train_indices), (1,)).item())]
        batch_args = sample_batch([idx])
        accumulation_count = (i % runtime.grad_accumulate) + 1
        do_update = (i + 1) % runtime.grad_accumulate == 0 or i + 1 == runtime.iterations
        loss, accum_grads = step_fn(batch_args, accum_grads, do_update, accumulation_count)
        if do_update:
            mx.eval(train_module.parameters(), optimizer.state)
            last_update_iter = i + 1
        mx.eval(loss)
        completed_iter = i + 1
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
                val_avg = _deterministic_val_loss(
                    loss_fn, sample_batch, val_indices,
                    mlx_ctx=mlx_ctx, compiled_state=state,
                )
                row["val_loss"] = val_avg
                training_log(
                    exec_ctx,
                    "info",
                    f"Val loss={val_avg:.4f} ({len(val_indices)} samples, fixed σ/ε)",
                )
                # A mid-accumulation validation has no new optimizer step to resume.
                # Do not publish a misleading checkpoint for this micro-step.
                if do_update and (best_val is None or val_avg < best_val):
                    best_val = val_avg
                    stale_val = 0
                    best_path = adapter_dir / "best_adapters.safetensors"
                    meta = {
                        "iteration": i + 1,
                        "seed": runtime.seed,
                        "lora_rank": runtime.lora_rank,
                        "lora_alpha": runtime.lora_alpha,
                        "lora_blocks": runtime.lora_blocks,
                        "base_model": base_model_id,
                        "val_loss": val_avg,
                        "train_type": runtime.train_type,
                        "lora_module_keys": runtime.lora_module_keys,
                    }
                    save_training_checkpoint(
                        best_path,
                        train_module,
                        optimizer,
                        rank=runtime.lora_rank,
                        meta=meta,
                    )
                    training_log(exec_ctx, "info", f"New best val loss; saved {best_path.name}")
                    (work_dir / "loss_history.json").write_text(json.dumps(loss_history + [row]), encoding="utf-8")
                else:
                    stale_val += 1
            loss_history.append(row)
            loss_path.write_text(json.dumps(loss_history), encoding="utf-8")
            losses = []
            tic = time.time()

            if (
                do_update
                and runtime.early_stop_patience > 0
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
                stop_requested = True
                break

        training_progress(exec_ctx, step=i + 1, total=runtime.iterations, loss=loss_val)

        if on_progress_preview and (i + 1) % runtime.progress_every == 0:
            try:
                with preserve_training_random_streams(compiled_state=state):
                    mx.random.seed(runtime.seed)
                    random.seed(runtime.seed)
                    on_progress_preview(i + 1)
            except Exception as e:
                training_log(exec_ctx, "warning", f"Progress preview failed: {e}")

        if do_update and (i + 1) % runtime.checkpoint_every == 0:
            from backend.engine.training.lora_layers_mlx import assert_lora_adapter_received_gradients

            assert_lora_adapter_received_gradients(train_module, step=i + 1)
            ckpt = adapter_dir / f"{i + 1:07d}_adapters.safetensors"
            meta = {
                "iteration": i + 1,
                "seed": runtime.seed,
                "lora_rank": runtime.lora_rank,
                "lora_alpha": runtime.lora_alpha,
                "lora_blocks": runtime.lora_blocks,
                "base_model": base_model_id,
                "qlora_bits": runtime.qlora_bits,
                "train_type": runtime.train_type,
                "lora_module_keys": runtime.lora_module_keys,
            }
            save_training_checkpoint(
                ckpt,
                train_module,
                optimizer,
                rank=runtime.lora_rank,
                meta=meta,
            )

    final_path = adapter_dir / "final_adapters.safetensors"
    save_training_checkpoint(
        final_path,
        train_module,
        optimizer,
        rank=runtime.lora_rank,
        meta={
            "iteration": last_update_iter,
            "seed": runtime.seed,
            "lora_rank": runtime.lora_rank,
            "lora_alpha": runtime.lora_alpha,
            "lora_blocks": runtime.lora_blocks,
            "base_model": base_model_id,
            "train_type": runtime.train_type,
            "lora_module_keys": runtime.lora_module_keys,
            "early_stopped": stop_requested,
        },
    )
    return loss_history, best_path
