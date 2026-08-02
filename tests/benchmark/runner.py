"""Image eval runner: L1 integrity + L2 PickScore."""
from __future__ import annotations

import json
import gc
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .eval_cases import (
    EVAL_UPSCALE_SCALE,
    EvalCase,
    ensure_edit_source,
    ensure_edit_mask,
    ensure_upscale_source,
    expand_eval_cases,
    get_eval_case,
    golden_reward,
    iter_runnable_eval_cases,
    list_skipped_eval_cases,
    load_golden_scores,
    save_golden_scores,
)
from .integrity import IntegrityResult, check_output_image_integrity
from .judge import JUDGE_MODEL_ID, JudgeResult, judge_image, reset_judge_cache
from .registry_utils import bundle_ready, repo_root, resolve_benchmark_data_root

Profile = Literal["smoke", "full"]

PROJECT_ROOT = repo_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "benchmark" / "outputs"
_DANQING_PY = PROJECT_ROOT / ".venv" / "bin" / "python3"
_DEFAULT_BENCH_MLX_MEMORY_GB = 64


def _bench_teacache_mode() -> str | None:
    raw = os.environ.get("DANQING_BENCH_TEACACHE_MODE", "auto").strip().lower()
    if raw in ("", "none", "off", "false", "0"):
        return None
    return raw


def _append_bench_inference_flags(cmd: list[str]) -> list[str]:
    tc = _bench_teacache_mode()
    if tc:
        cmd += ["--teacache-mode", tc]
    return cmd


def _bench_mlx_memory_gb() -> int:
    raw = os.environ.get("DANQING_BENCH_MLX_MEMORY_GB", str(_DEFAULT_BENCH_MLX_MEMORY_GB)).strip()
    try:
        return max(16, min(int(raw), 120))
    except ValueError:
        return _DEFAULT_BENCH_MLX_MEMORY_GB


def _danqing_python() -> str:
    if _DANQING_PY.is_file():
        return str(_DANQING_PY)
    return sys.executable


@dataclass
class EvalResult:
    case_id: str
    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    l1: IntegrityResult | None = None
    l2: JudgeResult | None = None
    gen_sec: float | None = None
    judge_sec: float | None = None
    output_path: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "ok": self.ok,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "reason": self.reason,
            "gen_sec": self.gen_sec,
            "judge_sec": self.judge_sec,
            "output_path": self.output_path,
        }
        if self.l1 is not None:
            payload["l1"] = asdict(self.l1)
        if self.l2 is not None:
            payload["l2"] = {
                "ok": self.l2.ok,
                "score": self.l2.score,
                "min_required": self.l2.min_required,
                "reason": self.l2.reason,
                "model_id": self.l2.model_id,
            }
        return payload


@dataclass
class EvalRunReport:
    schema_version: int = 1
    profile: str = "full"
    judge_model: str = JUDGE_MODEL_ID
    started_at: str = ""
    finished_at: str = ""
    summary: dict[str, int] = field(default_factory=dict)
    cases: list[dict[str, Any]] = field(default_factory=list)


class EvalRunner:
    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        *,
        release_judge_each_case: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[tuple[EvalCase, EvalResult]] = []
        self.started_at = datetime.now(timezone.utc)
        self.release_judge_each_case = release_judge_each_case

    def _output_path(self, case: EvalCase) -> Path:
        safe = case.id.replace(":", "__")
        return self.output_dir / f"{safe}.png"

    def _subprocess_env(self, case: EvalCase | None = None) -> dict[str, str]:
        env = os.environ.copy()
        mlx_gb = _bench_mlx_memory_gb()
        env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
        env["MLX_METAL_MEMORY_LIMIT"] = str(mlx_gb)
        env["DANQING_MLX_MEMORY_LIMIT_GB"] = str(mlx_gb)
        env["DANQING_BENCH_EVAL"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        _ = case
        return env

    def _run_generate(self, case: EvalCase, output_path: Path) -> bool:
        if case.action == "create":
            return self._run_generate_create(case, output_path)
        if case.action == "upscale":
            return self._run_generate_upscale(case, output_path)
        if case.action in {"rewrite", "retouch", "extend"}:
            return self._run_generate_edit(case, output_path)
        print(f"    [gen] unknown action: {case.action}")
        return False

    def _run_generate_create(self, case: EvalCase, output_path: Path) -> bool:
        cli = PROJECT_ROOT / "bin" / "danqing-generate"
        cmd = [
            _danqing_python(),
            str(cli),
            "--model",
            case.model_field,
            "--prompt",
            case.encoded_prompt,
            "--seed",
            str(case.seed),
            "--steps",
            str(case.steps),
            "--guidance",
            str(case.guidance),
            "--size",
            f"{case.width}x{case.height}",
            "--output",
            str(output_path),
        ]
        cmd = _append_bench_inference_flags(cmd)
        return self._exec_cli(cmd, label="generate", timeout_sec=case.timeout_sec, case=case)

    def _run_generate_edit(self, case: EvalCase, output_path: Path) -> bool:
        ensure_edit_source()
        cli = PROJECT_ROOT / "bin" / "danqing-edit"
        cmd = [
            _danqing_python(),
            str(cli),
            "--model",
            case.model_field,
            "--operation",
            case.action,
            "--source-image",
            str(ensure_edit_source()),
            "--prompt",
            case.encoded_prompt,
            "--seed",
            str(case.seed),
            "--guidance",
            str(case.guidance),
            "--output",
            str(output_path),
        ]
        if case.steps > 0:
            cmd += ["--steps", str(case.steps)]
        if not case.omit_image_strength:
            cmd += ["--source-fidelity", str(case.image_strength)]
        if case.action == "extend":
            cmd += ["--extend-directions", "right"]
        if case.action == "retouch":
            cmd += ["--mask-image", str(ensure_edit_mask())]
        cmd = _append_bench_inference_flags(cmd)
        return self._exec_cli(cmd, label="edit", timeout_sec=case.timeout_sec, case=case)

    def _run_generate_upscale(self, case: EvalCase, output_path: Path) -> bool:
        ensure_upscale_source()
        cli = PROJECT_ROOT / "bin" / "danqing-upscale"
        scale = int(case.upscale_scale or EVAL_UPSCALE_SCALE)
        if scale not in (2, 4):
            scale = 2
        cmd = [
            _danqing_python(),
            str(cli),
            "--model",
            case.model_field,
            "--source-image",
            str(ensure_upscale_source()),
            "--scale-factor",
            str(scale),
            "--seed",
            str(case.seed),
            "--output",
            str(output_path),
        ]
        return self._exec_cli(cmd, label="upscale", timeout_sec=case.timeout_sec, case=case)

    def _exec_cli(
        self,
        cmd: list[str],
        *,
        label: str,
        timeout_sec: int,
        case: EvalCase | None = None,
    ) -> bool:
        try:
            t0 = time.time()
            proc = subprocess.run(
                cmd,
                timeout=int(timeout_sec),
                env=self._subprocess_env(case),
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                print(f"    [{label}] fail exit={proc.returncode} ({elapsed:.1f}s)")
                return False
            print(f"    [{label}] ok ({elapsed:.1f}s) -> {cmd[-1]}")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [{label}] timeout ({timeout_sec}s)")
            return False
        except FileNotFoundError:
            print(f"    [{label}] cli missing")
            return False
        finally:
            gc.collect()

    def run_one(
        self,
        case: EvalCase,
        *,
        calibrate: bool = False,
        skip_judge: bool = False,
    ) -> EvalResult:
        if not case.model_id:
            res = EvalResult(case.id, ok=False, reason="invalid_case")
            self.results.append((case, res))
            return res

        out_path = self._output_path(case)
        vk = f":{case.version_key}" if case.version_key else ""
        print(f"[{case.id}] model={case.model_id}{vk} action={case.action} {case.width}x{case.height}")

        t0 = time.time()
        gen_ok = self._run_generate(case, out_path)
        gen_sec = time.time() - t0
        if not gen_ok:
            res = EvalResult(
                case.id,
                ok=False,
                gen_sec=gen_sec,
                output_path=str(out_path),
                reason="generation_failed",
            )
            self.results.append((case, res))
            print(f"  FAIL L0: generation_failed ({gen_sec:.1f}s)")
            return res

        l1 = check_output_image_integrity(
            out_path,
            expected_width=case.l1_expected_width,
            expected_height=case.l1_expected_height,
        )
        if not l1.ok:
            res = EvalResult(
                case.id,
                ok=False,
                l1=l1,
                gen_sec=gen_sec,
                output_path=str(out_path),
                reason=l1.reason,
            )
            self.results.append((case, res))
            print(f"  FAIL L1: {l1.reason} ({gen_sec:.1f}s)")
            return res
        print(f"  L1 PASS: {l1.width}x{l1.height} {l1.bytes // 1024}KB ({gen_sec:.1f}s)")

        if skip_judge:
            res = EvalResult(case.id, ok=True, l1=l1, gen_sec=gen_sec, output_path=str(out_path))
            self.results.append((case, res))
            return res

        judge_prompt = case.judge_prompt.strip()
        if not judge_prompt:
            res = EvalResult(
                case.id,
                ok=False,
                l1=l1,
                gen_sec=gen_sec,
                output_path=str(out_path),
                reason="missing_judge_prompt",
            )
            self.results.append((case, res))
            print("  FAIL L2: missing_judge_prompt")
            return res

        golden = golden_reward(case.id) if not calibrate else None
        t1 = time.time()
        try:
            try:
                l2 = judge_image(
                    judge_prompt,
                    out_path,
                    golden=golden,
                    judge_floor=case.judge_floor,
                )
            except Exception as exc:
                res = EvalResult(
                    case.id,
                    ok=False,
                    l1=l1,
                    gen_sec=gen_sec,
                    judge_sec=time.time() - t1,
                    output_path=str(out_path),
                    reason=str(exc),
                )
                self.results.append((case, res))
                print(f"  FAIL L2: {exc}")
                return res
            judge_sec = time.time() - t1

            if calibrate:
                data = load_golden_scores()
                cases = data.setdefault("cases", {})
                cases[case.id] = {
                    "reward": l2.score,
                    "prompt_id": case.prompt_id,
                    "model_id": case.model_id,
                    "action": case.action,
                    "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
                save_golden_scores(data)

            ok = l2.ok if not calibrate else True
            res = EvalResult(
                case.id,
                ok=ok,
                l1=l1,
                l2=l2,
                gen_sec=gen_sec,
                judge_sec=judge_sec,
                output_path=str(out_path),
                reason="" if ok else l2.reason,
            )
            self.results.append((case, res))
            tag = "PASS" if ok else "FAIL"
            golden_note = f" golden={golden:.3f}" if golden is not None else ""
            print(
                f"  {tag} L2: PickScore={l2.score:.3f} min={l2.min_required:.3f}{golden_note} "
                f"({judge_sec:.1f}s judge)"
            )
            return res
        finally:
            if self.release_judge_each_case:
                reset_judge_cache()

    def run_all(self, *, profile: Profile = "full", calibrate: bool = False) -> None:
        skipped_models = list_skipped_eval_cases(profile=profile)
        cases = iter_runnable_eval_cases(profile=profile)
        print(f"\n{'=' * 60}")
        print(f"Danmo Make Image Eval — profile={profile} runnable={len(cases)}")
        print(f"Data root: {resolve_benchmark_data_root()}")
        print(f"Judge: {JUDGE_MODEL_ID}")
        print(f"Gen subprocess MLX limit: {_bench_mlx_memory_gb()} GB")
        if self.release_judge_each_case:
            print("Judge: release after each case")
        if skipped_models:
            print(f"Skipped models (bundle not ready): {len(skipped_models)}")
        print(f"{'=' * 60}\n")
        for model_id, reason in skipped_models:
            print(f"  [SKIP model] {model_id}: {reason}")

        reset_judge_cache()
        for case in cases:
            self.run_one(case, calibrate=calibrate)
        self._print_summary(calibrate=calibrate)
        self._write_report(profile=profile)

    def _print_summary(self, *, calibrate: bool) -> None:
        total = len(self.results)
        passed = sum(1 for _, r in self.results if r.ok and not r.skipped)
        failed = sum(1 for _, r in self.results if not r.ok and not r.skipped)
        print(f"\n{'=' * 60}")
        if calibrate:
            print(f"Calibrate summary: {total} cases written/updated in golden/eval_scores.json")
        else:
            print(f"Eval summary: {total} cases — {passed} PASS / {failed} FAIL")
        print(f"{'=' * 60}")

    def _write_report(self, *, profile: Profile) -> None:
        report = EvalRunReport(
            profile=profile,
            started_at=self.started_at.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            summary={
                "pass": sum(1 for _, r in self.results if r.ok and not r.skipped),
                "fail": sum(1 for _, r in self.results if not r.ok and not r.skipped),
                "total": len(self.results),
            },
            cases=[r.as_dict() for _, r in self.results],
        )
        path = self.output_dir / "eval_report.json"
        path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_eval(
    case_id: str = "",
    *,
    output_dir: str = "tests/benchmark/outputs",
    profile: Profile = "full",
    calibrate: bool = False,
    release_judge_each_case: bool = True,
) -> int:
    runner = EvalRunner(
        output_dir=output_dir,
        release_judge_each_case=release_judge_each_case,
    )
    if case_id:
        case = get_eval_case(case_id, profile=profile)
        if case is None:
            print(f"Unknown eval case: {case_id}")
            print("Available:", "\n  ".join(list_eval_case_ids(profile=profile)[:20]), "...")
            return 2
        ready, reason = bundle_ready(case.model_id)
        if not ready:
            print(f"[SKIP] {case.id}: {reason}")
            return 0
        reset_judge_cache()
        result = runner.run_one(case, calibrate=calibrate)
        runner._write_report(profile=profile)
        return 0 if (result.ok or calibrate) else 1

    runner.run_all(profile=profile, calibrate=calibrate)
    if calibrate:
        return 0
    failed = sum(1 for _, r in runner.results if not r.ok)
    return 1 if failed else 0


def list_eval_case_ids(profile: Profile = "full") -> list[str]:
    return [c.id for c in expand_eval_cases(profile=profile)]
