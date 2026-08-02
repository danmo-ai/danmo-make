"""``python -m tests.benchmark eval|download-judge``"""
from __future__ import annotations

import argparse
import sys

from tests.benchmark.eval_cases import iter_runnable_eval_cases, list_skipped_eval_cases
from tests.benchmark.registry_utils import resolve_benchmark_data_root
from tests.benchmark.runner import list_eval_case_ids, run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Danmo Make image eval (L1 integrity + L2 PickScore)")
    sub = parser.add_subparsers(dest="suite", required=True)

    dl = sub.add_parser("download-judge", help="Download PickScore judge via ModelScope (魔塔)")
    dl.add_argument("--force", action="store_true", help="Re-download even if bundle exists")

    p = sub.add_parser("eval", help="Image model eval suite")
    p.add_argument("--all", action="store_true", help="Run all runnable cases")
    p.add_argument("--case", metavar="ID", default="", help="Single case id")
    p.add_argument("--list", action="store_true", help="List case ids")
    p.add_argument(
        "--list-runnable",
        action="store_true",
        help="List runnable case ids (bundle installed)",
    )
    p.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="full",
        help="smoke=P1+E2 per model; full=all prompts",
    )
    p.add_argument("--calibrate", action="store_true", help="Write PickScore baselines to golden/")
    p.add_argument("--output-dir", default="tests/benchmark/outputs")
    p.add_argument(
        "--keep-judge-loaded",
        action="store_true",
        help="Keep PickScore loaded across cases (faster eval, higher parent RAM)",
    )

    args = parser.parse_args(argv)

    if args.suite == "download-judge":
        from tests.benchmark.judge_assets import download_pickscore_modelscope

        try:
            path = download_pickscore_modelscope(force=bool(args.force))
        except Exception as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print(path)
        return 0

    if args.suite != "eval":
        parser.error(f"Unknown suite: {args.suite}")

    profile = args.profile

    if args.list_runnable:
        print(resolve_benchmark_data_root())
        print("\n".join(c.id for c in iter_runnable_eval_cases(profile=profile)))
        skipped = list_skipped_eval_cases(profile=profile)
        if skipped:
            print("\n# skipped models (bundle not ready)", file=sys.stderr)
            for mid, reason in skipped:
                print(f"{mid}\t{reason}", file=sys.stderr)
        return 0

    if args.list:
        print("\n".join(list_eval_case_ids(profile=profile)))
        return 0

    if args.all and args.case:
        parser.error("use either --all or --case, not both")
    if not args.all and not args.case:
        parser.error("specify --all or --case <id>")

    return run_eval(
        "" if args.all else args.case,
        output_dir=args.output_dir,
        profile=profile,
        calibrate=args.calibrate,
        release_judge_each_case=not bool(args.keep_judge_loaded),
    )


if __name__ == "__main__":
    raise SystemExit(main())
