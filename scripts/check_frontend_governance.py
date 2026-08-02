#!/usr/bin/env python3
"""Unified frontend governance: EP boundary, theme legacy, dq-ui selector compat."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"
STUDIO_STYLES = FRONTEND_SRC / "styles"
DQ_UI_ROOT = (ROOT / ".." / "dq-ui" / "packages").resolve()

SCAN_SUFFIXES = {".vue", ".ts", ".css"}

EP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"""from\s+['"]element-plus['"]"""), "do not import element-plus"),
    (re.compile(r"""@element-plus/icons-vue"""), "use Lucide / registerDqIcons"),
    (re.compile(r"\bElMessageBox\b"), "use confirm from @/utils/feedback"),
    (re.compile(r"\bElMessage\b"), "use toast from @/utils/feedback"),
    (re.compile(r"\bElNotification\b"), "use toast.notify from @/utils/feedback"),
    (re.compile(r"\bv-loading\b"), "use v-dq-loading"),
    (re.compile(r"<el-[a-z]"), "use Dq* components in templates"),
    (re.compile(r"\belement-plus\b", re.I), "remove element-plus reference"),
    (re.compile(r"Element\s+Plus"), "remove Element Plus reference"),
    (re.compile(r"\bstudio-ep[-_]"), "rename studio-ep-* layout classes"),
    (re.compile(r"\bsettings-ep[-_]"), "rename settings-ep-* layout classes"),
    (re.compile(r"\bgallery-ep[-_]"), "rename gallery-ep-* layout classes"),
]

THEME_FORBIDDEN_CLASS = re.compile(
    r"""class\s*=\s*['"][^'"]*\bel-(?:button|input|select|menu|tabs|table|form|card|dialog|drawer|row|col)\b"""
)
THEME_EP_PREFIX = re.compile(r"\b(?:studio-ep|settings-ep|gallery-ep)[-_]")
THEME_STYLE_EL = re.compile(r"\.el-[a-z0-9_-]+")
THEME_EL_TOKEN = re.compile(r"--el-")
THEME_BROKEN_MODEL_CARD = re.compile(r"\.modsurface\s+card")

UI_STYLE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.dq-select-dropdown\b"), "use `.dq-select__content` / `.dq-select__option`"),
    (re.compile(r"\.dq-dropdown-menu__item\b"), "use `.dq-dropdown-item`"),
    (re.compile(r"\.dq-input-number__increase\b"), "use `.dq-input-number__btn`"),
    (re.compile(r"\.dq-input-number__decrease\b"), "use `.dq-input-number__btn`"),
    (re.compile(r"\.dq-tag[^{]*\.is-plain\b"), "use `.dq-tag--plain` (DqTag effect=plain)"),
    # Typography — 4 steps only; no sub-caption px / product font stacks
    (re.compile(r"font-size:\s*(?:9|10|11)px\b"), "use var(--dq-font-size-caption|body|title)"),
    (re.compile(r"font-size:\s*0\.(?:6|7)\d*rem\b"), "sub-caption rem sizes forbidden; use --dq-font-size-*"),
    (re.compile(r"font-family:\s*ui-monospace\b"), "use var(--dq-font-mono)"),
    (re.compile(r"var\(--dq-font-size-[^,)]+,\s*\d+px\)"), "remove px fallbacks on --dq-font-size-*"),
]
UI_VUE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\bv-if="motion"\b'), 'use `v-if="divided"` on DqDropdownItem'),
    (re.compile(r"\bv-if='motion'\b"), "use `v-if='divided'` on DqDropdownItem"),
    (re.compile(r'\bsize="small"\b'), 'use `size="sm"` (not `small`)'),
    (re.compile(r"\bsize='small'\b"), "use `size='sm'` (not `small`)"),
    (re.compile(r'\bsize="mini"\b'), 'use `size="sm"` (not `mini`)'),
    (re.compile(r'\bsize="xs"\b'), 'use `size="sm"` (not `xs`)'),
    (re.compile(r"\bsize='xs'\b"), "use `size='sm'` (not `xs`)"),
]

ALL_RULES = ("ep", "theme", "ui", "canvas")


def _scan_tree(
    root: Path,
    label: str,
    patterns: list[tuple[re.Pattern[str], str]],
    failures: list[str],
) -> None:
    if not root.is_dir():
        failures.append(f"{label}: directory missing ({root})")
        return
    for path in root.rglob("*"):
        if path.suffix not in SCAN_SUFFIXES or "node_modules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for pattern, hint in patterns:
            if pattern.search(text):
                failures.append(f"{label}/{rel}: {hint}")


def _scan_lines(path: Path, label: str, rules: list[tuple[re.Pattern[str], str]], failures: list[str]) -> None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for pattern, hint in rules:
            if pattern.search(line):
                failures.append(f"{label}/{path.name}:{i}: {hint}")


def check_ep() -> list[str]:
    failures: list[str] = []
    _scan_tree(FRONTEND_SRC, "frontend/src", EP_PATTERNS, failures)
    for pkg in ("ui", "shell", "tokens"):
        _scan_tree(DQ_UI_ROOT / pkg / "src", f"dq-ui/{pkg}/src", EP_PATTERNS, failures)
    return failures


def _css_brace_depth(text: str) -> int:
    depth = 0
    for line in text.splitlines():
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        if depth < 0:
            return depth
    return depth


def check_theme() -> list[str]:
    failures: list[str] = []
    if STUDIO_STYLES.is_dir():
        for path in STUDIO_STYLES.glob("*.css"):
            text = path.read_text(encoding="utf-8")
            brace_depth = _css_brace_depth(text)
            if brace_depth != 0:
                failures.append(
                    f"frontend/styles/{path.name}: unbalanced '{{' '}}' (depth={brace_depth})"
                )
            for i, line in enumerate(text.splitlines(), 1):
                if THEME_STYLE_EL.search(line):
                    failures.append(f"frontend/styles/{path.name}:{i}: remove `.el-*` selector")
                if THEME_EL_TOKEN.search(line):
                    failures.append(f"frontend/styles/{path.name}:{i}: remove `--el-*` token")
                if THEME_BROKEN_MODEL_CARD.search(line):
                    failures.append(f"frontend/styles/{path.name}:{i}: fix corrupted `.model-card` selector")
                if THEME_EP_PREFIX.search(line):
                    failures.append(f"frontend/styles/{path.name}:{i}: remove *-ep-* layout class prefix")
    if FRONTEND_SRC.is_dir():
        for path in FRONTEND_SRC.rglob("*.vue"):
            text = path.read_text(encoding="utf-8")
            rel = path.as_posix().split("/src/", 1)[-1]
            for i, line in enumerate(text.splitlines(), 1):
                if THEME_FORBIDDEN_CLASS.search(line):
                    failures.append(f"frontend/{rel}:{i}: EP component class in template")
                if THEME_EP_PREFIX.search(line):
                    failures.append(f"frontend/{rel}:{i}: remove *-ep-* layout class prefix")
                if "--el-" in line:
                    failures.append(f"frontend/{rel}:{i}: use `--dq-*` instead of `--el-*`")
    if DQ_UI_ROOT.is_dir():
        for path in DQ_UI_ROOT.rglob("*.css"):
            if "node_modules" in path.parts:
                continue
            rel = path.relative_to(DQ_UI_ROOT)
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if THEME_STYLE_EL.search(line):
                    failures.append(f"dq-ui/{rel.parent}/{path.name}:{i}: remove `.el-*` selector")
                if THEME_EL_TOKEN.search(line):
                    failures.append(f"dq-ui/{rel.parent}/{path.name}:{i}: remove `--el-*` token")
                if THEME_EP_PREFIX.search(line):
                    failures.append(f"dq-ui/{rel.parent}/{path.name}:{i}: remove *-ep-* layout class prefix")
        for path in DQ_UI_ROOT.rglob("*.vue"):
            if "node_modules" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                if THEME_FORBIDDEN_CLASS.search(line):
                    failures.append(f"dq-ui/{path.relative_to(DQ_UI_ROOT)}:{i}: EP component class")
                if THEME_EP_PREFIX.search(line):
                    failures.append(f"dq-ui/{path.relative_to(DQ_UI_ROOT)}:{i}: remove *-ep-* prefix")
                if "--el-" in line:
                    failures.append(f"dq-ui/{path.relative_to(DQ_UI_ROOT)}:{i}: use `--dq-*` not `--el-*`")
    return failures


def check_ui() -> list[str]:
    failures: list[str] = []
    style_paths: list[Path] = []
    if STUDIO_STYLES.is_dir():
        style_paths.extend(STUDIO_STYLES.glob("*.css"))
    if DQ_UI_ROOT.is_dir():
        for pkg in ("ui", "shell", "tokens"):
            pkg_src = DQ_UI_ROOT / pkg / "src"
            if pkg_src.is_dir():
                style_paths.extend(pkg_src.rglob("*.css"))
    for path in style_paths:
        if "node_modules" not in path.parts:
            _scan_lines(path, "styles", UI_STYLE_RULES, failures)
    vue_roots = [FRONTEND_SRC]
    if DQ_UI_ROOT.is_dir():
        vue_roots.extend((DQ_UI_ROOT / pkg / "src" for pkg in ("ui", "shell")))
    for root in vue_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.vue"):
            if "node_modules" not in path.parts:
                try:
                    rel = path.relative_to(ROOT).as_posix()
                except ValueError:
                    rel = path.as_posix()
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    for pattern, hint in UI_VUE_RULES:
                        if pattern.search(line):
                            failures.append(f"vue/{rel}:{i}: {hint}")
    return failures


def check_canvas() -> list[str]:
    failures: list[str] = []
    frontend = ROOT / "frontend"
    pkg = frontend / "package.json"
    if not pkg.is_file():
        failures.append("frontend/package.json missing")
        return failures
    proc = subprocess.run(
        ["npm", "run", "canvas-unit"],
        cwd=frontend,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-3:] if detail else ["unknown error"]
        failures.append("canvas-unit failed: " + " | ".join(tail))
    return failures


RULE_RUNNERS: dict[str, Callable[[], list[str]]] = {
    "ep": check_ep,
    "theme": check_theme,
    "ui": check_ui,
    "canvas": check_canvas,
}


def run_rules(rules: tuple[str, ...]) -> int:
    failed = False
    for rule in rules:
        violations = RULE_RUNNERS[rule]()
        if violations:
            failed = True
            print(f"[{rule}] frontend governance failed ({len(violations)}):", file=sys.stderr)
            for item in violations[:50]:
                print(f"  - {item}", file=sys.stderr)
            if len(violations) > 50:
                print(f"  … and {len(violations) - 50} more", file=sys.stderr)
    if failed:
        return 1
    if len(rules) == 1:
        print(f"Frontend governance [{rules[0]}] OK")
    else:
        print("Frontend governance OK (Studio + dq-ui)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rule", choices=ALL_RULES, action="append", help="Run one rule (default: all).")
    args = ap.parse_args()
    rules = tuple(args.rule) if args.rule else ALL_RULES
    return run_rules(rules)


if __name__ == "__main__":
    raise SystemExit(main())
