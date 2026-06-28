#!/usr/bin/env python
"""Unified validation runner and summarizer for DSMC/BAMPS cases.

The default mode is intentionally light: it reads existing artifacts and writes
machine-readable summaries. Use --run to launch expensive simulation scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


@dataclass
class MetricResult:
    name: str
    status: str
    value: float | None = None
    detail: str = ""
    threshold: dict[str, Any] | None = None


@dataclass
class ComparisonResult:
    name: str
    status: str
    bamps: float | None = None
    lammps: float | None = None
    abs_error: float | None = None
    rel_error_pct: float | None = None
    detail: str = ""
    threshold: dict[str, Any] | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_case_path(case_arg: str) -> Path:
    candidate = Path(case_arg)
    if candidate.is_file():
        return candidate.resolve()

    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")

    search_paths = [
        ROOT / candidate,
        ROOT / "cases" / candidate.name,
    ]
    for path in search_paths:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"Cannot find validation case config: {case_arg}")


def case_workdir(case_config: dict[str, Any], case_path: Path) -> Path:
    configured = case_config.get("working_dir", ".")
    return (case_path.parent / configured).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def window_rows(rows: list[dict[str, str]], window: str | None) -> list[dict[str, str]]:
    if not rows:
        return rows
    if window in (None, "all"):
        return rows
    if window == "second_half":
        return rows[len(rows) // 2 :]
    if window == "last_80_percent":
        return rows[len(rows) // 5 :]
    if window.startswith("last_fraction:"):
        frac = float(window.split(":", 1)[1])
        start = max(0, min(len(rows) - 1, int(len(rows) * (1.0 - frac))))
        return rows[start:]
    raise ValueError(f"Unknown window specifier: {window}")


def column_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = [to_float(row.get(column)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan, math.nan, math.nan
    x_bar = mean(xs)
    y_bar = mean(ys)
    ss_xx = sum((x - x_bar) ** 2 for x in xs)
    if ss_xx == 0:
        return math.nan, math.nan, math.nan
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / ss_xx
    intercept = y_bar - slope * x_bar
    ss_tot = sum((y - y_bar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def check_threshold(value: float | None, threshold: dict[str, Any] | None) -> tuple[str, str]:
    if threshold is None:
        return "info", "no threshold"
    if value is None or not math.isfinite(value):
        return "fail", "value is not finite"

    checks: list[tuple[bool, str]] = []
    if "min" in threshold:
        checks.append((value >= float(threshold["min"]), f">= {threshold['min']}"))
    if "max" in threshold:
        checks.append((value <= float(threshold["max"]), f"<= {threshold['max']}"))
    if "target" in threshold and "max_abs_error" in threshold:
        target = float(threshold["target"])
        max_error = float(threshold["max_abs_error"])
        checks.append((abs(value - target) <= max_error, f"|value - {target}| <= {max_error}"))

    if not checks:
        return "info", "threshold has no scalar rule"
    failed = [text for ok, text in checks if not ok]
    if failed:
        return "fail", "; ".join(failed)
    return "pass", "; ".join(text for _, text in checks)


def compare_values(
    name: str,
    bamps: float | None,
    lammps: float | None,
    threshold: dict[str, Any] | None = None,
    detail_prefix: str = "",
) -> ComparisonResult:
    if bamps is None or not math.isfinite(bamps):
        return ComparisonResult(name, "missing_bamps", detail=f"{detail_prefix}BAMPS value is missing")
    if lammps is None or not math.isfinite(lammps):
        return ComparisonResult(name, "missing_reference", bamps=bamps, detail=f"{detail_prefix}LAMMPS value is missing")

    abs_error = abs(bamps - lammps)
    denom = abs(lammps)
    rel_error_pct = 0.0 if denom == 0.0 and abs_error == 0.0 else (100.0 * abs_error / max(denom, 1.0e-12))
    threshold = threshold or {}
    checks: list[tuple[bool, str]] = []
    if "abs_error_max" in threshold:
        checks.append((abs_error <= float(threshold["abs_error_max"]), f"abs_error <= {threshold['abs_error_max']}"))
    if "rel_error_pct_max" in threshold:
        checks.append((rel_error_pct <= float(threshold["rel_error_pct_max"]), f"rel_error_pct <= {threshold['rel_error_pct_max']}"))

    if not checks:
        status = "info"
        detail = "no threshold"
    else:
        failed = [text for ok, text in checks if not ok]
        status = "fail" if failed else "pass"
        detail = "; ".join(failed or [text for _ok, text in checks])

    if detail_prefix:
        detail = f"{detail_prefix}{detail}"
    return ComparisonResult(
        name,
        status,
        bamps=bamps,
        lammps=lammps,
        abs_error=abs_error,
        rel_error_pct=rel_error_pct,
        detail=detail,
        threshold=threshold,
    )


def evaluate_csv_column(metric: dict[str, Any], workdir: Path) -> MetricResult:
    path = workdir / metric["path"]
    if not path.is_file():
        status = "missing_required" if metric.get("required", True) else "missing_optional"
        return MetricResult(metric["name"], status, detail=f"missing {path}")

    rows = window_rows(read_csv_rows(path), metric.get("window"))
    values = column_values(rows, metric["column"])
    stat = metric.get("stat", "mean")

    if not values:
        return MetricResult(metric["name"], "fail", detail=f"no numeric values in {path}:{metric['column']}")

    if stat == "mean":
        value = mean(values)
    elif stat == "std":
        value = std(values)
    elif stat == "last":
        value = values[-1]
    elif stat == "relative_drift_abs_max":
        start = values[0]
        denom = max(abs(start), 1.0e-12)
        value = max(abs((value - start) / denom) for value in values)
    else:
        return MetricResult(metric["name"], "fail", detail=f"unknown stat {stat}")

    status, detail = check_threshold(value, metric.get("threshold"))
    return MetricResult(metric["name"], status, value=value, detail=detail, threshold=metric.get("threshold"))


def evaluate_summary_observable(metric: dict[str, Any], workdir: Path) -> MetricResult:
    path = workdir / metric["path"]
    if not path.is_file():
        status = "missing_required" if metric.get("required", True) else "missing_optional"
        return MetricResult(metric["name"], status, detail=f"missing {path}")

    rows = read_csv_rows(path)
    target_row = next((row for row in rows if row.get("observable") == metric["observable"]), None)
    if target_row is None:
        return MetricResult(metric["name"], "fail", detail=f"observable {metric['observable']} not found")

    column = metric.get("error_column", "rel_error_pct")
    value = to_float(target_row.get(column))
    status, detail = check_threshold(value, metric.get("threshold"))
    return MetricResult(metric["name"], status, value=value, detail=detail, threshold=metric.get("threshold"))


def evaluate_msd_fit(metric: dict[str, Any], workdir: Path) -> list[MetricResult]:
    path = workdir / metric["path"]
    if not path.is_file():
        status = "missing_required" if metric.get("required", True) else "missing_optional"
        return [MetricResult(metric["name"], status, detail=f"missing {path}")]

    rows = window_rows(read_csv_rows(path), metric.get("window"))
    steps = column_values(rows, metric["step_column"])
    msd = column_values(rows, metric["msd_column"])
    count = min(len(steps), len(msd))
    steps = steps[:count]
    msd = msd[:count]
    dt = float(metric.get("dt", 1.0))
    time = [step * dt for step in steps]
    slope, _intercept, r2 = linear_fit(time, msd)
    diffusion = slope / 6.0 if math.isfinite(slope) else math.nan
    threshold = metric.get("threshold", {})

    results: list[MetricResult] = []
    if "r2" in metric.get("outputs", ["r2"]):
        status = "pass" if math.isfinite(r2) and r2 >= float(threshold.get("r2_min", -math.inf)) else "fail"
        results.append(
            MetricResult(
                f"{metric['name']}.r2",
                status,
                value=r2,
                detail=f">= {threshold.get('r2_min', '-inf')}",
                threshold={"min": threshold.get("r2_min")},
            )
        )
    if "D" in metric.get("outputs", ["D"]):
        d_threshold: dict[str, Any] = {}
        if "D_min" in threshold:
            d_threshold["min"] = threshold["D_min"]
        if "D_max" in threshold:
            d_threshold["max"] = threshold["D_max"]
        status, detail = check_threshold(diffusion, d_threshold)
        results.append(
            MetricResult(
                f"{metric['name']}.D",
                status,
                value=diffusion,
                detail=detail,
                threshold=d_threshold,
            )
        )
    return results


def evaluate_metric(metric: dict[str, Any], workdir: Path) -> list[MetricResult]:
    metric_type = metric.get("type")
    if metric_type == "csv_column":
        return [evaluate_csv_column(metric, workdir)]
    if metric_type == "summary_observable":
        return [evaluate_summary_observable(metric, workdir)]
    if metric_type == "msd_fit":
        return evaluate_msd_fit(metric, workdir)
    return [MetricResult(metric.get("name", "unnamed"), "fail", detail=f"unknown metric type {metric_type}")]


def csv_column_stat(path: Path, column: str, stat: str, window: str | None) -> float | None:
    if not path.is_file():
        return None
    rows = window_rows(read_csv_rows(path), window)
    values = column_values(rows, column)
    if not values:
        return None
    if stat == "mean":
        return mean(values)
    if stat == "std":
        return std(values)
    if stat == "last":
        return values[-1]
    raise ValueError(f"Unknown comparison stat {stat}")


def evaluate_summary_comparison(comparison: dict[str, Any], workdir: Path) -> list[ComparisonResult]:
    path = workdir / comparison["path"]
    observables = comparison.get("observables", [])
    if not path.is_file():
        status = "missing_required" if comparison.get("required", True) else "missing_reference"
        return [
            ComparisonResult(
                observable["name"],
                status,
                detail=f"missing comparison table {path}",
                threshold=observable.get("threshold"),
            )
            for observable in observables
        ]

    rows = read_csv_rows(path)
    by_observable = {row.get("observable"): row for row in rows}
    results: list[ComparisonResult] = []
    for observable in observables:
        name = observable["name"]
        row = by_observable.get(name)
        if row is None:
            results.append(ComparisonResult(name, "missing_observable", detail=f"{name} not found in {path}"))
            continue
        bamps = to_float(row.get("BAMPS"))
        lammps = to_float(row.get("LAMMPS"))
        result = compare_values(name, bamps, lammps, observable.get("threshold"))
        if "rel_error_pct" in row and math.isfinite(to_float(row.get("rel_error_pct"))):
            result.rel_error_pct = to_float(row.get("rel_error_pct"))
            threshold = observable.get("threshold", {})
            if "rel_error_pct_max" in threshold:
                ok = result.rel_error_pct <= float(threshold["rel_error_pct_max"])
                result.status = "pass" if ok else "fail"
                result.detail = f"rel_error_pct <= {threshold['rel_error_pct_max']}"
        results.append(result)
    return results


def evaluate_paired_csv_comparison(comparison: dict[str, Any], workdir: Path) -> list[ComparisonResult]:
    bamps_path = workdir / comparison["bamps_path"]
    lammps_path = workdir / comparison["lammps_path"]
    bamps = csv_column_stat(
        bamps_path,
        comparison["bamps_column"],
        comparison.get("stat", "mean"),
        comparison.get("window"),
    )
    lammps = csv_column_stat(
        lammps_path,
        comparison["lammps_column"],
        comparison.get("stat", "mean"),
        comparison.get("window"),
    )
    detail_prefix = ""
    if not bamps_path.is_file():
        detail_prefix += f"missing BAMPS file {bamps_path}; "
    if not lammps_path.is_file():
        detail_prefix += f"missing LAMMPS file {lammps_path}; "
    return [
        compare_values(
            comparison["name"],
            bamps,
            lammps,
            comparison.get("threshold"),
            detail_prefix=detail_prefix,
        )
    ]


def evaluate_comparisons(case_config: dict[str, Any], workdir: Path) -> list[ComparisonResult]:
    results: list[ComparisonResult] = []
    for comparison in case_config.get("comparisons", []):
        comparison_type = comparison.get("type")
        if comparison_type == "summary_table":
            results.extend(evaluate_summary_comparison(comparison, workdir))
        elif comparison_type == "paired_csv_column":
            results.extend(evaluate_paired_csv_comparison(comparison, workdir))
        else:
            results.append(
                ComparisonResult(
                    comparison.get("name", "unnamed"),
                    "fail",
                    detail=f"unknown comparison type {comparison_type}",
                )
            )
    return results


def artifact_results(case_config: dict[str, Any], workdir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in case_config.get("artifacts", []):
        path = workdir / artifact["path"]
        exists = path.is_file()
        required = bool(artifact.get("required", True))
        if exists:
            status = "present"
            size = path.stat().st_size
        else:
            status = "missing_required" if required else "missing_optional"
            size = 0
        rows.append(
            {
                "name": artifact["name"],
                "path": artifact["path"],
                "required": required,
                "status": status,
                "size_bytes": size,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(
    path: Path,
    case_config: dict[str, Any],
    metrics: list[MetricResult],
    comparisons: list[ComparisonResult],
    artifacts: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for result in metrics if result.status == "pass")
    fail_count = sum(1 for result in metrics if result.status == "fail")
    missing_required = sum(1 for result in metrics if result.status == "missing_required")
    missing_optional = sum(1 for result in metrics if result.status == "missing_optional")
    comparison_pass = sum(1 for result in comparisons if result.status == "pass")
    comparison_fail = sum(1 for result in comparisons if result.status == "fail")
    comparison_missing = sum(1 for result in comparisons if result.status in {"missing_reference", "missing_required", "missing_bamps", "missing_observable"})

    lines = [
        f"# Validation Report: {case_config['name']}",
        "",
        case_config.get("description", ""),
        "",
        "## Summary",
        "",
        f"- Passed metrics: {pass_count}",
        f"- Failed metrics: {fail_count}",
        f"- Missing required metrics: {missing_required}",
        f"- Missing optional metrics: {missing_optional}",
        f"- Passed comparisons: {comparison_pass}",
        f"- Failed comparisons: {comparison_fail}",
        f"- Missing comparisons/reference data: {comparison_missing}",
        "",
        "## Check Metrics",
        "",
        "| Metric | Status | Value | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    def md_cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    for result in metrics:
        value = "" if result.value is None else f"{result.value:.8g}"
        lines.append(f"| {md_cell(result.name)} | {md_cell(result.status)} | {value} | {md_cell(result.detail)} |")

    lines.extend(
        [
            "",
            "## BAMPS vs LAMMPS Comparisons",
            "",
            "| Observable | Status | BAMPS | LAMMPS | Abs Error | Rel Error % | Detail |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in comparisons:
        bamps = "" if result.bamps is None else f"{result.bamps:.8g}"
        lammps = "" if result.lammps is None else f"{result.lammps:.8g}"
        abs_error = "" if result.abs_error is None else f"{result.abs_error:.8g}"
        rel_error = "" if result.rel_error_pct is None else f"{result.rel_error_pct:.8g}"
        lines.append(
            f"| {md_cell(result.name)} | {md_cell(result.status)} | {bamps} | {lammps} | {abs_error} | {rel_error} | {md_cell(result.detail)} |"
        )

    lines.extend(["", "## Artifacts", "", "| Artifact | Status | Required | Path |", "| --- | --- | --- | --- |"])
    for artifact in artifacts:
        lines.append(
            f"| {md_cell(artifact['name'])} | {md_cell(artifact['status'])} | {artifact['required']} | {md_cell(artifact['path'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case_steps(case_config: dict[str, Any], workdir: Path, dry_run: bool = False) -> None:
    for step in case_config.get("run_steps", []):
        command = step["command"]
        print(f"[run] {step['name']}: {' '.join(command)}")
        if dry_run:
            continue
        subprocess.run(command, cwd=workdir, check=True)


def summarize_case(case_path: Path, output_dir: Path | None = None) -> int:
    case_config = load_json(case_path)
    workdir = case_workdir(case_config, case_path)
    output_dir = output_dir or (ROOT / "results" / case_config["name"])

    artifacts = artifact_results(case_config, workdir)
    metric_results: list[MetricResult] = []
    for metric in case_config.get("metrics", []):
        metric_results.extend(evaluate_metric(metric, workdir))
    comparison_results = evaluate_comparisons(case_config, workdir)

    metric_rows = [
        {
            "name": result.name,
            "status": result.status,
            "value": "" if result.value is None else result.value,
            "detail": result.detail,
            "threshold": json.dumps(result.threshold or {}, ensure_ascii=True),
        }
        for result in metric_results
    ]
    comparison_rows = [
        {
            "name": result.name,
            "status": result.status,
            "BAMPS": "" if result.bamps is None else result.bamps,
            "LAMMPS": "" if result.lammps is None else result.lammps,
            "abs_error": "" if result.abs_error is None else result.abs_error,
            "rel_error_pct": "" if result.rel_error_pct is None else result.rel_error_pct,
            "detail": result.detail,
            "threshold": json.dumps(result.threshold or {}, ensure_ascii=True),
        }
        for result in comparison_results
    ]
    write_csv(output_dir / "check_metrics.csv", metric_rows, ["name", "status", "value", "detail", "threshold"])
    write_csv(
        output_dir / "comparison_metrics.csv",
        comparison_rows,
        ["name", "status", "BAMPS", "LAMMPS", "abs_error", "rel_error_pct", "detail", "threshold"],
    )
    # Backward-compatible combined table. The comparison columns make it clear
    # which rows are scalar checks and which rows are reference comparisons.
    combined_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        combined_rows.append(
            {
                "kind": "check",
                "name": row["name"],
                "status": row["status"],
                "value": row["value"],
                "BAMPS": row["value"],
                "LAMMPS": "",
                "abs_error": "",
                "rel_error_pct": "",
                "detail": row["detail"],
                "threshold": row["threshold"],
            }
        )
    for row in comparison_rows:
        combined_rows.append(
            {
                "kind": "comparison",
                "name": row["name"],
                "status": row["status"],
                "value": "",
                "BAMPS": row["BAMPS"],
                "LAMMPS": row["LAMMPS"],
                "abs_error": row["abs_error"],
                "rel_error_pct": row["rel_error_pct"],
                "detail": row["detail"],
                "threshold": row["threshold"],
            }
        )
    write_csv(
        output_dir / "metrics.csv",
        combined_rows,
        ["kind", "name", "status", "value", "BAMPS", "LAMMPS", "abs_error", "rel_error_pct", "detail", "threshold"],
    )
    write_csv(
        output_dir / "artifacts.csv",
        artifacts,
        ["name", "path", "required", "status", "size_bytes"],
    )
    write_report(output_dir / "validation_report.md", case_config, metric_results, comparison_results, artifacts)

    failing_statuses = {"fail", "missing_required"}
    has_failures = any(result.status in failing_statuses for result in metric_results)
    has_failures = has_failures or any(result.status == "fail" for result in comparison_results)
    has_failures = has_failures or any(row["status"] == "missing_required" for row in artifacts)
    print(f"[summary] wrote {output_dir}")
    return 1 if has_failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="lj_argon_baseline", help="Case name or JSON path.")
    parser.add_argument("--output-dir", default=None, help="Directory for normalized validation outputs.")
    parser.add_argument("--run", action="store_true", help="Run case simulation scripts before summarizing.")
    parser.add_argument("--dry-run", action="store_true", help="Print run commands without executing them.")
    parser.add_argument("--allow-fail", action="store_true", help="Always exit 0 after writing reports.")
    args = parser.parse_args(argv)

    case_path = resolve_case_path(args.case)
    case_config = load_json(case_path)
    workdir = case_workdir(case_config, case_path)

    if args.run or args.dry_run:
        run_case_steps(case_config, workdir, dry_run=args.dry_run)
        if args.dry_run:
            return 0

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    exit_code = summarize_case(case_path, output_dir)
    return 0 if args.allow_fail else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
