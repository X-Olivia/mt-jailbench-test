#!/usr/bin/env python3
"""MT-JailBench batch runner and XLSX/CSV reporter.

Run from the mt-jailbench repository root. Requires PyYAML (already in project)
and XlsxWriter (optional for XLSX; install with `uv pip install ... xlsxwriter`).
Does not mutate run_demo.py or source files.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

OUTCOMES = {"SUCCESS_BY_SELF_EVAL", "SUCCESS_BY_INDP_EVAL", "FAILURE_BY_INDP_EVAL"}


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:100]


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_excel(path, rows, details):
    try:
        import xlsxwriter
    except ImportError:
        print("XLSX skipped: install XlsxWriter into project .venv. CSV is available.")
        return
    wb = xlsxwriter.Workbook(str(path))
    head = wb.add_format({"bold": True, "bg_color": "#1C3D5A", "font_color": "white", "border": 0, "text_wrap": True})
    pct = wb.add_format({"num_format": "0.0%"})
    warning = wb.add_format({"bg_color": "#FFF3CC"})
    for title, data in [("Summary", rows), ("Behaviors", details)]:
        ws = wb.add_worksheet(title)
        if not data:
            ws.write(0, 0, "No rows yet")
            continue
        cols = list(data[0].keys())
        for j, key in enumerate(cols):
            ws.write(0, j, key, head)
            ws.set_column(j, j, min(42, max(13, len(key) + 2)))
        for i, row in enumerate(data, start=1):
            for j, key in enumerate(cols):
                val = row.get(key)
                fmt = pct if key.endswith("_asr") and isinstance(val, (int, float)) else None
                if isinstance(val, bool):
                    ws.write_boolean(i, j, val)
                elif isinstance(val, (int, float)):
                    ws.write_number(i, j, val, fmt)
                elif val is not None:
                    # Avoid Excel formulas from generated values
                    text = str(val)
                    ws.write_string(i, j, text, warning if text.startswith("ERROR") else None)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(data), len(cols) - 1)
    readme = wb.add_worksheet("Readme")
    notes = [
        "MT-JailBench experiment matrix", "Self-reported ASR = successful completed behaviors / completed behaviors.",
        "Independent-judge ASR reported only if enabled and available.",
        "Do not treat failed or timed-out behaviors as unsuccessful model attacks.",
        "Planned behaviors are the first N behaviors returned by the repo's loader.",
        "Outputs under outputs/matrix_runs/ are excluded from Git by default; archive only sanitized summaries.",
    ]
    for i, t in enumerate(notes):
        readme.write_string(i, 0, t)
    readme.set_column(0, 0, 110)
    wb.close()


def read_behavior_logs(run_dir, expected):
    folder = run_dir / "logs"
    rows = []
    if folder.exists():
        for path in sorted(folder.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                oc = obj.get("outcome_code")
                indep = obj.get("independent_judge") or {}
                rows.append({
                    "behavior_id": obj.get("behavior_id", path.stem),
                    "outcome_code": oc,
                    "self_reported_success": oc in OUTCOMES,
                    "independent_judge_success": indep.get("attack_successful") if indep.get("status") == "triggered" else None,
                    "turns_used": obj.get("total_turns_used"),
                    "epochs_used": obj.get("total_epochs_used"),
                    "log_file": str(path),
                })
            except Exception as exc:
                rows.append({"behavior_id": path.stem, "outcome_code": "INVALID_LOG", "self_reported_success": None,
                             "independent_judge_success": None, "turns_used": None, "epochs_used": None,
                             "log_file": str(path), "parse_error": str(exc)})
    valid = [x for x in rows if x.get("outcome_code") not in ("INVALID_LOG", None)]
    self_success = sum(x["self_reported_success"] is True for x in valid)
    indep_judged = [x for x in valid if isinstance(x["independent_judge_success"], bool)]
    return rows, len(valid), self_success, (sum(x["independent_judge_success"] for x in indep_judged) / len(indep_judged) if indep_judged else None)


def _wait(r, timeout):
    """Wait for one condition's process; timeout counts from that condition's own start."""
    remaining = max(1, timeout - (time.monotonic() - r["start"]))
    try:
        code = r["proc"].wait(timeout=remaining)
        r["status"] = "finished" if code == 0 else f"exit_{code}"
    except subprocess.TimeoutExpired:
        r["proc"].kill()
        r["proc"].wait()
        r["status"] = "timeout"
    r["logfile"].close()
    r["elapsed"] = round(time.monotonic() - r["start"], 1)


def _merge(path, new_rows, names):
    """Keep CSV rows from conditions not re-run in this invocation, replace the rest."""
    old = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            old = [row for row in csv.DictReader(f) if row.get("condition") not in names]
    return old + new_rows


def main():
    p = argparse.ArgumentParser(description="Run a model x attack-method MT-JailBench matrix")
    p.add_argument("--plan", default="experiments/matrix.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="*", help="Optional condition names to run")
    p.add_argument("--parallel", action="store_true", help="Run all selected conditions at the same time")
    args = p.parse_args()
    repo = Path.cwd()
    if not (repo / "run_benchmark.py").exists():
        sys.exit("Run from the mt-jailbench repository root")
    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
    output_dir = Path(plan.get("output_dir", "outputs/matrix_runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(plan.get("report_dir", "experiments/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in plan["conditions"] if c.get("enabled", True) and (not args.only or c["name"] in args.only)]
    if not conditions:
        sys.exit("No matching conditions")
    base = plan["shared"]
    expected = int(base["num_behaviors"])
    timeout = int(plan.get("timeout_seconds", 1200))
    py = str(repo / ".venv/bin/python")
    if not Path(py).exists():
        sys.exit("Missing .venv/bin/python")
    summary_rows, behavior_rows, running = [], [], []
    for c in conditions:
        name = safe_name(c["name"])
        attacker = c["attacker"]
        if attacker.get("model", "").startswith("REPLACE_") or (attacker.get("provider") == "local" and not attacker.get("base_url")):
            sys.exit(f"Condition {name}: set a real attacker model and local base_url in {args.plan}")
        conf = {
            "num_behaviors": expected, "concurrency_limit": int(base.get("concurrency_limit", 1)),
            "max_turns": int(base.get("max_turns", 2)), "max_epochs": int(base.get("max_epochs", 1)),
            "independent_judge": bool(base.get("independent_judge", False)),
            "target_config": base["target"], "defense": None,
            "attack_type": "XTeaming" if c["method"] == "xteaming" else "Mix",
        }
        judge = base["judge"]
        if c["method"] == "xteaming":
            conf["attack_config"] = {
                "attacker_model": attacker, "strategy_model": attacker,
                "judge_model": judge, "textgrad_engine_model": attacker,
                "use_multiple_strategies": False, "max_refines_per_turn": int(base.get("max_refines_per_turn", 2)),
                "num_sets": 1,
            }
        elif c["method"] == "mix":
            conf["attack_config"] = {
                "generator": "Crescendo", "updater": "XTeaming", "judge_and_flow": "XTeaming",
                "attacker_model": attacker, "judge_model": judge,
                "use_llm_for_similarity": False, "max_refines_per_turn": int(base.get("max_refines_per_turn", 2)),
                "max_restarts": 0,
            }
        else:
            sys.exit(f"Invalid method for {name}: {c['method']}")
        config_path = output_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")
        # Nested Mix modules are AttackType enum objects in the engine, not plain strings.
        entrypoint = (
            "import sys; import run_benchmark as b; from engine.attack_type import AttackType; "
            "config=b.load_config(sys.argv[1]); "
            "ac=config['attack_config']; "
            "[ac.__setitem__(k, AttackType(ac[k])) for k in ('generator','updater','judge_and_flow') if k in ac]; "
            "b.run_benchmark(config,sys.argv[1],sys.argv[2],sys.argv[3])"
        )
        cmd = [py, "-c", entrypoint, str(config_path), name, str(output_dir)]
        if args.dry_run:
            print(f"DRY RUN {name}: {cmd}\n{config_path.read_text()}")
            continue
        log_path = output_dir / f"{name}.console.log"
        logfile = log_path.open("w", encoding="utf-8")
        # Each condition is its own process (textgrad's backward engine is process-global).
        proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
        print(f">>> Started {name} (pid {proc.pid}), log {log_path}", flush=True)
        running.append({"c": c, "name": name, "attacker": attacker, "judge": judge, "proc": proc,
                        "logfile": logfile, "log_path": log_path, "start": time.monotonic()})
        if not args.parallel:
            _wait(running[-1], timeout)
    if args.dry_run:
        print(f"\nReports: {report_dir}")
        return

    for r in running:
        if "status" not in r:
            _wait(r, timeout)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for r in running:
        c, name, attacker, judge = r["c"], r["name"], r["attacker"], r["judge"]
        run_dir = output_dir / name
        behavior, completed, succeeded, indp_asr = read_behavior_logs(run_dir, expected)
        for b in behavior:
            behavior_rows.append({"condition": name, "method": c["method"], "attacker": attacker["model"], **b})
        summary_rows.append({
            "condition": name, "method": c["method"], "attacker": attacker["model"],
            "target": base["target"]["model"], "judge": judge["model"],
            "planned": expected, "completed": completed, "errors_or_missing": max(0, expected - completed),
            "self_reported_successes": succeeded,
            "self_reported_asr": succeeded / completed if completed else None,
            "independent_judge_asr": indp_asr,
            "status": r["status"] if completed == expected else r["status"] + "_partial",
            "elapsed_seconds": r["elapsed"],
            "git_commit": commit,
            "console_log": str(r["log_path"]), "run_dir": str(run_dir),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"{name}: {completed}/{expected} logs, successes={succeeded}, status={summary_rows[-1]['status']}", flush=True)

    # Merge with rows from earlier invocations so separate runs don't overwrite each other's reports.
    summary_rows = _merge(report_dir / "matrix_summary.csv", summary_rows, {r["name"] for r in running})
    behavior_rows = _merge(report_dir / "matrix_behaviors.csv", behavior_rows, {r["name"] for r in running})
    save_csv(report_dir / "matrix_summary.csv", summary_rows, list(dict.fromkeys(k for r in summary_rows for k in r)))
    if behavior_rows:
        save_csv(report_dir / "matrix_behaviors.csv", behavior_rows, list(dict.fromkeys(k for r in behavior_rows for k in r)))
    save_excel(report_dir / "matrix_results.xlsx", summary_rows, behavior_rows)
    print(f"\nReports: {report_dir}")


if __name__ == "__main__":
    main()
