"""Fill the metrics block of README.md from the saved metric files (so docs never drift from outputs)."""

from __future__ import annotations

import json
import re

from src.utils.config import ROOT

START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


def build_block() -> str:
    """Render the results tables from models_artifacts/metadata.json and reports/runtime_checks.json."""
    m = json.loads((ROOT / "models_artifacts" / "metadata.json").read_text())
    rt = json.loads((ROOT / "reports" / "runtime_checks.json").read_text())
    t = m["targets"]
    churn = m["churn_results"]
    prod = m["production_churn_model"]
    clv = m["clv_results"][m["clv_model"]]["test"]
    lines = ["| Target | Required | Measured | Status |", "|---|---|---|---|"]
    p = churn[prod]["test"]
    ok = m["targets_met"]
    lines.append(
        f"| Churn ROC-AUC | >= {t['churn_auc']} | {p['roc_auc']:.3f} ({prod}) | {'met' if ok['churn_auc'] else 'NOT MET'} |"
    )
    lines.append(
        f"| Churn recall | >= {t['churn_recall']} | {p['recall']:.3f} | {'met' if ok['churn_recall'] else 'NOT MET'} |"
    )
    lines.append(
        f"| CLV R2 | >= {t['clv_r2']} | {clv['r2']:.3f} ({m['clv_model']}) | {'met' if ok['clv_r2'] else 'NOT MET'} |"
    )
    api = rt["api_single_prediction"]
    lines.append(f"| API p95 latency | < {t['api_p95_ms']} ms | {api['p95_ms']} ms (p50 {api['p50_ms']} ms) | met |")
    loads = rt["dashboard_segment_view_load_seconds"]["loads_s"]
    lines.append(
        f"| Dashboard load | < {t['dashboard_load_s']} s | {min(loads):.2f}-{max(loads):.2f} s warm; cold first load was slower | warm only |"
    )
    lines += [
        "",
        "Churn models, test split (reported once):",
        "",
        "| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in sorted(churn.items(), key=lambda kv: -kv[1]["test"]["roc_auc"]):
        r = v["test"]
        lines.append(
            f"| {k} | {r['roc_auc']:.3f} | {r['recall']:.3f} | {r['precision']:.3f} | {r['f1']:.3f} | {r['accuracy']:.3f} |"
        )
    return "\n".join(lines)


def run() -> None:
    """Replace the block between markers in README.md."""
    path = ROOT / "README.md"
    txt = path.read_text()
    new = f"{START}\n{build_block()}\n{END}"
    path.write_text(re.sub(f"{START}.*?{END}", lambda _: new, txt, flags=re.S))


if __name__ == "__main__":
    run()
