"""Markdown reports and matplotlib charts from recomputed saved-log metrics."""

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import STRATEGIES
from .analysis import aggregate, load_logs


def display(value, *, percent=False):
    return "N/A" if value is None else (f"{value:.1%}" if percent else f"{value:.4f}")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def charts(metrics, output, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2563eb", "#059669", "#d97706"]
    label = "MOCK / DRY RUN — synthetic fixture; no VLM performance claim" if mode == "mock" else "REAL — saved experiment logs"
    for key, title, filename in (("accuracy", "Answer accuracy", "strategy_accuracy.png"),
                                 ("evidence_hit_rate", "Temporal evidence hit rate", "evidence_hit.png")):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        values = [metrics[s][key] for s in STRATEGIES]
        bars = ax.bar(range(3), values, color=colors, width=0.55)
        ax.set_xticks(range(3), [f"{s}\nn={metrics[s]['n']}" for s in STRATEGIES])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Proportion of ALL planned task attempts")
        ax.set_title(f"{title}\n{label}", fontsize=11)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center")
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    panels = [("mean_elapsed_seconds", "End-to-end latency (seconds)", "latency_n"),
              ("mean_requests", "Model requests (attempted)", "latency_n"),
              ("mean_observed_frames", "Observed frames", "latency_n"),
              ("tokens", "Tokens (usage available; cost proxy)", "usage_n"),
              ("mean_estimated_cost", "Estimated cost (user price currency)", "cost_n")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (key, title, n_key) in zip(axes.flat, panels):
        for i, strategy in enumerate(STRATEGIES):
            values = metrics[strategy]
            value = (values["mean_input_tokens"] + values["mean_output_tokens"]
                     if key == "tokens" and values["mean_input_tokens"] is not None else
                     None if key == "tokens" else values[key])
            if value is None:
                ax.text(i, 0.02, "N/A", ha="center", transform=ax.get_xaxis_transform())
            else:
                ax.bar(i, value, color=colors[i], width=0.55)
                ax.annotate(f"{value:.4g}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_xticks(range(3), [f"{s}\nn={metrics[s]['n']}; available={metrics[s][n_key]}" for s in STRATEGIES], fontsize=8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.22)
    axes.flat[-1].axis("off")
    mode_note = ("Mock latency is Python fixture runtime.\nMock tokens are synthetic, not provider usage." if mode == "mock"
                 else "Tokens come from logged provider usage.\nCost uses user-supplied prices, not billing data.")
    axes.flat[-1].text(0, 0.9, "All failures retained in accuracy/hit denominators.\nResource means use observed values only.\nUnavailable usage/cost is N/A, never zero.\n" + mode_note, va="top", fontsize=10)
    fig.suptitle(f"Latency / resources / cost comparison\n{label}", fontsize=12)
    fig.savefig(output / "resources_cost.png", dpi=150)
    plt.close(fig)


def generate_reports(paths, report_dir, *, mode, run_id=None):
    records, manifests, notes = load_logs(paths, mode=mode, run_id=run_id)
    metrics = {s: aggregate([r for r in records if r["strategy"] == s]) for s in STRATEGIES}
    overall = aggregate(records)
    root = Path(report_dir).resolve()
    archive = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8])
    archive.mkdir(parents=True, exist_ok=False)
    selected_runs = {m["run_id"] for m in manifests}
    inputs = []
    for path in paths:
        path = Path(path).resolve()
        with path.open(encoding="utf-8-sig") as stream:
            first = json.loads(next(line for line in stream if line.strip()))
        if first["run_id"] in selected_runs:
            inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    saved = {"mode": mode, "run_ids": sorted(selected_runs), "inputs": inputs, "overall": overall,
             "strategies": metrics, "notes": notes}
    (archive / "metrics.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    charts(metrics, archive, mode)
    heading = "MOCK / DRY RUN — 未运行真实 VLM" if mode == "mock" else "REAL — 真实 runner 日志"
    lines = [f"# EgoAgent-Eval: {heading}", "", "所有图表和指标均从下列已保存 JSONL 重算，图中 n 为计划任务尝试数。",
             "mock 的回答、token 和帧引用为内置合成 fixture；耗时为实际 Python 执行耗时，不能用于推断真实模型性能。" if mode == "mock" else "真实 runner 负责实际 provider 调用与视频解码；日志保存其来源声明和请求轨迹。",
             "", f"Run IDs: {', '.join(sorted(selected_runs))}", "", "## 来源", ""]
    for item in inputs:
        lines.append(f"- `{item['path']}` — SHA256 `{item['sha256']}`")
    lines.extend(["", "## 配置与判分口径", "", "```json", json.dumps(manifests[0]["config"], ensure_ascii=False, indent=2), "```", "",
                  f"Seed supported: {manifests[0]['seed_supported']}; effective seed: {manifests[0]['config']['seed'] if manifests[0]['seed_supported'] else None}.",
                  "封闭答案：NFKC、casefold、合并空白，保留标点；仅接受显式 standard_answer/aliases。开放题标记 manual_review，不自动判正确。",
                  "准确率和证据命中率均除以全部计划尝试数；调用失败、缺失记录、待人工判分均不贡献正确数，失败也不贡献命中数。",
                  "timestamp_hit：任一预测时间点落入容差扩展的事件窗口，或预测区间与扩展窗口相交（含边界）。tIoU 使用未扩展窗口，多窗口取最大值。",
                  "资源均值仅使用已记录值，失败的已消耗资源仍计入；缺失值不是 0。token 总量仅在全部请求 usage 完整时可用；价格由用户提供，成本为估算，不代表账单。",
                  "any_data_missing_ratio 按执行记录、预测、证据、token usage、耗时中至少一个缺失统计；成本缺失单列，不纳入该合并比例。",
                  "", "## 总体", "", f"全部策略尝试 n={overall['n']}；准确率 {overall['accuracy']:.1%}；证据命中率 {overall['evidence_hit_rate']:.1%}；待人工判分 {overall['manual_review_count']}。",
                  "", "| 策略 | n | 正确 / 准确率 | 证据命中率 | 平均请求 | 平均观察帧 | 平均输入图像 | 平均耗时 s | usage n | 平均输入/输出 token | 平均估算成本 | 任一数据缺失 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for s, m in metrics.items():
        lines.append(f"| {s} | {m['n']} | {m['correct_count']} / {m['accuracy']:.1%} | {m['evidence_hit_rate']:.1%} | {display(m['mean_requests'])} | {display(m['mean_observed_frames'])} | {display(m['mean_model_input_images'])} | {display(m['mean_elapsed_seconds'])} | {m['usage_n']} | {display(m['mean_input_tokens'])} / {display(m['mean_output_tokens'])} | {display(m['mean_estimated_cost'])} | {m['any_data_missing_ratio']:.1%} |")
    lines.extend(["", "## 题型与缺失数据", "", "| 策略 | 题型 | n | 正确数 | 准确率 |", "|---|---|---:|---:|---:|"])
    for s, m in metrics.items():
        for kind, group in m["by_question_type"].items():
            lines.append(f"| {s} | {cell(kind)} | {group['n']} | {group['correct_count']} | {group['accuracy']:.1%} |")
    lines.extend(["", "| 策略 | execution | prediction | evidence | token_usage | latency | cost | 平均 tIoU（可用 n） |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for s, m in metrics.items():
        ratios = " | ".join(f"{v:.1%}" for v in m["missing_ratios"].values())
        lines.append(f"| {s} | {ratios} | {display(m['mean_tiou'])} (n={m['tiou_n']}) |")
    lines.extend(["", "## 图表", "", "![策略准确率](strategy_accuracy.png)", "", "![证据命中率](evidence_hit.png)", "", "![耗时和成本](resources_cost.png)", "", "## 完整性提示", ""])
    lines.extend([f"- {cell(note)}" for note in notes] or ["日志包含全部计划记录和完成标记。"])
    lines.extend(["", "详见 [failures.md](failures.md) 和 [metrics.json](metrics.json)。", ""])
    (archive / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    failure_lines = [f"# EgoAgent-Eval failures: {heading}", "", "## 检测与待人工复核计数", "", "| 策略 | 类别 | 来源 | n |", "|---|---|---|---:|"]
    for s, m in metrics.items():
        for code, count in sorted(m["failure_counts"].items()):
            source = "requires_human" if code == "manual_review" else "automatic"
            failure_lines.append(f"| {s} | {code} | {source} | {count} |")
    failure_lines.extend(["", "## 原因判定边界", "", "incorrect 是精确/别名比较失败；insufficient_evidence 仅表示未提供时间证据；invalid_json 包括答案 JSON schema 无效。",
                          "tool_error、budget_exceeded、runner_error、protocol_error 与 missing_record 来自执行轨迹和异常。一个样本可有多个错误，计数不可直接相加作为失败样本数。",
                          "这些自动检测结果不证明模型因果原因。遮挡、视觉识别失败、问题歧义、标注缺漏和证据是否充分均需人工判断；本报告未自动归因或填写人工结论。",
                          "manual_review 的答案不自动算作正确，后续人工复核应另存有审计信息的判分文件。", "", "## 逐条记录（包括人工待复核）", "",
                          "| run_id | task_id | 策略 | 来源 | 类别 | 详情 | 预测 | 人工结论 |", "|---|---|---|---|---|---|---|---|"])
    for r in records:
        for f in r["failures"]:
            prediction = (r.get("prediction") or {}).get("answer", "N/A")
            failure_lines.append("| " + " | ".join(cell(v) for v in (r["run_id"], r["task_id"], r["strategy"], f["source"], f["code"], f["detail"], prediction, "未评审")) + " |")
    (archive / "failures.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    for name in ("summary.md", "failures.md", "metrics.json", "strategy_accuracy.png", "evidence_hit.png", "resources_cost.png"):
        shutil.copyfile(archive / name, root / name)
    return root / "summary.md", archive, saved
