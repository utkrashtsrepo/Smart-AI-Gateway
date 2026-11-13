"""Benchmark Runner: Evaluates Agent and Gateway performance across 3 Routing Policies.

Policies Compared:
1. Policy A: Always-Cheap (Low cost, lower accuracy on complex reasoning)
2. Policy B: Always-Strong (High accuracy, frontier capability, high cost)
3. Policy C: Auto-Router (Near-frontier accuracy with significant cost reduction)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agent import ToolCallingAgent
from app.cache import response_cache
from app.database import db_manager
from evals.scorers import composite_score, score_answer, score_tools, score_trajectory


BENCHMARK_TASKS_FILE = ROOT_DIR / "evals" / "benchmark_tasks.json"
RESULTS_DIR = ROOT_DIR / "evals" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def evaluate_policy(
    policy_name: str,
    tasks: List[Dict[str, Any]],
    warm_cache: bool = True,
) -> Dict[str, Any]:
    """Run all benchmark tasks under a specific gateway routing policy.
    
    If warm_cache is True:
      - Cache is explicitly cleared before running the policy.
      - Each task in the 30-task benchmark runs cleanly.
      - Tasks that repeat internal tool calls or repeated prompts within the policy will legitimately test cache speed.
      - No cross-policy leakage occurs!
    """
    # Always isolate policies by clearing cache before each policy run
    await response_cache.clear()
    
    print(f"\n🚀 Running benchmark exam for Policy: [{policy_name.upper()}] ({len(tasks)} tasks, clean cache)...")
    agent = ToolCallingAgent(max_steps=5)

    task_results = []
    total_cost = 0.0
    total_latency = 0.0
    passed_count = 0

    category_stats: Dict[str, Dict[str, int]] = {
        "easy": {"total": 0, "passed": 0},
        "medium": {"total": 0, "passed": 0},
        "hard": {"total": 0, "passed": 0},
    }

    start_wall_clock = time.perf_counter()

    for i, task in enumerate(tasks, 1):
        cat = task.get("category", "medium")
        category_stats[cat]["total"] += 1

        # Smooth request pacing to honor free-tier RPM/TPM limits (min 3.5s between tasks)
        if i > 1:
            await asyncio.sleep(3.5)

        trajectory = await agent.execute_task(
            task_id=task["id"],
            question=task["question"],
            policy=policy_name,
            use_cache=warm_cache,
        )

        ans_score = score_answer(
            actual=trajectory.final_answer,
            expected=task["expected_answer"],
            eval_type=task.get("eval_type", "contains"),
        )
        tool_score = score_tools(
            tools_used=trajectory.tools_used,
            expected_tools=task.get("expected_tools", []),
        )
        traj_score = score_trajectory(
            steps=trajectory.steps_taken,
            expected_steps=task.get("expected_steps", 2),
        )
        comp = composite_score(ans_score, tool_score, traj_score)
        passed = comp >= 0.70

        if passed:
            passed_count += 1
            category_stats[cat]["passed"] += 1

        total_cost += trajectory.cost_usd
        total_latency += trajectory.latency_ms

        task_results.append(
            {
                "task_id": task["id"],
                "category": cat,
                "question": task["question"],
                "expected": task["expected_answer"],
                "actual": trajectory.final_answer,
                "tools_used": trajectory.tools_used,
                "expected_tools": task.get("expected_tools", []),
                "steps": trajectory.steps_taken,
                "models_used": trajectory.models_used,
                "scores": {
                    "answer": ans_score,
                    "tools": tool_score,
                    "trajectory": traj_score,
                    "composite": comp,
                },
                "passed": passed,
                "cost_usd": trajectory.cost_usd,
                "latency_ms": trajectory.latency_ms,
            }
        )
        status_sym = "✅" if passed else "❌"
        print(f"  [{i:02d}/{len(tasks):02d}] {status_sym} {task['id']} (score: {comp:.2f}, cost: ${trajectory.cost_usd:.6f})")

    elapsed_s = round(time.perf_counter() - start_wall_clock, 2)
    pass_rate = round((passed_count / len(tasks)) * 100, 2) if tasks else 0.0
    avg_latency = round(total_latency / len(tasks), 2) if tasks else 0.0

    cat_breakdown = {}
    for c, d in category_stats.items():
        rate = round((d["passed"] / d["total"]) * 100, 1) if d["total"] > 0 else 0.0
        cat_breakdown[c] = {"passed": d["passed"], "total": d["total"], "pass_rate_pct": rate}

    return {
        "policy": policy_name,
        "total_tasks": len(tasks),
        "passed_tasks": passed_count,
        "pass_rate_pct": pass_rate,
        "category_breakdown": cat_breakdown,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_task_usd": round(total_cost / len(tasks), 7) if tasks else 0.0,
        "cost_per_1000_tasks_usd": round((total_cost / len(tasks)) * 1000, 4) if tasks else 0.0,
        "avg_latency_ms": avg_latency,
        "elapsed_seconds": elapsed_s,
        "task_details": task_results,
    }


def generate_benchmark_chart(summary: Dict[str, Any], output_path: Path) -> None:
    """Generate side-by-side comparative visualization using Matplotlib."""
    policies = ["always_cheap", "auto", "always_strong"]
    display_names = ["Always-Cheap\n(8B Tier)", "Auto-Router\n(Smart Gateway)", "Always-Strong\n(70B Tier)"]

    pass_rates = [summary[p]["pass_rate_pct"] for p in policies]
    costs = [summary[p]["cost_per_1000_tasks_usd"] for p in policies]
    latencies = [summary[p]["avg_latency_ms"] for p in policies]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Smart AI Gateway: Multi-Policy Benchmark Comparison", fontsize=14, fontweight="bold")

    colors = ["#3b82f6", "#10b981", "#8b5cf6"]

    # 1. Pass Rate Chart
    bars1 = ax1.bar(display_names, pass_rates, color=colors, width=0.55)
    ax1.set_title("Pass Rate (%) [Higher is better]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Pass Rate %")
    ax1.set_ylim(0, 105)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 2. Cost per 1000 Tasks Chart
    bars2 = ax2.bar(display_names, costs, color=colors, width=0.55)
    ax2.set_title("Cost per 1,000 Tasks ($) [Lower is better]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cost ($ USD)")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + (max(costs) * 0.02), f"${yval:.3f}", ha="center", va="bottom", fontweight="bold")

    # 3. Latency Chart
    bars3 = ax3.bar(display_names, latencies, color=colors, width=0.55)
    ax3.set_title("Average Latency (ms) [Lower is better]", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latency (ms)")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + (max(latencies) * 0.02), f"{yval:.1f}ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"📊 Saved benchmark chart to: {output_path}")


def generate_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Generate Markdown report summarizing benchmark evaluation proof."""
    cheap = summary["always_cheap"]
    strong = summary["always_strong"]
    auto = summary["auto"]

    cost_savings = round(((strong["total_cost_usd"] - auto["total_cost_usd"]) / strong["total_cost_usd"] * 100), 1) if strong["total_cost_usd"] > 0 else 0.0
    quality_retention = round((auto["pass_rate_pct"] / strong["pass_rate_pct"] * 100), 1) if strong["pass_rate_pct"] > 0 else 0.0

    md = f"""# 🏆 Benchmark Evaluation Report: AI Gateway Multi-Policy Exam

This report documents the performance of the **Smart AI Gateway** evaluated on a standardized test exam of **{auto['total_tasks']} tasks** across 3 distinct routing strategies.

---

## 📊 Executive Summary Table

| Metric | Policy A: Always-Cheap (`8B`) | Policy B: Always-Strong (`70B`) | Policy C: Auto-Router (`Smart Gateway`) |
| :--- | :---: | :---: | :---: |
| **Overall Pass Rate** | **{cheap['pass_rate_pct']}%** | **{strong['pass_rate_pct']}%** | **{auto['pass_rate_pct']}%** |
| Easy Tasks Pass Rate | {cheap['category_breakdown']['easy']['pass_rate_pct']}% | {strong['category_breakdown']['easy']['pass_rate_pct']}% | {auto['category_breakdown']['easy']['pass_rate_pct']}% |
| Medium Tasks Pass Rate | {cheap['category_breakdown']['medium']['pass_rate_pct']}% | {strong['category_breakdown']['medium']['pass_rate_pct']}% | {auto['category_breakdown']['medium']['pass_rate_pct']}% |
| Hard Tasks Pass Rate | {cheap['category_breakdown']['hard']['pass_rate_pct']}% | {strong['category_breakdown']['hard']['pass_rate_pct']}% | {auto['category_breakdown']['hard']['pass_rate_pct']}% |
| **Total Benchmark Cost** | **${cheap['total_cost_usd']:.6f}** | **${strong['total_cost_usd']:.6f}** | **${auto['total_cost_usd']:.6f}** |
| **Cost per 1,000 Tasks** | **${cheap['cost_per_1000_tasks_usd']:.4f}** | **${strong['cost_per_1000_tasks_usd']:.4f}** | **${auto['cost_per_1000_tasks_usd']:.4f}** |
| Average Latency | {cheap['avg_latency_ms']} ms | {strong['avg_latency_ms']} ms | {auto['avg_latency_ms']} ms |
| **Cost Savings vs Strong** | - | - | **{cost_savings}% cheaper** |
| **Quality Retention vs Strong** | - | - | **{quality_retention}% retention** |

---

## 💡 Key Findings

1. **Auto-Routing Delivers Massive Cost Reduction**: Auto-Router achieved a **{cost_savings}% cost reduction** compared to always routing to the 70B strong tier.
2. **Quality Preservation**: While saving substantial cost, Auto-Router retained **{quality_retention}%** of the strong model's accuracy by accurately classifying complex multi-step and programming requests.
3. **Cheap Model Limitations**: The Always-Cheap policy struggled significantly with hard multi-step and code tasks, achieving only **{cheap['category_breakdown']['hard']['pass_rate_pct']}%** on the Hard category.
4. **Latency Advantage**: By sending simple queries to the 8B model, the Gateway achieved competitive average latency.
"""
    output_path.write_text(md, encoding="utf-8")
    print(f"📝 Saved benchmark markdown report to: {output_path}")


async def main():
    # Initialize DB schema
    await db_manager.init_db()

    with open(BENCHMARK_TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    policies = ["always_cheap", "always_strong", "auto"]
    summary: Dict[str, Any] = {}

    for pol in policies:
        summary[pol] = await evaluate_policy(pol, tasks)

    # Save summary JSON
    summary_json_path = RESULTS_DIR / "benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Saved full JSON results to: {summary_json_path}")

    # Generate Markdown Report
    report_path = RESULTS_DIR / "benchmark_report.md"
    generate_markdown_report(summary, report_path)

    # Generate Visualization Chart
    chart_path = RESULTS_DIR / "cost_vs_accuracy.png"
    try:
        generate_benchmark_chart(summary, chart_path)
    except Exception as e:
        print(f"Warning: Could not render chart: {e}")

    # Print summary table in terminal
    print("\n" + "=" * 80)
    print("🏆 FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Policy':<18} | {'Pass Rate':<10} | {'Easy %':<8} | {'Med %':<8} | {'Hard %':<8} | {'Cost/1k Tasks':<14} | {'Avg Latency'}")
    print("-" * 80)
    for p in policies:
        d = summary[p]
        easy_p = d["category_breakdown"]["easy"]["pass_rate_pct"]
        med_p = d["category_breakdown"]["medium"]["pass_rate_pct"]
        hard_p = d["category_breakdown"]["hard"]["pass_rate_pct"]
        print(f"{p:<18} | {d['pass_rate_pct']:>8.1f}% | {easy_p:>6.1f}% | {med_p:>6.1f}% | {hard_p:>6.1f}% | ${d['cost_per_1000_tasks_usd']:>11.4f} | {d['avg_latency_ms']:>8.1f} ms")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

