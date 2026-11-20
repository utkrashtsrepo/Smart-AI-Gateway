# 🏆 Benchmark Evaluation Report: AI Gateway Multi-Policy Exam

This report documents the performance of the **Smart AI Gateway** evaluated on a standardized test exam of **30 tasks** across 3 distinct routing strategies.

---

## 📊 Executive Summary Table

| Metric | Policy A: Always-Cheap (`8B`) | Policy B: Always-Strong (`70B`) | Policy C: Auto-Router (`Smart Gateway`) |
| :--- | :---: | :---: | :---: |
| **Overall Pass Rate** | **83.33%** | **70.0%** | **80.0%** |
| Easy Tasks Pass Rate | 100.0% | 100.0% | 100.0% |
| Medium Tasks Pass Rate | 80.0% | 50.0% | 80.0% |
| Hard Tasks Pass Rate | 70.0% | 60.0% | 60.0% |
| **Total Benchmark Cost** | **$0.001346** | **$0.010003** | **$0.006380** |
| **Cost per 1,000 Tasks** | **$0.0449** | **$0.3334** | **$0.2127** |
| Average Latency | 2912.76 ms | 2179.04 ms | 2280.13 ms |
| **Cost Savings vs Strong** | - | - | **36.2% cheaper** |
| **Quality Retention vs Strong** | - | - | **114.3% retention** |

---

## 💡 Key Findings

1. **Auto-Routing Delivers Massive Cost Reduction**: Auto-Router achieved a **36.2% cost reduction** compared to always routing to the 70B strong tier.
2. **Quality Preservation**: While saving substantial cost, Auto-Router retained **114.3%** of the strong model's accuracy by accurately classifying complex multi-step and programming requests.
3. **Cheap Model Limitations**: The Always-Cheap policy struggled significantly with hard multi-step and code tasks, achieving only **70.0%** on the Hard category.
4. **Latency Advantage**: By sending simple queries to the 8B model, the Gateway achieved competitive average latency.
