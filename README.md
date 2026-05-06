

# CACP: Causal-Adaptive Conformal Prediction for LLMs

This repository contains the core implementation of the **Causal-Adaptive Conformal Prediction (CACP)** framework, designed to mitigate "confidence hallucinations" in Large Language Models (LLMs) during complex causal reasoning tasks.

---

## 🌟 Key Innovations

*   **Logical Topology Extraction**: Maps raw reasoning paths $r_i$ to a formal **Logical Topology** $G_i = (V_i, E_i)$ via a projection function $\phi$ based on Predicate-Argument Structures (PAS). This process grounds abstract text into a measurable mathematical entity.
*   **MPSC Metric**: The **Multi-Path Structural Consistency** score $s(x)$ quantifies the stability of logical transitions across $K$ sampled paths. It is defined using a combination-based average to ensure non-redundant structural comparison:
    $$s(x) = \frac{1}{C_K^2} \sum_{1 \le i < j \le K} \text{Sim}(G_i, G_j)$$
*   **Adaptive Scaling**: Dynamically adjusts the statistical prediction boundaries using a difficulty-scaling factor $\sigma(x)$ derived from the topology consistency:
    $$\sigma(x) = \exp(\gamma \cdot (1 - s(x)))$$

---

## 📊 Performance Highlights

*   **Reliability**: Successfully restored empirical coverage from **71%** to over **91%** in challenging causal reasoning subsets.
*   **Efficiency**: The logical extraction overhead is approximately **0.98%** of the total inference latency.
*   **Compatibility**: Plug-and-play support for **DeepSeek-R1**, LLaMA-3, and GPT-4o-mini without parameter updates.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLP Primitives
```bash
python -m spacy download en_core_web_sm
```

### 3. Execution
Run the evaluation pipeline with your desired significance level $\alpha$:
```bash
python main.py --dataset e-care --k_samples 5 --alpha 0.1
```

---
