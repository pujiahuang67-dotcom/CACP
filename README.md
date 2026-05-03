# CACP: Causal-Adaptive Conformal Prediction for LLMs

This repository contains the core implementation of the **Causal-Adaptive Conformal Prediction (CACP)** framework, designed to mitigate "confidence hallucinations" in Large Language Models (LLMs) during complex causal reasoning.

## 🌟 Key Innovations
- **Logical Topology Calibration (LTC)**: Extracts the "logical skeleton" of reasoning paths using Predicate-Argument Structures (PAS).
- **MPSC Metric**: A structural consistency score that outperforms traditional semantic metrics in detecting "Topological Collapse".
- **Adaptive Scaling**: Dynamically adjusts statistical boundaries using a difficulty-scaling factor:
  $$\sigma(x) = \exp(\gamma \cdot (1 - STR(x)))$$

## 📊 Performance Highlights
- **Reliability**: Successfully restored empirical coverage from **71%** to over **91%** in difficult causal reasoning subsets.
- **Efficiency**: The logical extraction overhead is only **0.98%** of the total inference latency.
- **Compatibility**: Plug-and-play support for DeepSeek-R1, LLaMA-3, and GPT-4o-mini without parameter updates.

## 🚀 Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download SpaCy model: `python -m spacy download en_core_web_sm`


