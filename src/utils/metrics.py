import numpy as np
from typing import List, Dict, Any

def compute_detailed_metrics(results: List[Dict[str, Any]], threshold: float = 0.3) -> Dict[str, float]:
    """
    Computes standard Conformal Prediction metrics and conditional robustness 
    on 'Hard Slices' (samples with low structural consistency).
    
    Args:
        results: List of dictionaries containing 'is_covered', 'set_size', and 's_tr'.
        threshold: The MPSC score threshold defining a 'Hard Slice'.
        
    Returns:
        A dictionary containing overall and conditional metrics.
    """
    # 1. Overall Metrics
    overall_cov = np.mean([r['is_covered'] for r in results])
    overall_size = np.mean([r['set_size'] for r in results])
    
    # 2. Hard Slice Analysis (The core of CACP's contribution)
    # Samples with low S_TR (Logic Topology Collapse)
    hard_samples = [r for r in results if r['s_tr'] < threshold]
    easy_samples = [r for r in results if r['s_tr'] >= threshold]
    
    hard_slice_cov = np.mean([r['is_covered'] for r in hard_samples]) if hard_samples else 0.0
    hard_slice_size = np.mean([r['set_size'] for r in hard_samples]) if hard_samples else 0.0
    
    easy_slice_cov = np.mean([r['is_covered'] for r in easy_samples]) if easy_samples else 0.0
    
    # 3. Efficiency Metric (Set Size Variance)
    # High variance in set size indicates effective adaptive scaling
    size_variance = np.var([r['set_size'] for r in results])

    return {
        "overall_cov": float(overall_cov),
        "overall_size": float(overall_size),
        "hard_slice_cov": float(hard_slice_cov),
        "hard_slice_size": float(hard_slice_size),
        "easy_slice_cov": float(easy_slice_cov),
        "size_variance": float(size_variance),
        "hard_sample_ratio": len(hard_samples) / len(results)
    }

def print_metrics_table(metrics: Dict[str, float]):
    """Helper to print a clean summary for academic reporting."""
    print("\n" + "="*40)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:<25} | {v:<10.4f}")
    print("="*40 + "\n")