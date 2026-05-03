import argparse
import os
import json
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm

# Import custom modules from the src package
from src.core.topology import MPSCEstimator
from src.core.calibration import AdaptiveConformalPredictor
from src.data.loader import CausalDataLoader
from src.utils.metrics import compute_detailed_metrics

def setup_logging():
    """Set up professional logging for experiment tracking."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger("CACP-Eval")

def main(args):
    logger = setup_logging()
    logger.info(f"Initializing CACP Pipeline for dataset: {args.dataset}")

    # 1. Initialization
    # We use Logic Topology instead of Skeletons for academic precision
    estimator = MPSCEstimator(similarity_metric=args.sim_metric)
    predictor = AdaptiveConformalPredictor(alpha=args.alpha, gamma=args.gamma)
    data_loader = CausalDataLoader(dataset_name=args.dataset)

    # 2. Data Loading (Stage 1 results)
    # Load pre-computed LLM outputs: Softmax probabilities and reasoning paths
    logger.info("Stage 1: Loading multi-path reasoning samples and base scores...")
    data_bundle = data_loader.load_inference_results(k_paths=args.k_samples)
    
    # Unpack data
    base_scores = data_bundle['base_scores']  # S(x,y) = 1 - P(y|x)
    reasoning_paths = data_bundle['paths']    # K paths per sample
    labels = data_bundle['labels']
    
    # 3. MPSC Estimation (Stage 2)
    # Calculate Multi-Path Structural Consistency (S_TR)
    logger.info("Stage 2: Quantifying Topological Reliability (MPSC)...")
    s_tr_scores = []
    for paths in tqdm(reasoning_paths, desc="Estimating Topology Consistency"):
        s_tr = estimator.calculate_mpsc(paths)
        s_tr_scores.append(s_tr)
    s_tr_array = np.array(s_tr_scores)

    # 4. Data Partitioning
    # Standard split for Conformal Prediction: Calibration set and Test set
    num_samples = len(labels)
    indices = np.arange(num_samples)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    
    cal_split = int(num_samples * args.cal_ratio)
    cal_idx = indices[:cal_split]
    test_idx = indices[cal_split:]

    # 5. Adaptive Calibration (Stage 3)
    # Normalize scores and find the safe quantile threshold (q_hat)
    logger.info("Stage 3: Normalizing scores and calibrating on held-out set...")
    cal_true_scores = base_scores[cal_idx, labels[cal_idx]]
    cal_s_tr = s_tr_array[cal_idx]
    
    predictor.calibrate(cal_true_scores, cal_s_tr)
    logger.info(f"Calibration successful. q_hat determined as: {predictor.q_hat:.6f}")

    # 6. Prediction and Evaluation (Stage 4)
    # Build valid prediction sets and evaluate coverage/efficiency
    logger.info("Stage 4: Generating prediction sets for test data...")
    test_results = []
    
    for i in tqdm(test_idx, desc="Evaluating Test Samples"):
        # Construct prediction set C(x) using adaptive threshold
        p_set = predictor.predict(base_scores[i], s_tr_array[i])
        
        test_results.append({
            'sample_id': int(i),
            's_tr': float(s_tr_array[i]),
            'is_covered': int(labels[i] in p_set),
            'set_size': len(p_set),
            'prediction_set': p_set.tolist()
        })

    # 7. Detailed Reporting (including Hard Slices)
    # Analyze performance on 'Topological Collapse' cases
    metrics = compute_detailed_metrics(test_results, threshold=0.3)
    
    logger.info("="*50)
    logger.info(f"OVERALL COVERAGE: {metrics['overall_cov']:.4f} (Target: {1-args.alpha})")
    logger.info(f"AVERAGE SET SIZE: {metrics['overall_size']:.4f}")
    logger.info(f"HARD SLICE COVERAGE: {metrics['hard_slice_cov']:.4f}")
    logger.info("="*50)

    # 8. Persistence
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/{args.dataset}_CACP_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({'args': vars(args), 'metrics': metrics, 'raw': test_results}, f, indent=4)
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal-Adaptive Conformal Prediction (CACP)")
    
    # Dataset and Path arguments
    parser.add_argument("--dataset", type=str, default="e-care", help="Target causal benchmark")
    parser.add_argument("--k_samples", type=int, default=5, help="Number of reasoning paths per query")
    
    # CP Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1, help="Allowed error rate (miscoverage)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Scaling intensity for S_TR")
    parser.add_argument("--cal_ratio", type=float, default=0.5, help="Calibration set proportion")
    
    # Algorithm Settings
    parser.add_argument("--sim_metric", type=str, default="jaccard", help="Graph similarity metric")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument("--save_results", action="store_true", default=True)

    args = parser.parse_args()
    main(args)