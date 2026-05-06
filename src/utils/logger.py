import logging
import os
import time
from datetime import datetime

class CACPLogger:
    """
    Experimental logging utility designed for the Causal-Adaptive Conformal Prediction 
    (CACP) framework.
    """
    def __init__(self, log_dir="logs", model_name="DeepSeek-R1"):
        # Ensure the target directory for log storage exists
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Define log filename with pattern: YYYYMMDD_HHMMSS_ModelName.log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f"{timestamp}_{model_name}.log")
        
        # Configure the primary logger for the research project
        self.logger = logging.getLogger("CACP_Research")
        self.logger.setLevel(logging.INFO)
        
        # Standardized log entry format for timestamp and severity level
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 1. Console Handler: Provides real-time monitoring during model inference
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # 2. File Handler: Persists experimental data for formal paper citation and review[cite: 2]
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log_header(self, config):
        """
        Records the experimental hyperparameter configurations.
        """
        self.logger.info("="*30 + " Experiment Started " + "="*30)
        for key, value in config.items():
            self.logger.info(f"Config - {key}: {value}")

    def log_inference(self, sample_id, s_x, latency):
        """
        Logs per-sample metrics during the inference phase:
        - s_x: Multi-Path Structural Consistency (MPSC) score
        - latency: Computational overhead of Logic Topology extraction (to verify the 0.98% claim)[cite: 1]
        """
        self.logger.info(
            f"Sample {sample_id} | MPSC s(x): {s_x:.4f} | "
            f"Topo-Inference Latency: {latency:.4f}s"
        )

    def log_final_results(self, coverage, avg_size):
        """
        Logs aggregated performance metrics to verify restoration of nominal coverage (1-alpha).[cite: 1, 2]
        """
        self.logger.info("="*20 + " Final Results " + "="*20)
        self.logger.info(f"Empirical Coverage: {coverage*100:.2f}%")
        self.logger.info(f"Average Prediction Set Size: {avg_size:.4f}")
        self.logger.info("="*55)
