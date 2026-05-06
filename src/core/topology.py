import numpy as np
from itertools import combinations
from topology import projection_phi, LogicalTopology, calculate_similarity

def cacp_inference(x, alpha, gamma, q_hat, model):
    """
    Performs Causal-Adaptive Conformal Prediction inference.
    
    Args:
        x: Input query.
        alpha: Significance level (error budget).
        gamma: Difficulty scaling parameter.
        q_hat: Calibrated quantile from the calibration set.
        model: The underlying LLM or reasoning agent.
    """
    # 1. Path Sampling (Algorithm Line 1-2)
    topology_set = []
    reasoning_paths = model.sample_paths(x, K=10) # Sample K paths
    
    # 2. Topology Construction (Algorithm Line 3-6)
    for r_i in reasoning_paths:
        # Map path to components (V_i, E_i) via projection phi (Eq. 1)
        V_i, E_i = projection_phi(r_i)
        # Construct Logical Topology G_i
        G_i = LogicalTopology(V_i, E_i)
        topology_set.append(G_i)
        
    # 3. MPSC Score Calculation (Algorithm Line 7)
    # Uses the combination-based average: 1 / C_K^2 * sum(Sim(G_i, G_j))
    K = len(topology_set)
    if K < 2:
        s_x = 1.0 # Default stability for single paths
    else:
        sim_sum = sum(calculate_similarity(G_i, G_j) 
                      for G_i, G_j in combinations(topology_set, 2))
        combinations_count = K * (K - 1) / 2 # Represents C_K^2
        s_x = sim_sum / combinations_count # Final MPSC score s(x)
    
    # 4. Difficulty Scaling (Algorithm Line 8-10)
    # sigma(x) = exp(gamma * (1 - s(x))) (Eq. 3)
    sigma_x = np.exp(gamma * (1 - s_x))
    
    # Obtain base non-conformity score S(x, y) = 1 - P(y|x)
    probs = model.get_probs(x) 
    S_xy = 1 - probs 
    
    # 5. Prediction Set Construction (Algorithm Line 11-13)
    # Condition: S(x, y) <= q_hat * sigma(x) (Eq. 5)
    prediction_set = []
    for y_idx, score in enumerate(S_xy):
        if score <= q_hat * sigma_x:
            prediction_set.append(y_idx)
            
    return prediction_set
