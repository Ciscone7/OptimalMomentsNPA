import numpy as np
import random
import math as m
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Dict, Optional, Any, Callable

def generate_binary_vectors(N: int, k: int, num_samples: int) -> np.ndarray:
    """
    Generate unique random binary vectors of length N with Hamming weight k.
    
    Parameters:
    -----------
    N : int
        Length of the binary vector.
    k : int
        Number of ones (Hamming weight).
    num_samples : int
        Number of unique samples to generate.
        
    Returns:
    --------
    np.ndarray
        Array of shape (num_samples, N) containing the binary vectors.
    """
    samples = set()
    n_choose_k = m.factorial(N) // (m.factorial(N-k) * m.factorial(k))
    
    if n_choose_k < num_samples:
        num_samples = n_choose_k
        
    while len(samples) < num_samples:
        indices = tuple(sorted(random.sample(range(N), k)))
        selection = np.zeros(N, dtype=int)
        selection[list(indices)] = 1
        samples.add(tuple(selection))
        
    return np.array(list(samples))

def acquisition_ucb(mean: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
    """
    Upper Confidence Bound acquisition function.
    
    Parameters:
    -----------
    mean : np.ndarray
        Predicted mean values.
    std : np.ndarray
        Predicted standard deviation values.
    beta : float
        Exploration-exploitation trade-off parameter.
        
    Returns:
    --------
    np.ndarray
        Acquisition values (lower is better for minimization).
    """
    return mean - beta * std

def bayesian(
    obj_func: Callable[[np.ndarray], float],
    N: int,
    k: int,
    beta: float,
    n_init: int,
    n_iter: int,
    candidates_per_iter: int = 100,
    previous_best: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform Bayesian Optimization to minimize an objective function over binary vectors.
    
    Parameters:
    -----------
    obj_func : Callable
        Objective function to minimize. Takes a binary vector and returns a scalar.
    N : int
        Dimension of the binary vector.
    k : int
        Hamming weight constraint (number of ones).
    beta : float
        UCB parameter.
    n_init : int
        Number of initial random samples.
    n_iter : int
        Number of optimization iterations.
    candidates_per_iter : int
        Number of candidate vectors to evaluate per iteration.
    previous_best : Optional[np.ndarray]
        Optional starting point for warm-starting.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, shows a progress bar.
        
    Returns:
    --------
    Dict
        Dictionary containing 'best_selection', 'best_value', and 'history'.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    X: List[np.ndarray] = []
    y: List[float] = []
    
    if previous_best is not None:
        n_init -= 10
        # Add one new 1 to previous_best
        indices_one = list(np.where(previous_best == 1)[0])
        zero_indices = list(np.where(previous_best == 0)[0])
        for _ in range(10):
            if not zero_indices:
                break
            new_index = random.choice(zero_indices)
                
            x_warm = np.array(previous_best, copy=True)
            x_warm[new_index] = 1
            
            val = obj_func(x_warm)
            X.append(x_warm)
            y.append(val)
            print("warm start")
            
    # Initial random sampling
    for _ in range(n_init):
        x = generate_binary_vectors(N, k, 1)[0]
        val = obj_func(x)
        X.append(x)
        y.append(val)

    pbar = tqdm(range(n_iter), desc="Bayesian Optimization", leave=True, disable=not verbose)
    for t in pbar:
        # Surrogate model
        model = RandomForestRegressor(random_state=seed)
        model.fit(X, y)

        # Generate candidates
        candidates = generate_binary_vectors(N, k, candidates_per_iter)

        # Vectorized Prediction
        # Get predictions from all trees for all candidates at once
        # shape: (n_estimators, n_candidates)
        all_preds = np.array([tree.predict(candidates) for tree in model.estimators_])
        
        # Calculate mean and std across trees
        mu = np.mean(all_preds, axis=0)
        sigma = np.std(all_preds, axis=0)
        
        # Calculate acquisition values
        acq_values = acquisition_ucb(mu, sigma, beta)
        
        # Select best candidate (minimum acquisition value)
        best_idx_candidate = np.argmin(acq_values)
        best_x = candidates[best_idx_candidate]
        
        # Evaluate objective function
        best_y = obj_func(best_x)

        X.append(best_x)
        y.append(best_y)
        
        current_best = np.min(y)
        pbar.set_postfix({"best_val": f"{current_best:.4f}"})


    best_idx = np.argmin(y)
    return {
        "best_selection": X[best_idx],
        "best_value": y[best_idx],
        "history": {"X": X, "y": [v for v in y]}
    }
