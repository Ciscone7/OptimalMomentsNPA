import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm

from typing import List, Dict, Optional, Any, Callable, Union



def simulated_annealing(
    obj_func: Callable[[np.ndarray], float],
    N: int,
    k: int,
    initial_guess: Optional[Union[List[int], np.ndarray]] = None,
    steps: int = 100,
    T_start: float = 1.0,
    alpha: float = 1.0,
    record_history: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generic Simulated Annealing optimizer for binary vectors with fixed Hamming weight.
    
    Parameters:
    -----------
    record_history : bool, default=False
        If True, keeps a copy of the vector at every step (High Memory Usage).
        If False, only tracks the best and current state (Low Memory Usage).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, shows a progress bar.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Initialize selection array
    selection = np.zeros(N, dtype=int)

    # --- Initialization Phase ---
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=int)
        if guess.shape[0] != N:
            raise ValueError(f"initial_guess length must be {N}.")
        
        sel_indices = list(np.where(guess == 1)[0])
        free_indices = list(np.where(guess == 0)[0])
        
        current_k = len(sel_indices)
        if current_k > k:
             raise ValueError(f"initial_guess has {current_k} ones, but k={k}.")
        
        needed = k - current_k
        if needed > 0:
            sel_indices += random.sample(free_indices, needed)
            
        selection[sel_indices] = 1
    else:
        # Robust random initialization
        selection[np.random.choice(N, k, replace=False)] = 1

    # Track selected indices for efficient swapping
    selected_indices = set(np.where(selection == 1)[0])
    
    # Initial evaluation
    current_cost = obj_func(selection)
    
    best_selection = selection.copy()
    best_cost = current_cost

    # Only initialize these lists if we intend to fill them
    all_selections = []
    all_costs = []
    
    # Record initial state if requested
    if record_history:
        all_selections.append(selection.copy())
        all_costs.append(current_cost)

    # --- Optimization Loop ---
    if 0 < k < N:
        pbar = tqdm(range(steps), desc="SA Optimization", leave=True, disable=not verbose)
        for step in pbar:
            temperature = T_start * (alpha ** step)

            # Propose Swap
            # Note: converting set to list is O(k), acceptable for moderate k
            out_idx = random.choice(list(selected_indices))
            in_idx = random.choice(list(set(range(N)) - selected_indices))

            # Apply Swap
            selection[out_idx], selection[in_idx] = 0, 1
            selected_indices.remove(out_idx)
            selected_indices.add(in_idx)

            # Evaluate
            new_cost = obj_func(selection)

            # Acceptance Criterion
            delta = new_cost - current_cost
            
            if delta < 0:
                accept = True
            else:
                if temperature < 1e-10:
                    accept = False
                else:
                    accept = np.exp(-delta / temperature) > np.random.rand()

            if accept:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_selection = selection.copy()
                    pbar.set_postfix({"loss": f"{best_cost:.4f}"})
            else:
                # Revert Swap
                selection[out_idx], selection[in_idx] = 1, 0
                selected_indices.add(out_idx)
                selected_indices.remove(in_idx)
            
            # --- HISTORY RECORDING (Conditional) ---
            if record_history:
                all_selections.append(selection.copy())
                all_costs.append(current_cost)
                
    elif record_history:
        # Edge case (k=0 or k=N): just record the single state if requested
        all_selections.append(selection.copy())
        all_costs.append(current_cost)


    return {
    "best": {
        "value": best_cost,
        "selection": best_selection
    },
    "last": {  # <-- ADD THIS for Parallel Tempering correctness
        "value": current_cost,
        "selection": selection.copy()
    },
    "all": {
        "value": all_costs,
        "selection": all_selections
    }
    }


def parallel_tempering(
    obj_func: Callable[[np.ndarray], float],
    N: int,
    k: int,          
    num_chains: Optional[int] = None,
    num_epochs: int = 50,
    steps_per_epoch: int = 50,
    T_min: float = 0.1,
    T_max: float = 10.0,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Parallel Tempering / Replica Exchange Monte Carlo for fixed-Hamming-weight binary vectors.

    Requirements:
      - simulated_annealing must return:
          results["last"] = {"selection": x_last, "value": cost_last}
        and optionally:
          results["best"] = {"selection": x_best, "value": cost_best}

    This PT implementation:
      (1) Evolves each replica using its *last/current* state (not best).
      (2) Swaps adjacent replicas with acceptance prob:
            p = min(1, exp((beta_i - beta_j) * (E_j - E_i))).
    """
    if num_chains is None:
        num_chains = mp.cpu_count()

    # Seed only the manager RNG; each worker gets its own deterministic seed below
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Temperature ladder: chain 0 hottest, chain -1 coldest (geometric spacing)
    temperatures = np.logspace(np.log10(T_max), np.log10(T_min), num=num_chains)

    if verbose:
        print("--- PT starting ---")
        print(f"Chains: {num_chains}")
        print(f"Temperatures (hot -> cold): {[float(f'{t:.4g}') for t in temperatures]}")

    # Initialize replicas uniformly on S_N^k
    replicas_x = []
    for _ in range(num_chains):
        x = np.zeros(N, dtype=int)
        x[np.random.choice(N, k, replace=False)] = 1
        replicas_x.append(x)

    # Initial costs (serial; can be parallelized if you want)
    replicas_cost = [obj_func(x) for x in replicas_x]

    # Track best found across all time (optimization metric)
    global_best_cost = float("inf")
    global_best_x = None

    with mp.Pool(processes=num_chains) as pool:
        pbar = tqdm(range(num_epochs), desc="Parallel Tempering", leave=True, disable=not verbose)

        for epoch in pbar:
            # --- Phase 1: run short constant-temperature SA bursts in parallel ---
            tasks = []
            for i in range(num_chains):
                tasks.append((
                    obj_func,
                    N,
                    k,
                    replicas_x[i],             # initial_guess = current state
                    steps_per_epoch,
                    temperatures[i],           # T_start (constant in burst)
                    1.0,                       # alpha=1.0 => constant temperature
                    False,                     # record_history
                    None if seed is None else seed + i + epoch * num_chains,
                    False                      # verbose inside SA
                ))

            results = pool.starmap(simulated_annealing, tasks)

            # Update replicas using the LAST state (PT correctness)
            for i in range(num_chains):
                if "last" not in results[i]:
                    raise KeyError("simulated_annealing must return a 'last' field for PT.")

                last = results[i]["last"]
                replicas_x[i] = np.asarray(last["selection"], dtype=int)
                replicas_cost[i] = float(last["value"])

                # Update global best using best if present, otherwise last
                cand = results[i].get("best", last)
                cand_cost = float(cand["value"])
                if cand_cost < global_best_cost:
                    global_best_cost = cand_cost
                    global_best_x = np.asarray(cand["selection"], dtype=int).copy()
                    pbar.set_postfix({"best_loss": f"{global_best_cost:.6f}"})

            # --- Phase 2: attempt swaps between adjacent temperatures ---
            # Alternate pairing for better mixing: (0,1)(2,3)... then (1,2)(3,4)...
            start_idx = epoch % 2

            for i in range(start_idx, num_chains - 1, 2):
                j = i + 1

                Ti, Tj = temperatures[i], temperatures[j]
                beta_i, beta_j = 1.0 / Ti, 1.0 / Tj

                Ei, Ej = replicas_cost[i], replicas_cost[j]

                # Correct log acceptance ratio:
                # exponent = (beta_i - beta_j) * (E_j - E_i)
                exponent = (beta_i - beta_j) * (Ej - Ei)

                # Accept with probability min(1, exp(exponent)), computed stably
                if exponent >= 0:
                    swap_prob = 1.0
                else:
                    swap_prob = float(np.exp(exponent))

                if np.random.rand() < swap_prob:
                    replicas_x[i], replicas_x[j] = replicas_x[j], replicas_x[i]
                    replicas_cost[i], replicas_cost[j] = replicas_cost[j], replicas_cost[i]

    if verbose:
        print("--- PT finished ---")

    return {
        "best": {"value": global_best_cost, "selection": global_best_x},
        "temps": temperatures,
        "final_replicas": {
            "selection": replicas_x,
            "value": replicas_cost
        }
    }

