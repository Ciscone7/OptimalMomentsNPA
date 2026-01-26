"""
results_manager.py - Utilities for HDF5-based simulation results

Usage:
    from results_manager import (
        check_simulation_status,
        load_results_from_config,
        load_specific_params,
        get_fixed_parameters,
        results_to_dataframe
    )
    
    # Check which simulations completed
    status = check_simulation_status("config.json")
    
    # Get fixed parameters
    fixed = get_fixed_parameters("./results/results.h5")
    
    # Load all results
    results = load_results_from_config("config.json")
    
    # Load specific parameter set
    results = load_specific_params("./results/results.h5", 
                                   {"temperature": 200, "pressure": 1.0})
"""

import json
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py not installed. Install with: pip install h5py"
    )


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_parameter_combinations(param_dict):
    """Generate all combinations of parameters."""
    keys = param_dict.keys()
    values = param_dict.values()
    
    combinations = []
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations


def params_to_string(params):
    """Convert parameters dict to a readable string."""
    return "_".join([f"{k}={v}" for k, v in sorted(params.items())])



def save_run_to_hdf5(hdf5_path: str, run_id: str, results: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Save a single run's results and parameters to the HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file (or directory).
        run_id: Unique identifier for this run (used as group name).
        results: Dictionary of results to save.
        params: Dictionary of parameters to save as attributes.
    """
    hdf5_path = Path(hdf5_path)
    if hdf5_path.is_dir():
        hdf5_path = hdf5_path / "results.h5"
        
    with h5py.File(hdf5_path, 'a') as f:
        # Create group for this run (delete if exists to overwrite)
        if run_id in f:
            del f[run_id]
        run_group = f.create_group(run_id)
        
        # Save variable parameters as attributes
        for key, value in params.items():
            run_group.attrs[f"param_{key}"] = value
            
        # Recursive helper to save dictionary contents
        def _save_recursive(group, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    subgroup = group.create_group(key)
                    _save_recursive(subgroup, value)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    group.create_dataset(key, data=value)
                elif isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                    group.attrs[key] = value
                else:
                    # Fallback for other types -> string
                    group.attrs[key] = str(value)

        _save_recursive(run_group, results)


def log_run_completion(log_path: str, params: Dict[str, Any], repetition_idx: int) -> None:
    """
    Log the completion of a run to the text log file.
    
    Args:
        log_path: Path to the log file.
        params: Dictionary of parameters for this run.
        repetition_idx: Index of the current repetition.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    param_str = params_to_string(params)
    
    with open(log_path, 'a') as f:
        f.write(f"{timestamp} | {param_str} | rep={repetition_idx} | COMPLETED\n")


def parse_log_file(log_file_path):
    """
    Parse the log file and return completion information.
    
    Returns:
        Dictionary mapping param_string -> set of completed repetition indices
    """
    if not Path(log_file_path).exists():
        return {}
    
    completions = defaultdict(set)
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if "COMPLETED" not in line:
                continue
            
            try:
                # Parse line format: timestamp | param_str | rep=X | COMPLETED
                parts = [p.strip() for p in line.split('|')]
                param_str = parts[1]
                rep_part = parts[2]
                
                # Extract repetition number
                rep_num = int(rep_part.split('=')[1])
                
                completions[param_str].add(rep_num)
            except (IndexError, ValueError):
                continue
    
    return completions


def get_fixed_parameters(hdf5_path):
    """
    Get fixed parameters from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file (or directory containing results.h5)
    
    Returns:
        Dictionary of fixed parameters
    
    Example:
        fixed = get_fixed_parameters("./results/results.h5")
        # Returns: {'num_particles': 1000, 'timestep': 0.001, ...}
    """
    # Handle both file path and directory path
    if Path(hdf5_path).is_dir():
        hdf5_path = Path(hdf5_path) / "results.h5"
    
    if not Path(hdf5_path).exists():
        return {}
    
    fixed_params = {}
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            for key, value in f.attrs.items():
                if key.startswith('fixed_param_'):
                    param_name = key.replace('fixed_param_', '')
                    fixed_params[param_name] = value
    except Exception as e:
        print(f"Warning: Could not read fixed parameters: {e}")
    
    return fixed_params


def load_run_from_hdf5(run_group):
    """
    Load a single run from HDF5 group.
    
    Args:
        run_group: h5py Group object
        
    Returns:
        Dictionary with all data from the run
    """
    result = {}
    
    # Load attributes (scalars and metadata)
    for key, value in run_group.attrs.items():
        result[key] = value
    
    # Load datasets (arrays) and subgroups (nested structures)
    def load_recursive(group):
        """Recursively load group contents."""
        data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # Nested group
                data[key] = load_recursive(item)
            else:
                # Dataset (array)
                data[key] = item[()]
        return data
    
    # Add datasets to result
    result.update(load_recursive(run_group))
    
    return result


def load_results_from_path(results_path, include_fixed=False):
    """
    Load all simulation results from HDF5 file.
    
    Args:
        results_path: Path to the results directory (containing results.h5)
        include_fixed: If True, add fixed parameters to each result dict
    
    Returns:
        List of dictionaries, one per run, with all parameters and results
    
    Example:
        results = load_results_from_path("./my_results")
        
        for run in results:
            print(f"Seed: {run['seed']}")
            print(f"Temperature: {run['param_temperature']}")
            print(f"Energy: {run['energy']}")
            print(f"Trajectory shape: {run['trajectory'].shape}")
    """
    hdf5_path = Path(results_path) / "results.h5"
    
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found at {hdf5_path}")
    
    results = []
    fixed_params = get_fixed_parameters(hdf5_path) if include_fixed else {}
    
    with h5py.File(hdf5_path, 'r') as f:
        for run_id in sorted(f.keys()):
            run_data = load_run_from_hdf5(f[run_id])
            run_data['run_id'] = run_id
            
            # Add fixed parameters if requested
            if include_fixed:
                for key, value in fixed_params.items():
                    run_data[f'fixed_{key}'] = value
            
            results.append(run_data)
    
    return results


def load_results_from_config(config_path, include_fixed=False):
    """
    Load simulation results using the original config file.
    
    Args:
        config_path: Path to the config JSON file used for the simulation
        include_fixed: If True, add fixed parameters to each result dict
    
    Returns:
        List of dictionaries with all results
    
    Example:
        results = load_results_from_config("config.json")
        
        for run in results:
            params = {k: v for k, v in run.items() if k.startswith('param_')}
            print(f"Parameters: {params}")
            print(f"Energy: {run['energy']}")
    """
    config = load_config(config_path)
    save_path = config.get("save_result_path", "./results")
    
    return load_results_from_path(save_path, include_fixed)


def load_specific_params(hdf5_path, target_params, load_arrays=True):
    """
    Load only runs matching specific parameters.
    
    Args:
        hdf5_path: Path to HDF5 file (or directory containing results.h5)
        target_params: Dict like {'temperature': 200, 'pressure': 1.0}
        load_arrays: If False, only load scalar data (faster)
    
    Returns:
        List of result dictionaries for matching runs
    
    Example:
        # Load all repetitions for one parameter set
        results = load_specific_params(
            "my_results/results.h5",
            {'temperature': 200, 'pressure': 1.0}
        )
        
        # Get energies
        energies = [r['energy'] for r in results]
        
        # Load only scalars (faster if you don't need arrays)
        results = load_specific_params(
            "my_results/results.h5",
            {'temperature': 200, 'pressure': 1.0},
            load_arrays=False
        )
    """
    # Handle both file path and directory path
    if Path(hdf5_path).is_dir():
        hdf5_path = Path(hdf5_path) / "results.h5"
    
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"HDF5 file not found at {hdf5_path}")
    
    results = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for run_id in f.keys():
            run_group = f[run_id]
            
            # Check if parameters match
            match = True
            for param_name, param_value in target_params.items():
                attr_name = f'param_{param_name}'
                if attr_name not in run_group.attrs:
                    match = False
                    break
                if run_group.attrs[attr_name] != param_value:
                    match = False
                    break
            
            if match:
                if load_arrays:
                    # Load everything
                    run_data = load_run_from_hdf5(run_group)
                else:
                    # Only load attributes (scalars)
                    run_data = dict(run_group.attrs)
                
                run_data['run_id'] = run_id
                results.append(run_data)
    
    return results


def check_simulation_status(config_path, verbose=True):
    """
    Check which simulations are completed based on config and log file.
    
    Args:
        config_path: Path to the config JSON file
        verbose: If True, print detailed status
    
    Returns:
        Dictionary with status information
    
    Example:
        status = check_simulation_status("config.json")
        
        if not status['all_completed']:
            print("Missing experiments:")
            for params_str, reps in status['missing_by_params'].items():
                print(f"  {params_str}: {reps}")
    """
    # Load config
    config = load_config(config_path)
    save_path = Path(config.get("save_result_path", "./results"))
    num_experiments = config.get("num_experiments", 10)
    parameters = config.get("parameters", {})
    fixed_parameters = config.get("fixed_parameters", {})
    
    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations(parameters)
    total_param_sets = len(param_combinations)
    total_experiments = total_param_sets * num_experiments
    
    # Parse log file
    log_file = save_path / "experiment_log.txt"
    completions = parse_log_file(log_file)
    
    # Build status information
    completed_by_params = {}
    missing_by_params = {}
    completed_experiments = 0
    
    for params in param_combinations:
        param_str = params_to_string(params)
        completed_reps = sorted(completions.get(param_str, set()))
        expected_reps = list(range(num_experiments))
        missing_reps = [r for r in expected_reps if r not in completed_reps]
        
        completed_by_params[str(params)] = completed_reps
        completed_experiments += len(completed_reps)
        
        if missing_reps:
            missing_by_params[str(params)] = missing_reps
    
    all_completed = (completed_experiments == total_experiments)
    missing_experiments = total_experiments - completed_experiments
    
    # Print status if verbose
    if verbose:
        print("=" * 70)
        print("SIMULATION STATUS CHECK")
        print("=" * 70)
        print(f"\nConfig file: {config_path}")
        print(f"Results directory: {save_path}")
        print(f"Log file: {log_file}")
        print(f"HDF5 file: {save_path / 'results.h5'}")
        
        if fixed_parameters:
            print(f"\nFixed parameters:")
            for param, value in fixed_parameters.items():
                print(f"  {param}: {value}")
        
        print(f"\nOverall Progress:")
        print(f"  Total parameter sets: {total_param_sets}")
        print(f"  Experiments per set: {num_experiments}")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Completed: {completed_experiments}")
        print(f"  Missing: {missing_experiments}")
        print(f"  Progress: {completed_experiments/total_experiments*100:.1f}%")
        
        if all_completed:
            print("\n✓ All simulations completed!")
        else:
            print("\n✗ Some simulations are missing")
            print("\nMissing experiments by parameter set:")
            print("-" * 70)
            
            for params_str, missing_reps in missing_by_params.items():
                params_dict = eval(params_str)
                print(f"\nParameters: {params_dict}")
                print(f"  Missing repetitions: {missing_reps}")
                print(f"  ({len(missing_reps)}/{num_experiments} missing)")
        
        print("=" * 70)
    
    return {
        'all_completed': all_completed,
        'total_param_sets': total_param_sets,
        'total_experiments': total_experiments,
        'completed_experiments': completed_experiments,
        'missing_experiments': missing_experiments,
        'completed_by_params': completed_by_params,
        'missing_by_params': missing_by_params,
        'fixed_parameters': fixed_parameters
    }


def results_to_dataframe(config_path, scalars_only=True, include_fixed=False):
    """
    Load results and convert to a pandas DataFrame.
    
    Args:
        config_path: Path to the config JSON file
        scalars_only: If True, only include scalar values (not arrays)
        include_fixed: If True, add fixed parameters as columns
    
    Returns:
        pandas DataFrame with results
    
    Example:
        # Load only scalars (fast, good for overview)
        df = results_to_dataframe("config.json")
        print(df.groupby('param_temperature')['energy'].mean())
        
        # Include fixed parameters as columns
        df = results_to_dataframe("config.json", include_fixed=True)
        print(df[['param_temperature', 'fixed_num_particles', 'energy']])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    results = load_results_from_config(config_path, include_fixed=include_fixed)
    
    if scalars_only:
        # Only keep scalar values
        rows = []
        for r in results:
            row = {}
            for key, value in r.items():
                if isinstance(value, (int, float, bool, str, np.integer, np.floating)):
                    row[key] = value
            rows.append(row)
        return pd.DataFrame(rows)
    else:
        # Include everything (arrays will be object type in DataFrame)
        return pd.DataFrame(results)


def get_completion_summary(config_path):
    """
    Get a brief summary of completion status.
    
    Args:
        config_path: Path to the config JSON file
    
    Returns:
        Tuple of (completed_count, total_count, all_completed_bool)
    
    Example:
        completed, total, all_done = get_completion_summary("config.json")
        print(f"{completed}/{total} experiments completed")
    """
    status = check_simulation_status(config_path, verbose=False)
    return (
        status['completed_experiments'],
        status['total_experiments'],
        status['all_completed']
    )


# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python results_manager.py <config_file>")
        print("\nExamples:")
        print("  python results_manager.py config.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not Path(config_file).exists():
        print(f"Error: Config file '{config_file}' not found!")
        sys.exit(1)
    
    # Check status
    status = check_simulation_status(config_file, verbose=True)
    
    # Show fixed parameters from HDF5
    config = load_config(config_file)
    save_path = Path(config.get("save_result_path", "./results"))
    hdf5_path = save_path / "results.h5"
    
    if hdf5_path.exists():
        fixed = get_fixed_parameters(hdf5_path)
        if fixed:
            print("\nFixed parameters from HDF5:")
            for key, value in fixed.items():
                print(f"  {key}: {value}")
    
    # Try to load and show some stats
    try:
        import pandas as pd
        df = results_to_dataframe(config_file, scalars_only=True)
        
        if not df.empty:
            print("\nDataFrame Statistics:")
            print("-" * 70)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show statistics for result columns
            result_cols = [col for col in df.columns 
                          if not col.startswith('param_') 
                          and not col.startswith('seed')
                          and not col.startswith('run_id')
                          and not col.startswith('fixed_')]
            if result_cols:
                print(f"\nResult statistics:")
                print(df[result_cols].describe())
    except ImportError:
        print("\nNote: Install pandas to see DataFrame statistics")
        print("  pip install pandas")