"""Benchmark SDP solve time vs basis size.

This script times the moment-relaxation SDP solve for random subsets of an NPA basis.
It is meant to answer: "how long does it take to solve an SDP of a given size on my machine?"

We fix:
  - model (ising/heisenberg/heisenberg_j2)
  - N (number of spins)
  - boundary condition
  - a starting NPA level (start_level)
  - a final NPA level (end_level)

We then:
    - form starting_set = NPA(start_level)
    - form adding_set = NPA(end_level) minus starting_set
  - for each k, sample `num_trials` random k-subsets of the adding_set
  - solve the SDP and record timings.

Results are saved in a resumable way under (default):
    spins_sdp/results/spin_sdp_solve_benchmark/v1/<config_hash>/
    meta.json
    data.npz

Or, if you pass `--out-npz`, results are written to a single flat `.npz` file
with no hashing/run directories.

`data.npz` contains arrays keyed by:
  - run_idx (int)
  - k (int)
  - trial (int)
  - seed (int)
  - basis_size (int)
  - moment_dim (int)          # n (moment matrix is n x n; PSD block is 2n x 2n)
  - n_moments (int)           # number of moment variables (len(rep.labels))
  - psd_dim (int)             # 2 * moment_dim
  - objective_value (float)
  - status (str)
  - t_compile_s (float)
  - t_build_s (float)
  - t_solve_s (float)
  - t_total_s (float)

Examples:
    # Quick local benchmark (small)
    python -m spins_sdp.scripts.sdp_benchmark \
        --model heisenberg --N 4 --boundary periodic \
        --start-level 1 --end-level 2 \
        --k-max 16 --k-step 2 --num-trials 3 \
        --solver MOSEK --resume

    # Use SCS if you don't have MOSEK available
    python -m spins_sdp.scripts.sdp_benchmark \
        --model ising --N 4 --boundary open \
        --J 1 --h 0.7 --k 0.3 \
        --start-level 1 --end-level 2 \
        --ks 0 2 4 6 8 --num-trials 2 \
        --solver SCS --resume
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore

    def _tqdm(iterable, *, total=None, desc: str = "", disable: bool = False):
        return tqdm(iterable, total=total, desc=desc, disable=disable)

except Exception:  # pragma: no cover

    def _tqdm(iterable, *, total=None, desc: str = "", disable: bool = False):
        return iterable

from spins_sdp.pauli import compile_moment_matrix_rep
from spins_sdp.sdp import build_sdp_from_rep
from spins_sdp.scripts._artifact_io import atomic_save_npz, config_hash, upsert_meta_json, utc_now_iso
from spins_sdp.scripts._common import build_npa_basis_sets, hamiltonian_dict_fn, model_params_from_args


SCHEMA_VERSION = 1
ARTIFACT_NAME = "spin_sdp_solve_benchmark"


def _parse_ks_from_args(args: argparse.Namespace, *, adding_set_size: int) -> List[int]:
    if args.ks is not None and len(args.ks) > 0:
        ks = [int(x) for x in args.ks]
    else:
        if args.k_max is None:
            raise SystemExit("Provide either --ks or --k-max")
        k_min = int(args.k_min)
        k_max = int(args.k_max)
        k_step = int(args.k_step)
        if k_step <= 0:
            raise SystemExit("--k-step must be >= 1")
        if k_min < 0:
            raise SystemExit("--k-min must be >= 0")
        if k_max < k_min:
            raise SystemExit("--k-max must be >= --k-min")
        ks = list(range(k_min, k_max + 1, k_step))

    ks = sorted(set(ks))
    if any(k < 0 for k in ks):
        raise SystemExit("All k must be >= 0")
    if any(k > adding_set_size for k in ks):
        raise SystemExit(f"Some k exceed adding set size L={adding_set_size}: {ks}")
    return ks


def _solve_one(
    *,
    basis_words: List[Any],
    operator: Any,
    solver: str,
    mosek_tol: float,
    solver_opts: Optional[Dict[str, Any]],
    verbose: bool,
) -> Tuple[float, str, int, int, float, float, float, float]:
    """Solve one SDP and return (value, status, moment_dim, n_moments, t_compile, t_build, t_solve, t_total)."""

    t0 = time.perf_counter()

    t_compile0 = time.perf_counter()
    rep = compile_moment_matrix_rep(basis_words)
    t_compile = time.perf_counter() - t_compile0

    t_build0 = time.perf_counter()
    sdp = build_sdp_from_rep(rep, operator, sense="min")
    t_build = time.perf_counter() - t_build0

    default_solver_opts: Dict[str, Any] = {}
    if solver.upper() == "MOSEK":
        default_solver_opts = {
            "mosek_params": {
                "MSK_IPAR_NUM_THREADS": 0,  # let MOSEK pick
                # "MSK_DPAR_OPTIMIZER_MAX_TIME": 5.0,  # cheap pass (seconds)
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-4,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS":   1e-4,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS":   1e-4,
            }
            
            # "mosek_params": {
            #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": mosek_tol,
            #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": mosek_tol,
            #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": mosek_tol,
            # }
        }

    merged_solver_opts = dict(default_solver_opts)
    if solver_opts:
        merged_solver_opts.update(solver_opts)

    t_solve0 = time.perf_counter()
    sdp.problem.solve(solver=solver, verbose=verbose, **merged_solver_opts)
    t_solve = time.perf_counter() - t_solve0

    t_total = time.perf_counter() - t0

    value = float(sdp.problem.value)
    status = str(sdp.problem.status)
    moment_dim = int(rep.label_idx.shape[0])
    n_moments = int(len(rep.labels))

    return value, status, moment_dim, n_moments, float(t_compile), float(t_build), float(t_solve), float(t_total)


def compute_and_save(
    *,
    N: int,
    model_name: str,
    model_params: Dict[str, float],
    boundary: str,
    start_level: int,
    end_level: int,
    k_values: List[int],
    num_trials: int,
    base_seed: int,
    solver: str,
    mosek_tol: float,
    solver_opts_json: Optional[str],
    out_root: Path,
    out_npz: Optional[Path],
    resume: bool,
    force: bool,
    append_trials: bool = False,
    verbose: bool,
) -> Path:
    solver_opts: Optional[Dict[str, Any]] = None
    if solver_opts_json:
        try:
            parsed = json.loads(solver_opts_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--solver-opts-json must be valid JSON: {e}")
        if not isinstance(parsed, dict):
            raise SystemExit("--solver-opts-json must decode to a JSON object (dict)")
        solver_opts = parsed

    if start_level <= 0 or end_level <= 0:
        raise SystemExit("--start-level and --end-level must be >= 1")
    if end_level < start_level:
        raise SystemExit("--end-level must be >= --start-level")
    if num_trials <= 0:
        raise SystemExit("--num-trials must be >= 1")

    starting_set, adding_set, final_set = build_npa_basis_sets(N=N, start_level=start_level, end_level=end_level)
    L = len(adding_set)

    k_values = [int(k) for k in k_values]
    if any(k < 0 or k > L for k in k_values):
        raise SystemExit(f"All k must satisfy 0 <= k <= L={L}")

    H_dict_fn = hamiltonian_dict_fn(model_name)
    operator = H_dict_fn(N=N, boundary=boundary, **model_params)

    config: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT_NAME,
        "model": model_name,
        "model_params": dict(model_params),
        "N": int(N),
        "boundary": boundary,
        "start_level": int(start_level),
        "end_level": int(end_level),
        "method": "random_k_subset",
        "num_trials": int(num_trials),
        "base_seed": int(base_seed),
        "solver": solver,
        "mosek_tol": float(mosek_tol),
        "solver_opts_json": solver_opts_json,
    }

    # Output selection:
    # - default: hashed artifact directory (resumable, includes meta.json)
    # - out_npz: a single flat .npz file (no hashing/run dir)
    if out_npz is not None:
        data_path = Path(out_npz)
        run_dir = data_path.parent
        meta_path: Optional[Path] = None
        cfg_hash: Optional[str] = None
    else:
        cfg_hash = config_hash(config)
        run_dir = out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}" / cfg_hash
        meta_path = run_dir / "meta.json"
        data_path = run_dir / "data.npz"

    run_dir.mkdir(parents=True, exist_ok=True)

    if force:
        # If force is set, we intentionally overwrite the run.
        # data.npz (and meta.json, when present) will be re-written.
        pass

    existing: Dict[str, np.ndarray] = {}
    completed: Set[Tuple[int, int]] = set()
    if resume and data_path.exists() and not force:
        existing = dict(np.load(data_path, allow_pickle=False))
        if "k" in existing and "trial" in existing:
            completed = set(zip(existing["k"].astype(int).tolist(), existing["trial"].astype(int).tolist()))

    # Jobs
    if append_trials and existing and not force:
        # Append mode: add `num_trials` *new* trials for each requested k.
        # Trial indices continue from the current max trial per k.
        k_existing = existing.get("k")
        trial_existing = existing.get("trial")
        max_trial_by_k: Dict[int, int] = {}
        if k_existing is not None and trial_existing is not None and len(k_existing) == len(trial_existing):
            for kk, tt in zip(k_existing.astype(int).tolist(), trial_existing.astype(int).tolist()):
                prev = max_trial_by_k.get(int(kk), -1)
                if int(tt) > prev:
                    max_trial_by_k[int(kk)] = int(tt)

        all_jobs: List[Tuple[int, int]] = []
        for kk in k_values:
            start_t = max_trial_by_k.get(int(kk), -1) + 1
            all_jobs.extend([(int(kk), int(t)) for t in range(start_t, start_t + int(num_trials))])

        # In append mode, jobs should be new by construction, but filter just in case.
        missing_jobs = [(k, t) for (k, t) in all_jobs if (k, t) not in completed]
    else:
        # Target mode: ensure we have trials 0..num_trials-1 for each requested k.
        all_jobs = [(k, t) for k in k_values for t in range(num_trials)]
        missing_jobs = [(k, t) for (k, t) in all_jobs if (k, t) not in completed]

    if verbose:
        print(f"Model: {model_name}, N={N}, boundary={boundary}")
        print(f"Starting set size: {len(starting_set)}, Adding set size: {L}, Final set size: {len(final_set)}")
        print(f"Total jobs: {len(all_jobs)}, Already done: {len(completed)}, To compute: {len(missing_jobs)}")

    if not missing_jobs and data_path.exists() and not force:
        if verbose:
            print("All jobs already completed. Nothing to do.")
        return data_path if out_npz is not None else run_dir

    # Initialize arrays
    if existing and not force:
        run_idx_list = list(existing.get("run_idx", []))
        N_list = list(existing.get("N", []))
        k_list = list(existing.get("k", []))
        trial_list = list(existing.get("trial", []))
        seed_list = list(existing.get("seed", []))
        basis_size_list = list(existing.get("basis_size", []))
        moment_dim_list = list(existing.get("moment_dim", []))
        n_moments_list = list(existing.get("n_moments", []))
        psd_dim_list = list(existing.get("psd_dim", []))
        obj_list = list(existing.get("objective_value", []))
        status_list = list(existing.get("status", []))
        t_compile_list = list(existing.get("t_compile_s", []))
        t_build_list = list(existing.get("t_build_s", []))
        t_solve_list = list(existing.get("t_solve_s", []))
        t_total_list = list(existing.get("t_total_s", []))

        # Back-compat: older flat files may not have stored per-row N.
        if not N_list:
            n_existing = len(k_list)
            N_list = [int(N)] * int(n_existing)

        next_run_idx = int(max(run_idx_list) + 1) if run_idx_list else 0
    else:
        run_idx_list = []
        N_list = []
        k_list = []
        trial_list = []
        seed_list = []
        basis_size_list = []
        moment_dim_list = []
        n_moments_list = []
        psd_dim_list = []
        obj_list = []
        status_list = []
        t_compile_list = []
        t_build_list = []
        t_solve_list = []
        t_total_list = []
        next_run_idx = 0

    job_iter = _tqdm(
        missing_jobs,
        total=len(missing_jobs),
        desc="SDP benchmark"
    )

    for k, trial in job_iter:
        # deterministic seed per job (independent of execution order)
        job_seed = int(base_seed + 1_000_000 * int(k) + int(trial))

        if k == 0:
            chosen = []
        else:
            # Use a per-job RNG derived from job_seed to make resume/order irrelevant.
            job_rng = np.random.default_rng(job_seed)
            idxs = job_rng.choice(L, size=int(k), replace=False)
            chosen = [adding_set[int(i)] for i in idxs]

        basis_words = list(starting_set) + chosen

        value, status, moment_dim, n_moments, t_compile, t_build, t_solve, t_total = _solve_one(
            basis_words=basis_words,
            operator=operator,
            solver=solver,
            mosek_tol=mosek_tol,
            solver_opts=solver_opts,
            verbose=False,
        )

        run_idx_list.append(next_run_idx)
        N_list.append(int(N))
        k_list.append(int(k))
        trial_list.append(int(trial))
        seed_list.append(job_seed)
        basis_size_list.append(int(len(basis_words)))
        moment_dim_list.append(int(moment_dim))
        n_moments_list.append(int(n_moments))
        psd_dim_list.append(int(2 * moment_dim))
        obj_list.append(float(value))
        status_list.append(str(status))
        t_compile_list.append(float(t_compile))
        t_build_list.append(float(t_build))
        t_solve_list.append(float(t_solve))
        t_total_list.append(float(t_total))
        next_run_idx += 1

        if hasattr(job_iter, "set_postfix"):
            job_iter.set_postfix(
                {
                    "k": int(k),
                    "trial": int(trial),
                    "basis": int(len(basis_words)),
                    "t_total_s": f"{t_total:.3f}",
                    "status": str(status),
                },
                refresh=False,
            )

        if verbose:
            print(
                f"Done k={int(k)} trial={int(trial)}: basis_size={int(len(basis_words))}, "
                f"moment_dim={int(moment_dim)}, n_moments={int(n_moments)}, "
                f"obj={value:.6f}, status={status}, "
                f"t_compile={t_compile:.3f}s, t_build={t_build:.3f}s, t_solve={t_solve:.3f}s, t_total={t_total:.3f}s"
            )

    # Save arrays
    atomic_save_npz(
        data_path,
        run_idx=np.asarray(run_idx_list, dtype=np.int64),
        N=np.asarray(N_list, dtype=np.int32),
        k=np.asarray(k_list, dtype=np.int32),
        trial=np.asarray(trial_list, dtype=np.int32),
        seed=np.asarray(seed_list, dtype=np.int64),
        basis_size=np.asarray(basis_size_list, dtype=np.int64),
        moment_dim=np.asarray(moment_dim_list, dtype=np.int64),
        n_moments=np.asarray(n_moments_list, dtype=np.int64),
        psd_dim=np.asarray(psd_dim_list, dtype=np.int64),
        objective_value=np.asarray(obj_list, dtype=np.float64),
        status=np.asarray(status_list, dtype="U32"),
        t_compile_s=np.asarray(t_compile_list, dtype=np.float64),
        t_build_s=np.asarray(t_build_list, dtype=np.float64),
        t_solve_s=np.asarray(t_solve_list, dtype=np.float64),
        t_total_s=np.asarray(t_total_list, dtype=np.float64),
    )

    if meta_path is not None:
        meta: Dict[str, Any] = {
            **config,
            "config_hash": cfg_hash,
            "k_values_requested": [int(x) for x in k_values],
            "num_trials": int(num_trials),
            "L": int(L),
            "starting_set_size": int(len(starting_set)),
            "final_set_size": int(len(final_set)),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "python": sys.version,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
        }

        upsert_meta_json(meta_path, meta)

    # Return the most useful path to callers.
    # - hashed mode: the run directory
    # - flat mode: the .npz file path
    return data_path if out_npz is not None else run_dir


def compute_fixed_k_across_N_and_save(
    *,
    N_values: Iterable[int],
    fixed_k: int,
    model_name: str,
    model_params: Dict[str, float],
    start_level: int,
    end_level: int,
    num_trials: int,
    mosek_tol: float = 1e-9,
    out_root: Path,
    out_npz: Optional[Path],
    verbose: bool = True,
) -> Path:
    """Benchmark one fixed k across multiple system sizes.

    Opinionated convenience wrapper around `compute_and_save()`.
    
    Args:
        N_values: Iterable of N values to run.
        fixed_k: The subset size k to benchmark for each N.
        mosek_tol: MOSEK feasibility/gap tolerance (passed to `compute_and_save`).

    Returns:
        Path to the output `.npz` (flat mode) or the version directory
        `out_root/ARTIFACT_NAME/vSCHEMA_VERSION` (hashed mode).
    """

    Ns = [int(N) for N in N_values]
    if not Ns:
        raise ValueError("N_values must be non-empty")
    if int(fixed_k) < 0:
        raise ValueError("fixed_k must be >= 0")

    # Validate k against each N's adding-set size, so we can provide a clear
    # message (and optionally skip) instead of failing mid-run.
    valid_Ns: List[int] = []
    skipped: List[Tuple[int, int]] = []  # (N, L)
    for N in Ns:
        _starting_set, adding_set, _final_set = build_npa_basis_sets(N=N, start_level=start_level, end_level=end_level)
        L = int(len(adding_set))
        if int(fixed_k) <= L:
            valid_Ns.append(int(N))
        else:
            skipped.append((int(N), L))

    if skipped and verbose:
        msg = ", ".join([f"N={N} (L={L})" for N, L in skipped])
        print(f"Skipping invalid Ns for fixed_k={int(fixed_k)}: {msg}")

    if not valid_Ns:
        raise ValueError(
            f"No valid N values for fixed_k={int(fixed_k)} given start_level={int(start_level)} end_level={int(end_level)}."
        )

    out_path_last: Optional[Path] = None
    N_iter = _tqdm(
        valid_Ns,
        total=len(valid_Ns),
        desc=f"fixed k={int(fixed_k)}",
        disable=not bool(verbose),
    )

    for N in N_iter:
        out_path_last = compute_and_save(
            N=int(N),
            model_name=model_name,
            model_params=model_params,
            boundary="periodic",
            start_level=int(start_level),
            end_level=int(end_level),
            k_values=[int(fixed_k)],
            num_trials=int(num_trials),
            base_seed=0,
            solver="MOSEK",
            mosek_tol=float(mosek_tol),
            solver_opts_json=None,
            out_root=out_root,
            out_npz=out_npz,
            resume=True,
            force=False,
            append_trials=True,
            verbose=bool(verbose),
        )

    # In flat mode this is the .npz, in hashed mode this is a run dir.
    if out_npz is not None:
        return Path(out_npz)
    return out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}"


def compute_fixed_basis_size_across_N_and_save(
    *,
    N_values: Iterable[int],
    fixed_basis_size: int,
    model_name: str,
    model_params: Dict[str, float],
    start_level: int,
    end_level: int,
    num_trials: int,
    mosek_tol: float = 1e-9,
    out_root: Path,
    out_npz: Optional[Path],
    verbose: bool = True,
) -> Path:
    """Benchmark a fixed total basis size (aka L) across multiple system sizes.

    The SDP size is controlled most directly by the *total basis size* (stored as
    `basis_size` per row in the benchmark `.npz`). In this benchmark setup:

        basis_size = len(starting_set) + k

    so for each N we choose:

        k(N) = fixed_basis_size - len(starting_set(N))

    and skip N values where that k is invalid.

    This is an opinionated convenience wrapper around `compute_and_save()`.

    Returns:
        Path to the output `.npz` (flat mode) or the version directory
        `out_root/ARTIFACT_NAME/vSCHEMA_VERSION` (hashed mode).
    """

    Ns = [int(N) for N in N_values]
    if not Ns:
        raise ValueError("N_values must be non-empty")
    if int(fixed_basis_size) <= 0:
        raise ValueError("fixed_basis_size must be > 0")

    valid_jobs: List[Tuple[int, int]] = []  # (N, k)
    skipped: List[Tuple[int, int, int]] = []  # (N, starting_set_size, adding_set_size)

    for N in Ns:
        starting_set, adding_set, _final_set = build_npa_basis_sets(
            N=int(N), start_level=int(start_level), end_level=int(end_level)
        )
        starting_size = int(len(starting_set))
        adding_size = int(len(adding_set))
        k_needed = int(fixed_basis_size) - int(starting_size)
        if 0 <= int(k_needed) <= int(adding_size):
            valid_jobs.append((int(N), int(k_needed)))
        else:
            skipped.append((int(N), int(starting_size), int(adding_size)))

    if skipped and verbose:
        msg = ", ".join(
            [f"N={N} (start={s}, adding={a})" for (N, s, a) in skipped]
        )
        print(f"Skipping invalid Ns for fixed_basis_size={int(fixed_basis_size)}: {msg}")

    if not valid_jobs:
        raise ValueError(
            f"No valid N values for fixed_basis_size={int(fixed_basis_size)} "
            f"given start_level={int(start_level)} end_level={int(end_level)}."
        )

    out_path_last: Optional[Path] = None
    job_iter = _tqdm(
        valid_jobs,
        total=len(valid_jobs),
        desc=f"fixed basis_size={int(fixed_basis_size)}",
        disable=not bool(verbose),
    )

    for N, k_needed in job_iter:
        out_path_last = compute_and_save(
            N=int(N),
            model_name=model_name,
            model_params=model_params,
            boundary="periodic",
            start_level=int(start_level),
            end_level=int(end_level),
            k_values=[int(k_needed)],
            num_trials=int(num_trials),
            base_seed=0,
            solver="MOSEK",
            mosek_tol=float(mosek_tol),
            solver_opts_json=None,
            out_root=out_root,
            out_npz=out_npz,
            resume=True,
            force=False,
            append_trials=True,
            verbose=bool(verbose),
        )

    if out_npz is not None:
        return Path(out_npz)
    return out_root / ARTIFACT_NAME / f"v{SCHEMA_VERSION}"


def compute_fixed_N_across_basis_size_and_save(
    *,
    N: int,
    basis_sizes: Iterable[int],
    model_name: str,
    model_params: Dict[str, float],
    start_level: int,
    end_level: int,
    num_trials: int,
    mosek_tol: float = 1e-9,
    out_root: Path,
    out_npz: Optional[Path],
    verbose: bool = True,
) -> Path:
    """Benchmark a fixed N while sweeping total basis size (L).

    The `.npz` stores `basis_size = len(starting_set) + k` per row.
    For a fixed N, `len(starting_set)` is constant, so sweeping basis_size is
    equivalent to sweeping k.

    This convenience wrapper accepts target `basis_sizes` and translates them
    to the corresponding `k` values for this N, skipping any invalid basis sizes.

    Returns:
        Path to the output `.npz` (flat mode) or the run directory (hashed mode).
    """

    N_int = int(N)
    basis_list = [int(L) for L in basis_sizes]
    if not basis_list:
        raise ValueError("basis_sizes must be non-empty")
    if any(L <= 0 for L in basis_list):
        raise ValueError("All basis_sizes must be > 0")

    starting_set, adding_set, _final_set = build_npa_basis_sets(
        N=N_int, start_level=int(start_level), end_level=int(end_level)
    )
    starting_size = int(len(starting_set))
    adding_size = int(len(adding_set))

    valid_ks: List[int] = []
    skipped: List[int] = []
    for L in basis_list:
        k_needed = int(L) - int(starting_size)
        if 0 <= int(k_needed) <= int(adding_size):
            valid_ks.append(int(k_needed))
        else:
            skipped.append(int(L))

    valid_ks = sorted(set(valid_ks))
    if skipped and verbose:
        min_L = int(starting_size)
        max_L = int(starting_size + adding_size)
        print(
            f"Skipping invalid basis_sizes for N={N_int}: {sorted(set(skipped))}. "
            f"Valid range is {min_L}..{max_L} (start={starting_size}, adding={adding_size})."
        )

    if not valid_ks:
        min_L = int(starting_size)
        max_L = int(starting_size + adding_size)
        raise ValueError(
            f"No valid basis_sizes for N={N_int} given start_level={int(start_level)} end_level={int(end_level)}. "
            f"Valid range is {min_L}..{max_L} (start={starting_size}, adding={adding_size})."
        )

    out_path = compute_and_save(
        N=N_int,
        model_name=model_name,
        model_params=model_params,
        boundary="periodic",
        start_level=int(start_level),
        end_level=int(end_level),
        k_values=valid_ks,
        num_trials=int(num_trials),
        base_seed=0,
        solver="MOSEK",
        mosek_tol=float(mosek_tol),
        solver_opts_json=None,
        out_root=out_root,
        out_npz=out_npz,
        resume=True,
        force=False,
        append_trials=True,
        verbose=bool(verbose),
    )
    return Path(out_npz) if out_npz is not None else out_path


def _summarize_by_k(npz_path: Path) -> None:
    data = dict(np.load(npz_path, allow_pickle=False))
    ks = data["k"].astype(int)
    t_total = data["t_total_s"].astype(float)
    basis_size = data["basis_size"].astype(int)
    moment_dim = data["moment_dim"].astype(int)

    unique_ks = sorted(set(ks.tolist()))

    print("\nSummary (per k):")
    print("k  basis_size  moment_dim  t_total_mean(s)  t_total_std(s)  t_total_min(s)  t_total_max(s)")
    for k in unique_ks:
        idx = np.where(ks == k)[0]
        bt = t_total[idx]
        bs = basis_size[idx]
        md = moment_dim[idx]
        bs_show = int(np.median(bs)) if len(bs) else 0
        md_show = int(np.median(md)) if len(md) else 0
        print(
            f"{k:2d} {bs_show:10d} {md_show:10d} "
            f"{float(np.mean(bt)) :14.4f} {float(np.std(bt)) :13.4f} "
            f"{float(np.min(bt)) :13.4f} {float(np.max(bt)) :13.4f}"
        )


def plot_total_time_vs_N_for_fixed_ks(
    npz_or_run_dir: Path,
    fixed_ks: Iterable[int],
    *,
    yscale: str = "log",
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot total runtime vs N for multiple fixed k values.

    Expects the benchmark output `.npz` to contain per-row arrays: `N`, `k`, and `t_total_s`.
    If `npz_or_run_dir` is a directory, this function loads `npz_or_run_dir/data.npz`.

    For each k in `fixed_ks`, it groups rows by N and plots mean ± std as a shaded band,
    with the mean as a line.

    Returns `(fig, ax)`.
    """

    import matplotlib.pyplot as plt

    path = Path(npz_or_run_dir)
    if path.is_dir():
        path = path / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark .npz not found: {path}")

    data = dict(np.load(path, allow_pickle=False))
    Ns_all = data["N"].astype(int)
    ks_all = data["k"].astype(int)
    t_total = data["t_total_s"].astype(float)

    ks = [int(k) for k in fixed_ks]
    if not ks:
        raise ValueError("fixed_ks must be non-empty")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(ks))))

    plotted_any = False
    for k, color in zip(ks, colors):
        mask_k = ks_all == int(k)
        if not np.any(mask_k):
            continue

        Ns_k = Ns_all[mask_k]
        t_k = t_total[mask_k]
        unique_N = np.array(sorted(set(Ns_k.tolist())), dtype=int)

        means = np.array([float(np.mean(t_k[Ns_k == N])) for N in unique_N])
        stds = np.array([float(np.std(t_k[Ns_k == N])) for N in unique_N])

        ax.fill_between(
            unique_N,
            means - stds,
            means + stds,
            alpha=0.22,
            color=color,
            linewidth=0,
        )
        ax.plot(unique_N, means, marker="o", linewidth=2, color=color, label=f"k={int(k)}")
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f"No rows found for requested ks={ks} in {path}")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("System size N")
    ax.set_ylabel("Total time (s)")
    if title is None:
        title = "SDP benchmark: total time vs N (fixed k)"
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_total_time_vs_N_for_fixed_basis_sizes(
    npz_or_run_dir: Path,
    fixed_basis_sizes: Iterable[int],
    *,
    yscale: str = "log",
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot total runtime vs N for multiple fixed basis sizes (L).

    Expects the benchmark output `.npz` to contain per-row arrays: `N`, `basis_size`,
    and `t_total_s`. If `npz_or_run_dir` is a directory, this function loads
    `npz_or_run_dir/data.npz`.

    For each basis size L, it groups rows by N and plots mean ± std as a shaded band,
    with the mean as a line.

    Returns `(fig, ax)`.
    """

    import matplotlib.pyplot as plt

    path = Path(npz_or_run_dir)
    if path.is_dir():
        path = path / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark .npz not found: {path}")

    data = dict(np.load(path, allow_pickle=False))
    Ns_all = data["N"].astype(int)
    basis_all = data["basis_size"].astype(int)
    t_total = data["t_total_s"].astype(float)

    Ls = [int(L) for L in fixed_basis_sizes]
    if not Ls:
        raise ValueError("fixed_basis_sizes must be non-empty")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(Ls))))

    plotted_any = False
    for L_fixed, color in zip(Ls, colors):
        mask_L = basis_all == int(L_fixed)
        if not np.any(mask_L):
            continue

        Ns_L = Ns_all[mask_L]
        t_L = t_total[mask_L]
        unique_N = np.array(sorted(set(Ns_L.tolist())), dtype=int)
        means = np.array([float(np.mean(t_L[Ns_L == N])) for N in unique_N])
        stds = np.array([float(np.std(t_L[Ns_L == N])) for N in unique_N])

        ax.fill_between(unique_N, means - stds, means + stds, alpha=0.22, color=color, linewidth=0)
        ax.plot(unique_N, means, marker="o", linewidth=2, color=color, label=f"L={int(L_fixed)}")
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f"No rows found for requested fixed_basis_sizes={Ls} in {path}")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("System size N")
    ax.set_ylabel("Total time (s)")
    if title is None:
        title = "SDP benchmark: total time vs N (fixed basis size L)"
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_total_time_vs_k_for_fixed_N(
    npz_or_run_dir: Path,
    N_values: Iterable[int] | int,
    *,
    yscale: str = "log",
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot total runtime vs k for one or more fixed system sizes N.

    Expects the benchmark output `.npz` to contain per-row arrays: `N`, `k`, and `t_total_s`.
    If `npz_or_run_dir` is a directory, this function loads `npz_or_run_dir/data.npz`.

    For each N in `N_values`, it uses *all available* k values present in the file for that N,
    grouping rows by k and plotting mean ± std as a shaded band, with the mean as a line.

    Returns `(fig, ax)`.
    """

    import matplotlib.pyplot as plt

    path = Path(npz_or_run_dir)
    if path.is_dir():
        path = path / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark .npz not found: {path}")

    data = dict(np.load(path, allow_pickle=False))
    Ns_all = data["N"].astype(int)
    ks_all = data["k"].astype(int)
    t_total = data["t_total_s"].astype(float)

    if isinstance(N_values, int):
        Ns_plot = [int(N_values)]
    else:
        Ns_plot = [int(N) for N in N_values]
    if not Ns_plot:
        raise ValueError("N_values must be non-empty")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(Ns_plot))))

    plotted_any = False
    for N_fixed, color in zip(Ns_plot, colors):
        mask_N = Ns_all == int(N_fixed)
        if not np.any(mask_N):
            continue

        ks_N = ks_all[mask_N]
        t_N = t_total[mask_N]
        ks_sorted = np.array(sorted(set(ks_N.tolist())), dtype=int)
        means = np.array([float(np.mean(t_N[ks_N == k])) for k in ks_sorted])
        stds = np.array([float(np.std(t_N[ks_N == k])) for k in ks_sorted])

        ax.fill_between(ks_sorted, means - stds, means + stds, alpha=0.18, color=color, linewidth=0)
        ax.plot(ks_sorted, means, marker="o", linewidth=2, color=color, label=f"N={int(N_fixed)}")
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f"No rows found for requested N_values={Ns_plot} in {path}")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Subset size k")
    ax.set_ylabel("Total time (s)")
    if title is None:
        title = "SDP benchmark: total time vs k (fixed N)"
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_total_time_vs_basis_size_for_fixed_N(
    npz_or_run_dir: Path,
    N_values: Iterable[int] | int,
    *,
    yscale: str = "log",
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot total runtime vs basis size (L) for one or more fixed system sizes N.

    Expects the benchmark output `.npz` to contain per-row arrays: `N`, `basis_size`,
    and `t_total_s`. If `npz_or_run_dir` is a directory, this function loads
    `npz_or_run_dir/data.npz`.

    For each N in `N_values`, it uses *all available* basis_size values present in the
    file for that N, grouping rows by basis_size and plotting mean ± std as a shaded
    band, with the mean as a line.

    Returns `(fig, ax)`.
    """

    import matplotlib.pyplot as plt

    path = Path(npz_or_run_dir)
    if path.is_dir():
        path = path / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark .npz not found: {path}")

    data = dict(np.load(path, allow_pickle=False))
    Ns_all = data["N"].astype(int)
    basis_all = data["basis_size"].astype(int)
    t_total = data["t_total_s"].astype(float)

    if isinstance(N_values, int):
        Ns_plot = [int(N_values)]
    else:
        Ns_plot = [int(N) for N in N_values]
    if not Ns_plot:
        raise ValueError("N_values must be non-empty")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(Ns_plot))))

    plotted_any = False
    for N_fixed, color in zip(Ns_plot, colors):
        mask_N = Ns_all == int(N_fixed)
        if not np.any(mask_N):
            continue

        basis_N = basis_all[mask_N]
        t_N = t_total[mask_N]
        basis_sorted = np.array(sorted(set(basis_N.tolist())), dtype=int)
        means = np.array([float(np.mean(t_N[basis_N == L])) for L in basis_sorted])
        stds = np.array([float(np.std(t_N[basis_N == L])) for L in basis_sorted])

        ax.fill_between(basis_sorted, means - stds, means + stds, alpha=0.18, color=color, linewidth=0)
        ax.plot(basis_sorted, means, marker="o", linewidth=2, color=color, label=f"N={int(N_fixed)}")
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f"No rows found for requested N_values={Ns_plot} in {path}")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Basis size L")
    ax.set_ylabel("Total time (s)")
    if title is None:
        title = "SDP benchmark: total time vs basis size L (fixed N)"
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_total_time_vs_basis_size_combined_Ns_with_forecast(
    npz_or_run_dir: Path,
    N_values: Iterable[int],
    *,
    fit: str = "log-linear",
    fit_L_min: Optional[int] = None,
    fit_L_max: Optional[int] = None,
    forecast_to_basis_size: Optional[int] = None,
    forecast_step: int = 1,
    yscale: str = "log",
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot total runtime vs basis size L, combining multiple N values.

    This aggregates *across the selected N values* into a single curve:
      - x-axis: basis_size (aka L)
      - y-axis: mean total time across all matching rows for all selected Ns
      - shaded band: mean ± std across those rows

    Additionally fits a trendline and can forecast beyond the observed L range.

    Args:
        npz_or_run_dir: `.npz` path or run directory containing `data.npz`.
        N_values: Which N values to include in the combined aggregation.
        fit: "log-linear" fits log(t) = a*L + b (exponential growth);
             "power-law" fits t = a*L^b (polynomial growth);
             "linear" fits t = a*L + b (linear growth).
        fit_L_min: Optional lower bound (inclusive) on L used for the regression fit.
        fit_L_max: Optional upper bound (inclusive) on L used for the regression fit.
        forecast_to_basis_size: If provided and > max(observed L), extend the trendline
            out to this L (inclusive).
        forecast_step: Step size for the trendline x-grid.

    Returns:
        (fig, ax, fit_info) where fit_info includes coefficients and R^2.
    """

    import matplotlib.pyplot as plt

    path = Path(npz_or_run_dir)
    if path.is_dir():
        path = path / "data.npz"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark .npz not found: {path}")

    Ns_plot = [int(N) for N in N_values]
    if not Ns_plot:
        raise ValueError("N_values must be non-empty")

    if int(forecast_step) <= 0:
        raise ValueError("forecast_step must be >= 1")

    data = dict(np.load(path, allow_pickle=False))
    Ns_all = data["N"].astype(int)
    basis_all = data["basis_size"].astype(int)
    t_total = data["t_total_s"].astype(float)

    mask_N = np.isin(Ns_all, np.asarray(Ns_plot, dtype=int))
    if not np.any(mask_N):
        raise RuntimeError(f"No rows found for requested N_values={Ns_plot} in {path}")

    basis_sel = basis_all[mask_N]
    t_sel = t_total[mask_N]

    basis_sorted = np.array(sorted(set(basis_sel.tolist())), dtype=int)
    means = np.array([float(np.mean(t_sel[basis_sel == L])) for L in basis_sorted])
    stds = np.array([float(np.std(t_sel[basis_sel == L])) for L in basis_sorted])

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    ax.fill_between(
        basis_sorted,
        means - stds,
        means + stds,
        alpha=0.22,
        color="#1f77b4",
        linewidth=0,
        label="mean ± std",
    )
    ax.plot(
        basis_sorted,
        means,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label=f"mean over N={Ns_plot}",
    )

    fit_info: Dict[str, Any] = {"fit": str(fit), "fit_L_min": fit_L_min, "fit_L_max": fit_L_max}

    # Trendline
    fit_mode = str(fit).strip().lower()
    x = basis_sorted.astype(float)
    y = means.astype(float)

    # Apply optional fit range bounds on L.
    mask_fit = np.ones_like(x, dtype=bool)
    if fit_L_min is not None:
        mask_fit &= x >= float(int(fit_L_min))
    if fit_L_max is not None:
        mask_fit &= x <= float(int(fit_L_max))
    if not np.any(mask_fit):
        raise ValueError(
            f"Fit range excluded all points: fit_L_min={fit_L_min}, fit_L_max={fit_L_max}. "
            f"Observed L range is {int(np.min(x))}..{int(np.max(x))}."
        )

    L_min_obs = int(np.min(x))
    L_max_obs = int(np.max(x))

    # Filter invalid points for log fits.
    if fit_mode == "log-linear":
        mask_pos = y > 0
        x_fit = x[mask_fit & mask_pos]
        y_fit = y[mask_fit & mask_pos]
        if x_fit.size < 2:
            raise RuntimeError("Not enough positive points to fit log-linear trendline")

        coeff = np.polyfit(x_fit, np.log(y_fit), 1)
        a, b = float(coeff[0]), float(coeff[1])
        yhat_log = a * x_fit + b
        ss_res = float(np.sum((np.log(y_fit) - yhat_log) ** 2))
        ss_tot = float(np.sum((np.log(y_fit) - float(np.mean(np.log(y_fit)))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        fit_info.update({"a": a, "b": b, "r2_log": r2, "equation": f"log(t)=({a:.4g})·L+({b:.4g})"})

        L_max = int(forecast_to_basis_size) if forecast_to_basis_size is not None else L_max_obs
        L_max = max(L_max, L_max_obs)
        x_line = np.arange(L_min_obs, L_max + 1, int(forecast_step), dtype=float)
        y_line = np.exp(a * x_line + b)

        # Draw trendline: solid on observed range, dashed on forecast.
        ax.plot(
            x_line[x_line <= float(L_max_obs)],
            y_line[x_line <= float(L_max_obs)],
            color="black",
            linewidth=2,
            label="trend (fit)",
        )
        if L_max > L_max_obs:
            ax.plot(
                x_line[x_line >= float(L_max_obs)],
                y_line[x_line >= float(L_max_obs)],
                color="black",
                linewidth=2,
                linestyle="--",
                label=f"forecast to L={int(L_max)}",
            )

        eq_text = f"log(t) = {a:.3g}·L + {b:.3g}\nR²(log) = {r2:.3f}"

    elif fit_mode == "linear":
        x_fit = x[mask_fit]
        y_fit = y[mask_fit]
        if x_fit.size < 2:
            raise RuntimeError("Not enough points to fit linear trendline")
        coeff = np.polyfit(x_fit, y_fit, 1)
        a, b = float(coeff[0]), float(coeff[1])
        yhat = a * x_fit + b
        ss_res = float(np.sum((y_fit - yhat) ** 2))
        ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        fit_info.update({"a": a, "b": b, "r2": r2, "equation": f"t=({a:.4g})·L+({b:.4g})"})

        L_max = int(forecast_to_basis_size) if forecast_to_basis_size is not None else L_max_obs
        L_max = max(L_max, L_max_obs)
        x_line = np.arange(L_min_obs, L_max + 1, int(forecast_step), dtype=float)
        y_line = a * x_line + b

        ax.plot(
            x_line[x_line <= float(L_max_obs)],
            y_line[x_line <= float(L_max_obs)],
            color="black",
            linewidth=2,
            label="trend (fit)",
        )
        if L_max > L_max_obs:
            ax.plot(
                x_line[x_line >= float(L_max_obs)],
                y_line[x_line >= float(L_max_obs)],
                color="black",
                linewidth=2,
                linestyle="--",
                label=f"forecast to L={int(L_max)}",
            )

        eq_text = f"t = {a:.3g}·L + {b:.3g}\nR² = {r2:.3f}"

    elif fit_mode == "power-law":
        # Fit t = a * L^b  =>  log(t) = log(a) + b*log(L)
        mask_pos = (y > 0) & (x > 0)
        x_fit = x[mask_fit & mask_pos]
        y_fit = y[mask_fit & mask_pos]
        if x_fit.size < 2:
            raise RuntimeError("Not enough positive points to fit power-law trendline")

        # Fit log(t) = c + b*log(L) where c = log(a)
        coeff = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
        b_exp, log_a = float(coeff[0]), float(coeff[1])
        a_coef = float(np.exp(log_a))

        # R² in log-log space
        yhat_log = log_a + b_exp * np.log(x_fit)
        ss_res = float(np.sum((np.log(y_fit) - yhat_log) ** 2))
        ss_tot = float(np.sum((np.log(y_fit) - float(np.mean(np.log(y_fit)))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        fit_info.update({
            "a": a_coef, 
            "b": b_exp, 
            "r2_loglog": r2, 
            "equation": f"t=({a_coef:.4g})·L^({b_exp:.4g})"
        })

        L_max = int(forecast_to_basis_size) if forecast_to_basis_size is not None else L_max_obs
        L_max = max(L_max, L_max_obs)
        x_line = np.arange(L_min_obs, L_max + 1, int(forecast_step), dtype=float)
        # Avoid x=0
        x_line = x_line[x_line > 0]
        y_line = a_coef * (x_line ** b_exp)

        ax.plot(
            x_line[x_line <= float(L_max_obs)],
            y_line[x_line <= float(L_max_obs)],
            color="black",
            linewidth=2,
            label="trend (fit)",
        )
        if L_max > L_max_obs:
            ax.plot(
                x_line[x_line >= float(L_max_obs)],
                y_line[x_line >= float(L_max_obs)],
                color="black",
                linewidth=2,
                linestyle="--",
                label=f"forecast to L={int(L_max)}",
            )

        eq_text = f"t = {a_coef:.3g}·L^{b_exp:.3g}\nR²(log-log) = {r2:.3f}"

    else:
        raise ValueError("fit must be one of: 'log-linear', 'power-law', 'linear'")

    # Annotate equation in a small textbox.
    ax.text(
        0.02,
        0.98,
        eq_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Basis size L")
    ax.set_ylabel("Total time (s)")
    if title is None:
        title = "SDP benchmark: total time vs basis size L (combined Ns)"
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax, fit_info


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark SDP solve time vs basis size")

    p.add_argument("--model", choices=["ising", "heisenberg", "heisenberg_j2"], default="heisenberg")
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--boundary", choices=["open", "periodic"], default="open")

    # Ising params
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--k", type=float, default=0.0)

    # Heisenberg J2 param
    p.add_argument("--J2", type=float, default=0.0)

    p.add_argument("--start-level", type=int, default=1)
    p.add_argument("--end-level", type=int, default=2)

    # k schedule
    p.add_argument("--ks", type=int, nargs="*", default=None, help="Explicit list of k values")
    p.add_argument("--k-min", type=int, default=0)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--k-step", type=int, default=1)

    p.add_argument("--num-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="Base seed for reproducible random subsets")

    p.add_argument("--solver", type=str, default="MOSEK")
    p.add_argument("--mosek-tol", type=float, default=1e-9)
    p.add_argument(
        "--solver-opts-json",
        type=str,
        default=None,
        help="Optional JSON dict to pass through as solver options (merged on top of defaults)",
    )

    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root directory to write results (default: spins_sdp/results)",
    )

    p.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Write results to a single flat .npz file (no hashing/run dir). Overrides --out-root.",
    )

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--append-trials",
        action="store_true",
        help="Append new trials each run instead of treating --num-trials as a fixed target per k.",
    )
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    model_params = model_params_from_args(args)

    # We need adding_set_size to validate/build ks. Compute it quickly.
    _starting_set, adding_set, _final_set = build_npa_basis_sets(
        N=int(args.N),
        start_level=int(args.start_level),
        end_level=int(args.end_level),
    )
    adding_set_size = int(len(adding_set))

    ks = _parse_ks_from_args(args, adding_set_size=adding_set_size)

    out_path = compute_and_save(
        N=int(args.N),
        model_name=args.model,
        model_params=model_params,
        boundary=args.boundary,
        start_level=int(args.start_level),
        end_level=int(args.end_level),
        k_values=ks,
        num_trials=int(args.num_trials),
        base_seed=int(args.seed),
        solver=args.solver,
        mosek_tol=float(args.mosek_tol),
        solver_opts_json=args.solver_opts_json,
        out_root=args.out_root,
        out_npz=args.out_npz,
        resume=bool(args.resume),
        force=bool(args.force),
        append_trials=bool(args.append_trials),
        verbose=bool(args.verbose),
    )

    if out_path.suffix.lower() == ".npz":
        npz_path = out_path
    else:
        npz_path = out_path / "data.npz"

    print(f"Wrote results to: {out_path}")
    _summarize_by_k(npz_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
