from typing import Optional, Dict, Any

from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from tools import *

from spins_sdp.basis_builder import generate_npa_basis, generate_heisenberg_paper_basis
from spins_sdp.exact_old import ising_hamiltonian


def npa_basis_size_heatmap(N: int, max_NPA_level: int) -> np.ndarray:
  """Generate a heatmap of NPA basis sizes for varying number of particles and NPA levels.

  Args:
      N (int): Maximum number of particles.
      max_NPA_level (int): Maximum NPA level.
  """

  def _basis_size(N: int, lvl: int) -> int:
    return len(generate_npa_basis(N=N, k=lvl).words)

  N_values = np.arange(1, N + 1)              # N = 1..N
  levels  = np.arange(1, max_NPA_level + 1)   # level = 1..max_NPA_level

  # sizes[level_index, N_index]
  sizes = np.array([[_basis_size(N, lvl) for N in N_values] for lvl in levels], dtype=int)

  plt.figure(figsize=(10, 6))
  im = plt.imshow(sizes, origin="lower", aspect="auto", cmap="viridis")

  cbar = plt.colorbar(im)
  cbar.set_label("NPA basis size")

  plt.xticks(np.arange(len(N_values)), N_values)
  plt.yticks(np.arange(len(levels)), levels)
  plt.xlabel("Number of particles N")
  plt.ylabel("NPA level")
  plt.title("NPA basis size vs N and NPA level")

  # annotate
  threshold = sizes.max() / 2
  for yi, lvl in enumerate(levels):
    for xi, N in enumerate(N_values):
      val = sizes[yi, xi]
      plt.text(xi, yi, str(val),
        ha="center", va="center",
        color=("white" if val > threshold else "black"),
        fontsize=8)

  plt.tight_layout()
  plt.show()


def plot_paper_heisenberg_results(
    N_values,
    boundary: str = "periodic",
    solver: str = "MOSEK",
    mosek_tol: float = 1e-9,
    per_site: bool = True,
) -> Dict:
    """
    Replicate the paper's findings for case B (Heisenberg chain).
    
    Plots exact ground state energy vs SDP lower bound using the paper's basis.
    
    Args:
        N_values: Iterable of system sizes
        boundary: "open" or "periodic" (paper uses periodic)
        solver: default "MOSEK"
        mosek_tol: MOSEK conic tolerance
        per_site: if True, plot E/N instead of E
        
    Returns:
        Dict with exact energies, SDP lower bounds, and gap
    """
    from spins_sdp import heisenberg_hamiltonian
    
    Ns = list(N_values)
    exact_energies = []
    sdp_lbs = []
    
    # Compute exact and SDP lower bound for each N
    for N in Ns:
        
        # print(f"Computing N={N}...")
        
        # Exact ground state energy
        H = heisenberg_hamiltonian(N, boundary=boundary)
        eigenvalues = H.eigenenergies()
        E0 = float(eigenvalues[0])
        exact_energies.append(E0)
        
        # print(f"  Exact ground energy: {E0:.8f}")
        
        # print(f"  Computing basis and operator for N={N}...")
        
        # SDP lower bound using paper basis
        basis = generate_heisenberg_paper_basis(N=N)
        operator = heisenberg_operator(N=N, boundary=boundary)
        
        # print(f"  Solving SDP lower bound for N={N}...")
        E_lb = solve_relaxation(
            basis=basis,
            operator=operator,
            sense="min",
            solver=solver,
            mosek_tol=mosek_tol,
            verbose=False,
        )
        sdp_lbs.append(E_lb)
        # print(f"  SDP lower bound: {E_lb:.8f}")
    
    Ns_arr = np.array(Ns, dtype=float)
    exact_energies = np.array(exact_energies)
    sdp_lbs = np.array(sdp_lbs)
    
    # Compute gap
    gap = exact_energies - sdp_lbs
    
    # Choose what to plot
    if per_site:
        exact_plot = exact_energies / Ns_arr
        lb_plot = sdp_lbs / Ns_arr
        gap_plot = gap / Ns_arr
        ylabel = "Energy per site (E/N)"
        ylabel_gap = "Gap per site (Exact - LB)/N"
    else:
        exact_plot = exact_energies
        lb_plot = sdp_lbs
        gap_plot = gap
        ylabel = "Energy (E)"
        ylabel_gap = "Gap (Exact - LB)"
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energies
    axes[0].plot(Ns, exact_plot, marker="o", label="Exact ground energy", linewidth=2)
    axes[0].plot(Ns, lb_plot, marker="s", label="SDP lower bound (paper basis)", linewidth=2)
    axes[0].set_xlabel("Number of particles N")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("Heisenberg Chain: Exact vs SDP Lower Bound")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot 2: Gap
    axes[1].plot(Ns, gap_plot, marker="^", color="red", linewidth=2)
    axes[1].set_xlabel("Number of particles N")
    axes[1].set_ylabel(ylabel_gap)
    axes[1].set_title("Tightness of Relaxation: Gap")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("=" * 80)
    print("Heisenberg Chain (Paper Case B) - Exact vs SDP Comparison")
    print("=" * 80)
    print(f"{'N':<8} {'Exact E':<15} {'SDP LB':<15} {'Gap':<15} {'Gap/N':<15}")
    print("-" * 80)
    for i, N in enumerate(Ns):
        print(f"{N:<8} {exact_energies[i]:<15.8f} {sdp_lbs[i]:<15.8f} "
              f"{gap[i]:<15.8e} {gap[i]/N:<15.8e}")
    print("=" * 80)
    
    return {
        "N": Ns_arr,
        "exact_energies": exact_energies,
        "sdp_lbs": sdp_lbs,
        "gap": gap,
    }


def plot_heisenberg_j2_vs_coupling(
    N: int,
    J2_values,
    boundary: str = "periodic",
    solver: str = "MOSEK",
    mosek_tol: float = 1e-9,
    per_site: bool = True,
) -> Dict:
    """
    Plot ground state energy vs J2 for Heisenberg chain with second-neighbor couplings (case C).
    
    Compares exact diagonalization with SDP lower bound using the appropriate basis
    for weak (J2 <= 1) and strong (J2 > 1) regimes.
    
    Args:
        N: Number of spins
        J2_values: Iterable of J2 values to evaluate
        boundary: "open" or "periodic" (paper uses periodic)
        solver: default "MOSEK"
        mosek_tol: MOSEK conic tolerance
        per_site: if True, plot E/N instead of E
        
    Returns:
        Dict with exact energies, SDP lower bounds, and gap for each J2
    """
    from spins_sdp import heisenberg_j2_hamiltonian
    from spins_sdp.basis_builder import generate_heisenberg_j2_basis_weak, generate_heisenberg_j2_basis_strong
    
    J2_values = np.array(list(J2_values))
    exact_energies = []
    sdp_lbs = []
    
    # Compute exact and SDP lower bound for each J2
    for J2 in J2_values:
        # print(f"Computing for J2 = {J2:.3f}...")
        
        # Exact ground state energy
        H = heisenberg_j2_hamiltonian(N, J2=J2, boundary=boundary)
        eigenvalues = H.eigenenergies()
        E0 = float(eigenvalues[0])
        exact_energies.append(E0)
        # print(f"  Exact: {E0:.8f}")
        
        # Choose basis depending on J2 regime
        if J2 <= 1.0:
            basis = generate_heisenberg_j2_basis_weak(N=N)
            basis_type = "weak"
        else:
            basis = generate_heisenberg_j2_basis_strong(N=N)
            basis_type = "strong"
        
        operator = heisenberg_j2_operator(N=N, J2=J2, boundary=boundary)
        
        E_lb = solve_relaxation(
            basis=basis,
            operator=operator,
            sense="min",
            solver=solver,
            mosek_tol=mosek_tol,
            verbose=False,
        )
        sdp_lbs.append(E_lb)
        # print(f"  SDP LB ({basis_type}): {E_lb:.8f}")
    
    exact_energies = np.array(exact_energies)
    sdp_lbs = np.array(sdp_lbs)
    
    # Compute gap
    gap = exact_energies - sdp_lbs
    
    # Choose what to plot
    if per_site:
        exact_plot = exact_energies / N
        lb_plot = sdp_lbs / N
        gap_plot = gap / N
        ylabel = "Energy per site (E/N)"
        ylabel_gap = "Gap per site (Exact - LB)/N"
    else:
        exact_plot = exact_energies
        lb_plot = sdp_lbs
        gap_plot = gap
        ylabel = "Energy (E)"
        ylabel_gap = "Gap (Exact - LB)"
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energies vs J2
    axes[0].plot(J2_values, exact_plot, marker="o", label="Exact ground energy", linewidth=2, markersize=8)
    axes[0].plot(J2_values, lb_plot, marker="s", label="SDP lower bound", linewidth=2, markersize=8)
    axes[0].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="J₂=1 (regime transition)")
    axes[0].set_xlabel("Second-neighbor coupling $J_2$")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"Heisenberg Chain with $J_2$: Exact vs SDP (N={N})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Gap vs J2
    axes[1].plot(J2_values, gap_plot, marker="^", color="red", linewidth=2, markersize=8)
    axes[1].axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Second-neighbor coupling $J_2$")
    axes[1].set_ylabel(ylabel_gap)
    axes[1].set_title("Tightness of Relaxation: Gap")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("=" * 100)
    print(f"Heisenberg Chain with J2 (Paper Case C) - N={N} - Exact vs SDP Comparison")
    print("=" * 100)
    print(f"{'J2':<10} {'Exact E':<15} {'SDP LB':<15} {'Gap':<15} {'Gap/N':<15} {'Regime':<15}")
    print("-" * 100)
    for i, J2 in enumerate(J2_values):
        regime = "Weak (J2<=1)" if J2 <= 1.0 else "Strong (J2>1)"
        print(f"{J2:<10.3f} {exact_energies[i]:<15.8f} {sdp_lbs[i]:<15.8f} "
              f"{gap[i]:<15.8e} {gap[i]/N:<15.8e} {regime:<15}")
    print("=" * 100)
    
    return {
        "J2": J2_values,
        "exact_energies": exact_energies,
        "sdp_lbs": sdp_lbs,
        "gap": gap,
    }


def plot_basis_size_comparison(
    N_values,
    npa_levels=[2, 3, 4],
    include_paper_basis=True,
    logy=True,
) -> Dict[str, np.ndarray]:
    """
    Compare basis sizes of different NPA levels and the paper basis across N.
    
    Args:
        N_values: Iterable of system sizes
        npa_levels: List of NPA levels to plot (default: [2, 3, 4])
        include_paper_basis: Whether to include Heisenberg paper basis
        logy: Use log scale for y-axis
        
    Returns:
        Dict with N and basis sizes for each method
    """
    Ns = np.array(list(N_values), dtype=int)
    
    result = {"N": Ns}
    
    # Compute NPA basis sizes
    for level in npa_levels:
        sizes = np.array([len(generate_npa_basis(N=N, k=level).words) for N in Ns])
        result[f"NPA_{level}"] = sizes
    
    # Compute paper basis size
    if include_paper_basis:
        paper_sizes = np.array([len(generate_heisenberg_paper_basis(N=N)) for N in Ns])
        result["paper"] = paper_sizes
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for level in npa_levels:
        plt.plot(Ns, result[f"NPA_{level}"], marker="o", label=f"NPA level {level}", linewidth=2)
    
    if include_paper_basis:
        plt.plot(Ns, result["paper"], marker="s", label="Paper basis", linewidth=2, linestyle="--")
    
    if logy:
        plt.yscale("log")
    
    plt.xlabel("Number of particles N")
    plt.ylabel("Basis size")
    
    # Force the x-axis to use integers only
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title("Basis size comparison: NPA levels vs Paper basis")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return result


def plot_exact_vs_npa_energy_vs_N(
  N_values: range,
  J: float,
  h: float,
  k: float,
  NPA_level: int,
  boundary: str = "periodic",
  solver: str = "MOSEK",
  solver_opts: Optional[Dict[str, Any]] = None,
  per_site: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Plot exact ground-state energy vs N and NPA lower bound vs N for a fixed NPA level, as well as the error at each N.

  Args:
    N_values: iterable of particle numbers (e.g. range(2, 11)).
    J,h,k: Ising parameters.
    NPA_level: fixed NPA hierarchy level for the bound.
    boundary: "open" or "periodic".
    solver: CVXPY solver name for npa_lb_energy.
    per_site: if True, plot E/N instead of E.
  Returns:
    (Ns, exact_Es, npa_LBs) as numpy arrays.
  """
  Ns = list(N_values)
  exact_Es = []
  npa_LBs = []

  for N in Ns:
    # Exact
    H = ising_hamiltonian(N, J=J, h=h, k=k, boundary=boundary)
    E0 = H.eigenstates(eigvals=1)[0]
    exact_Es.append(float(E0))

    # NPA lower bound
    E_lb = npa_lb_energy(
      J=J, h=h, k=k, N=N, NPA_level=NPA_level,
      solver=solver, boundary=boundary,
      solver_opts=solver_opts,
    )
    npa_LBs.append(float(E_lb))

  exact_Es = np.array(exact_Es, dtype=float)
  npa_LBs = np.array(npa_LBs, dtype=float)
  Ns_arr = np.array(Ns, dtype=float)

  if per_site:
    exact_plot = exact_Es / np.array(Ns, dtype=float)
    lb_plot = npa_LBs / np.array(Ns, dtype=float)
    ylabel = "Energy per spin (E/N)"
    gap = (exact_Es - npa_LBs) / Ns_arr
    ylabel_gap = "Gap per spin (Exact - LB)/N"
  else:
    exact_plot = exact_Es
    lb_plot = npa_LBs
    ylabel = "Energy (E)"
    gap = exact_Es - npa_LBs
    ylabel_gap = "Gap (Exact - LB)"

  # Plot 1: energies
  plt.figure(figsize=(8, 5))
  plt.plot(Ns, exact_plot, marker="o", label="Exact ground energy")
  plt.plot(Ns, lb_plot, marker="s", label=f"NPA lower bound (level {NPA_level})")

  # Force the x-axis to use integers only
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

  plt.xlabel("Number of particles N")
  plt.ylabel(ylabel)
  plt.title(f"Exact vs NPA bound (J={J}, h={h}, k={k}, boundary={boundary})")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.show()
  
  # Plot 2: gap
  plt.figure(figsize=(8, 4))
  plt.plot(Ns, gap, marker="^")
  plt.xlabel("Number of particles N")
  plt.yscale("log")
  plt.ylabel(ylabel_gap)
  plt.title("Tightness of the relaxation: Exact - NPA lower bound")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

  return np.array(Ns), exact_Es, npa_LBs


def plot_runtime_comparison_multi_levels(
    exact_data,
    npa_data_by_level,
    levels,
    use="t_avg",
    logy=True,
):
    """
    Plot exact runtime + multiple NPA runtimes (different levels) on one plot.
    Then plot performance gap vs the "longest" NPA dataset (largest max N / most points),
    comparing only on the overlap region with exact.

    Args:
        exact_data: dict from benchmark_exact_diagonalization
        npa_data_by_level: dict mapping level -> dict from benchmark_npa_relaxation
                           (each may have different N ranges)
        levels: iterable of NPA levels to include (order used in legend)
        use: "t_best" or "t_avg"
        logy: log scale for runtime axis
    Returns:
        dict with plotted series + chosen reference level for comparisons.
    """
    # --- exact series ---
    N_exact = np.array(exact_data["N"], dtype=int)
    t_exact = np.array(exact_data[use], dtype=float)
    ex_order = np.argsort(N_exact)
    N_exact, t_exact = N_exact[ex_order], t_exact[ex_order]

    # --- gather NPA series ---
    npa_series = {}  # lvl -> (N, t)
    for lvl in levels:
        if lvl not in npa_data_by_level:
            raise KeyError(f"Missing npa_data for level {lvl}.")
        d = npa_data_by_level[lvl]
        N = np.array(d["N"], dtype=int)
        t = np.array(d[use], dtype=float)
        order = np.argsort(N)
        npa_series[lvl] = (N[order], t[order])

    # --- choose reference: the "longest" NPA dataset ---
    # Primary: largest max N. Tie-breaker: most points.
    def ref_key(lvl):
        N, _ = npa_series[lvl]
        return (int(np.max(N)), int(N.size))

    ref_level = max(levels, key=ref_key)
    N_ref, t_ref = npa_series[ref_level]

    # ---- Plot 1: runtimes for exact + all NPA levels ----
    plt.figure(figsize=(9, 5))
    plt.plot(N_exact, t_exact, marker="o", label=f"Exact ({use})")

    for lvl in levels:
        N, t = npa_series[lvl]
        plt.plot(N, t, marker="s", label=f"NPA level {lvl} ({use})")

    plt.xlabel("Number of particles N")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime comparison: exact diagonalization vs NPA (multiple levels)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Performance comparison vs reference NPA level (overlap with exact only) ----
    overlap_N = np.intersect1d(N_exact, N_ref)
    if overlap_N.size == 0:
        return {
            "exact": {"N": N_exact, "t": t_exact},
            "npa": {lvl: {"N": npa_series[lvl][0], "t": npa_series[lvl][1]} for lvl in levels},
            "reference_level": ref_level,
            "overlap": {"N": overlap_N, "ratio": None, "diff": None},
        }

    exact_map = {int(n): float(t) for n, t in zip(N_exact, t_exact)}
    ref_map   = {int(n): float(t) for n, t in zip(N_ref, t_ref)}

    t_exact_ol = np.array([exact_map[int(n)] for n in overlap_N], dtype=float)
    t_ref_ol   = np.array([ref_map[int(n)]   for n in overlap_N], dtype=float)

    ratio = t_ref_ol / t_exact_ol
    diff  = t_ref_ol - t_exact_ol

    # ---- Plot 2: ratio ----
    plt.figure(figsize=(9, 5))
    plt.plot(overlap_N, ratio, marker="^")
    plt.xlabel("Number of particles N")
    plt.ylabel(f"Runtime ratio (NPA lvl {ref_level} / Exact)")
    plt.title(f"Performance gap (overlap only): ratio vs reference level {ref_level}")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: absolute difference ----
    plt.figure(figsize=(9, 5))
    plt.plot(overlap_N, diff, marker="d")
    plt.xlabel("Number of particles N")
    plt.ylabel(f"Runtime difference (NPA lvl {ref_level} - Exact) [s]")
    plt.title(f"Performance gap (overlap only): difference vs reference level {ref_level}")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "exact": {"N": N_exact, "t": t_exact},
        "npa": {lvl: {"N": npa_series[lvl][0], "t": npa_series[lvl][1]} for lvl in levels},
        "reference_level": ref_level,
        "overlap": {"N": overlap_N, "t_exact": t_exact_ol, "t_ref": t_ref_ol, "ratio": ratio, "diff": diff},
    }









