from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from tools import *

from spins_sdp.pauli_strings import npa_level
from spins_sdp.exact import ising_hamiltonian

def npa_basis_size_heatmap(N: int, max_NPA_level: int) -> np.ndarray:
  """Generate a heatmap of NPA basis sizes for varying number of particles and NPA levels.

  Args:
      N (int): Maximum number of particles.
      max_NPA_level (int): Maximum NPA level.
  """

  def _basis_size(N: int, lvl: int) -> int:
    return len(npa_level(N=N, NPA_level=lvl))

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


def plot_exact_vs_npa_energy_vs_N(
  N_values: range,
  J: float,
  h: float,
  k: float,
  NPA_level: int,
  boundary: str = "open",
  solver: str = "CVXOPT",
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
    E0, _ = exact_ground_state_eigenpair(H)
    exact_Es.append(float(E0))

    # NPA lower bound
    E_lb = npa_lb_energy(J=J, h=h, k=k, N=N, NPA_level=NPA_level, solver=solver, boundary=boundary)
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









