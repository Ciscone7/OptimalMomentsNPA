import numpy as np
import matplotlib.pyplot as plt

from spins_sdp.pauli_strings import npa_level

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