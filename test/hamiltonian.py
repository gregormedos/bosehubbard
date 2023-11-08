import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import matplotlib as mpl

HAMILTONIAN_DICT = {
    1: bh.HilbertSpace.op_hamiltonian_tunnel_pbc,
    2: bh.HilbertSpace.op_hamiltonian_tunnel_obc,
    3: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc,
    4: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_obc,
}
PRECISION = 14

def test_symmetries(num: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xticklabels([])
            axis.set_yticklabels([])
    norm = CenteredNorm()
    cmap = mpl.colormaps['RdBu']

    hs = bh.HilbertSpace(5, 2)
    h = HAMILTONIAN_DICT[num](hs)
    axes[0, 0].imshow(h, norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 0].hist(w, 60)
    s = hs.basis_transformation_n(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(h, norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 1].hist(w, 60)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_k(h)
    h = s.conj().T @ h @ s
    axes[0, 2].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 2].hist(w, 60)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_kn(h)
    h = s.conj().T @ h @ s
    axes[0, 3].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 3].hist(w, 60)

    fig.tight_layout()


def main():
    test_symmetries(1)
    test_symmetries(2)
    test_symmetries(3)
    test_symmetries(4)
    plt.show()


if __name__ == '__main__':
    main()
