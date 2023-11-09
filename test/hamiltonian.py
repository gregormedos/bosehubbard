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
HAMILTONIAN_K_DICT = {
    1: bh.HilbertSpace.op_hamiltonian_tunnel_k,
    2: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_k,
}
PRECISION = 15
BINS = 60

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
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_n(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(h, norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 1].hist(w, BINS)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_k(h)
    h = s.conj().T @ h @ s
    axes[0, 2].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 2].hist(w, BINS)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_kn(h)
    h = s.conj().T @ h @ s
    axes[0, 3].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()


def test_kblock(num: int):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
    for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xticklabels([])
            axis.set_yticklabels([])
    norm = CenteredNorm()
    cmap = mpl.colormaps['RdBu']

    hs = bh.HilbertSpace(5, 3, 'K', crystal_momentum=0)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 0].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 1].hist(w, BINS)
    h = HAMILTONIAN_K_DICT[num](hs)
    s = hs.basis_transformation_pkn(h)
    h = s.conj().T @ h @ s
    axes[0, 2].imshow(np.abs(h), norm=norm, cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 2].hist(w, BINS)

    fig.tight_layout()


def main():
    test_symmetries(1)
    test_symmetries(2)
    test_symmetries(3)
    test_symmetries(4)
    test_kblock(1)
    test_kblock(2)
    plt.show()


if __name__ == '__main__':
    main()
