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

cmap = mpl.colormaps['RdBu']


def test_symmetries(num: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    hs = bh.HilbertSpace(5, 2)
    h = HAMILTONIAN_DICT[num](hs)
    axes[0, 0].imshow(h, norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_n(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(h, norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 1].hist(w, BINS)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_k(h)
    h = s.conj().T @ h @ s
    axes[0, 2].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 2].hist(w, BINS)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_kn(h)
    h = s.conj().T @ h @ s
    axes[0, 3].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()


def test_decomposition_n(num: int):
    hs = bh.DecomposedHilbertSpace(5, 2, sym='N')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    w = []
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_DICT[num](hss)
        axes[i].imshow(h, norm=CenteredNorm(), cmap=cmap)
        w.extend(np.linalg.eigvalsh(h))

    fig.tight_layout()
    w = np.array(w)
    w = np.round(w, PRECISION)

    fig, axis = plt.subplots(figsize=(2.5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    axis.hist(w, BINS)

    fig.tight_layout()


def test_decomposition_k(num: int):
    hs = bh.DecomposedHilbertSpace(5, 2, sym='K')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    w = []
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_K_DICT[num](hss)
        axes[i].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
        w.extend(np.linalg.eigvalsh(h))

    fig.tight_layout()
    w = np.array(w)
    w = np.round(w, PRECISION)

    fig, axis = plt.subplots(figsize=(2.5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    axis.hist(w, BINS)

    fig.tight_layout()


def test_decomposition_kn(num: int):
    hs = bh.DecomposedHilbertSpace(5, 2, sym='KN')
    fig, axes = plt.subplots(5, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 5 * 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    w = []
    for i, hss in enumerate(hs.subspaces):
        for j, hsss in enumerate(hss.subspaces):
            h = HAMILTONIAN_K_DICT[num](hsss)
            axes[j, i].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
            w.extend(np.linalg.eigvalsh(h))

    fig.tight_layout()
    w = np.array(w)
    w = np.round(w, PRECISION)

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    axis.hist(w, BINS)

    fig.tight_layout()


def test_symmetries_k(num: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    hs = bh.HilbertSpace(6, 2, 'K', crystal_momentum=0)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 0].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(6, 2, 'K', crystal_momentum=3)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 2].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h @ s
    axes[0, 3].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()


def test_symmetries_kn(num: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    hs = bh.HilbertSpace(6, 2, 'KN', n_tot=3, crystal_momentum=0)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 0].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h @ s
    axes[0, 1].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(6, 2, 'KN', n_tot=3, crystal_momentum=3)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 2].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h @ s
    axes[0, 3].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
    w = np.linalg.eigvalsh(h)
    w = np.round(w, PRECISION)
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()


def test_decomposition_pk(num: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    w = [[], []]

    hs = bh.DecomposedHilbertSpace(6, 2, 'K', 'PK', crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_K_DICT[num](hss)
        axes[0, i].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
        w[0].extend(np.linalg.eigvalsh(h))
    hs = bh.DecomposedHilbertSpace(6, 2, 'K', 'PK', crystal_momentum=3)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_K_DICT[num](hss)
        axes[1, i].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
        w[1].extend(np.linalg.eigvalsh(h))

    fig.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    for i in range(2):
        w[i] = np.array(w[i])
        w[i] = np.round(w[i], PRECISION)
        axes[i].hist(w[i], BINS)

    fig.tight_layout()


def test_decomposition_pkn(num: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    w = [[], []]

    hs = bh.DecomposedHilbertSpace(6, 2, 'KN', 'PKN', n_tot=3, crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_K_DICT[num](hss)
        axes[0, i].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
        w[0].extend(np.linalg.eigvalsh(h))
    hs = bh.DecomposedHilbertSpace(6, 2, 'KN', 'PKN', n_tot=3, crystal_momentum=3)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_K_DICT[num](hss)
        axes[1, i].imshow(np.abs(h), norm=CenteredNorm(), cmap=cmap)
        w[1].extend(np.linalg.eigvalsh(h))

    fig.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    for i in range(2):
        w[i] = np.array(w[i])
        w[i] = np.round(w[i], PRECISION)
        axes[i].hist(w[i], BINS)

    fig.tight_layout()


def main():
    #test_symmetries(1)
    #test_decomposition_n(1)
    #test_decomposition_k(1)
    #test_decomposition_kn(1)
    #test_symmetries(2)
    #test_decomposition_n(2)
    #test_decomposition_k(1)
    #test_decomposition_kn(1)
    #test_symmetries(3)
    #test_decomposition_n(3)
    #test_decomposition_k(2)
    #test_decomposition_kn(2)
    #test_symmetries(4)
    #test_decomposition_n(4)
    #test_decomposition_k(2)
    #test_decomposition_kn(2)
    #test_symmetries_k(1)
    #test_decomposition_pk(1)
    test_symmetries_kn(1)
    test_decomposition_pkn(1)
    #test_symmetries_k(2)
    #test_decomposition_pk(2)
    #test_symmetries_kn(2)
    #test_decomposition_pkn(2)
    plt.show()


if __name__ == '__main__':
    main()
