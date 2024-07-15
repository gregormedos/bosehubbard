import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

HAMILTONIAN_DICT = {
    1: bh.HilbertSpace.op_hamiltonian_tunnel_pbc,
    2: bh.HilbertSpace.op_hamiltonian_tunnel_obc,
    3: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc,
    4: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_obc,
    5: bh.HilbertSpace.op_hamiltonian_annihilate_create
}
HAMILTONIAN_K_DICT = {
    1: bh.HilbertSpace.op_hamiltonian_tunnel_k,
    2: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_k,
    3: bh.HilbertSpace.op_hamiltonian_annihilate_create_k
}
HAMILTONIAN_PK_DICT = {
    1: bh.HilbertSpace.op_hamiltonian_tunnel_pk,
    2: bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pk,
    3: bh.HilbertSpace.op_hamiltonian_annihilate_create_pk
}
PRECISION = 14
BINS = 50


def main():
    test_symmetries(1, 6, 2)
    test_decomposition_n(1, 6, 2)
    test_decomposition_k(1, 6, 2)
    test_decomposition_kn(1, 6, 2)
    test_symmetries_k(1, 6, 2)
    test_decomposition_pk(1, 6, 2)
    test_symmetries_kn(1, 6, 2, 3)
    test_decomposition_pkn(1, 6, 2, 3)


def test_symmetries(num: int, num_sites: int, n_max: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max)
    h = HAMILTONIAN_DICT[num](hs)
    axes[0, 0].imshow(h)
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_spect.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_n(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(h)
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_spect_n.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 1].hist(w, BINS)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_k(h)
    h = s.conj().T @ h @ s
    axes[0, 2].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_spect_k.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 2].hist(w, BINS)
    h = HAMILTONIAN_DICT[num](hs)
    s = hs.basis_transformation_kn(h)
    h = s.conj().T @ h @ s
    axes[0, 3].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_spect_kn.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()
    fig.savefig('test/symmetries.pdf')
    plt.close(fig)


def test_decomposition_n(num: int, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='N')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_DICT[num](hss)

        axes[i].imshow(h)
        w_sectors[f'({hss.n_tot})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig('test/decomposition_n.pdf')
    plt.close(fig)

    spect = {}
    for sector, w_sector in w_sectors.items():
        for energy in w_sector:
            if energy not in spect:
                spect[energy] = [sector, 1]
            else:
                spect[energy][0] += sector
                spect[energy][1] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/decomposition_n_spect.txt', 'w') as file:
        for energy, (sector, degeneracy) in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]{sector}\n')

    fig, axis = plt.subplots(figsize=(2.5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])

    w = []
    for w_sector in w_sectors.values():
        w.extend(w_sector)
    axis.hist(w, BINS)

    fig.tight_layout()
    fig.savefig('test/decomposition_n_spect.pdf')
    plt.close(fig)


def test_decomposition_k(num: int, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='K')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_K_DICT[num](hss)

        axes[i].imshow(np.abs(h))
        w_sectors[f'({hss.crystal_momentum})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig('test/decompostion_k.pdf')
    plt.close(fig)

    spect = {}
    for sector, w_sector in w_sectors.items():
        for energy in w_sector:
            if energy not in spect:
                spect[energy] = [sector, 1]
            else:
                spect[energy][0] += sector
                spect[energy][1] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/decomposition_k_spect.txt', 'w') as file:
        for energy, (sector, degeneracy) in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]{sector}\n')

    fig, axis = plt.subplots(figsize=(2.5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])

    w = []
    for w_sector in w_sectors.values():
        w.extend(w_sector)
    axis.hist(w, BINS)

    fig.tight_layout()
    fig.savefig('test/decompostion_k_spect.pdf')
    plt.close(fig)


def test_decomposition_kn(num: int, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='KN')
    fig, axes = plt.subplots(num_sites + 1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, (num_sites + 1) * 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        for j, hsss in enumerate(hss.subspaces):
            h = HAMILTONIAN_K_DICT[num](hsss)
    
            axes[j, i].imshow(np.abs(h))
            w_sectors[f'({hsss.n_tot}|{hsss.crystal_momentum})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig('test/decompositon_kn.pdf')
    plt.close(fig)

    spect = {}
    for sector, w_sector in w_sectors.items():
        for energy in w_sector:
            if energy not in spect:
                spect[energy] = [sector, 1]
            else:
                spect[energy][0] += sector
                spect[energy][1] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/decomposition_kn_spect.txt', 'w') as file:
        for energy, (sector, degeneracy) in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]{sector}\n')

    fig, axis = plt.subplots(figsize=(2.5, 2.5))
    axis.set_xticks([])
    axis.set_yticks([])

    w = []
    for w_sector in w_sectors.values():
        w.extend(w_sector)
    axis.hist(w, BINS)

    fig.tight_layout()
    fig.savefig('test/decompositon_kn_spect.pdf')
    plt.close(fig)


def test_symmetries_k(num: int, num_sites: int, n_max: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max, space='K', crystal_momentum=0)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 0].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_k_spect_k=zero.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h @ s
    axes[0, 1].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_k_spect_pk_k=zero.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(num_sites, n_max, space='K', crystal_momentum=num_sites//2)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 2].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_k_spect_k=bragg.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h @ s
    axes[0, 3].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_k_spect_pk_k=bragg.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()
    fig.savefig('test/symmetries_k.pdf')
    plt.close(fig)


def test_decomposition_pk(num: int, num_sites: int, n_max: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors_list = [{}, {}]
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='K', sym='PK', crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_PK_DICT[num](hss)

        axes[0, i].imshow(h)
        w_sectors_list[0][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='K', sym='PK', crystal_momentum=num_sites//2)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_PK_DICT[num](hss)

        axes[1, i].imshow(h)
        w_sectors_list[1][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig('test/decompostion_pk.pdf')
    plt.close(fig)

    for i, k in enumerate(('k=zero', 'k=bragg')):
        spect = {}
        for sector, w_sector in w_sectors_list[i].items():
            for energy in w_sector:
                if energy not in spect:
                    spect[energy] = [sector, 1]
                else:
                    spect[energy][0] += sector
                    spect[energy][1] += 1
        spect = {energy: spect[energy] for energy in sorted(spect.keys())}
        with open(f'test/decomposition_pk_spect_{k}.txt', 'w') as file:
            for energy, (sector, degeneracy) in spect.items():
                file.write(f'{energy:.14f}[{degeneracy}]{sector}\n')

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w = [[], []]
    for i in range(2):
        for w_sector in w_sectors_list[i].values():
            w[i].extend(w_sector)
        axes[i].hist(w[i], BINS)

    fig.tight_layout()
    fig.savefig('test/decompostion_pk_spect.pdf')
    plt.close(fig)


def test_symmetries_kn(num: int, num_sites: int, n_max: int, n_tot: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max, space='KN', n_tot=n_tot, crystal_momentum=0)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 0].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_kn_spect_k=zero.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h @ s
    axes[0, 1].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_kn_spect_pk_k=zero.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(num_sites, n_max, space='KN', n_tot=n_tot, crystal_momentum=num_sites//2)
    h = HAMILTONIAN_K_DICT[num](hs)
    axes[0, 2].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_kn_spect_k=bragg.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h @ s
    axes[0, 3].imshow(np.real(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open('test/symmetries_kn_spect_pk_k=bragg.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 3].hist(w, BINS)

    fig.tight_layout()
    fig.savefig('test/symmetries_kn.pdf')
    plt.close(fig)


def test_decomposition_pkn(num: int, num_sites: int, n_max: int, n_tot: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors_list = [{}, {}]
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='KN', sym='PKN', n_tot=n_tot, crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_PK_DICT[num](hss)

        axes[0, i].imshow(h)
        w_sectors_list[0][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='KN', sym='PKN', n_tot=n_tot, crystal_momentum=num_sites//2)
    for i, hss in enumerate(hs.subspaces):
        h = HAMILTONIAN_PK_DICT[num](hss)

        axes[1, i].imshow(h)
        w_sectors_list[1][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig('test/decompostion_pkn.pdf')

    for i, k in enumerate(('k=zero', 'k=bragg')):
        spect = {}
        for sector, w_sector in w_sectors_list[i].items():
            for energy in w_sector:
                if energy not in spect:
                    spect[energy] = [sector, 1]
                else:
                    spect[energy][0] += sector
                    spect[energy][1] += 1
        spect = {energy: spect[energy] for energy in sorted(spect.keys())}
        with open(f'test/decomposition_pkn_spect_{k}.txt', 'w') as file:
            for energy, (sector, degeneracy) in spect.items():
                file.write(f'{energy:.14f}[{degeneracy}]{sector}\n')

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w = [[], []]
    for i in range(2):
        for w_sector in w_sectors_list[i].values():
            w[i].extend(w_sector)
        axes[i].hist(w[i], BINS)

    fig.tight_layout()
    fig.savefig('test/decompostion_pkn_spect.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
