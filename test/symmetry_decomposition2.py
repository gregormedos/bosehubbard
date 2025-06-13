import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

HAMILTONIAN_DICT = {
    't-pbc': bh.HilbertSpace.op_hamiltonian_tunnel_pbc,
    't-obc': bh.HilbertSpace.op_hamiltonian_tunnel_obc,
    'g2-pbc': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc,
    'g2-obc': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_obc,
    'g1': bh.HilbertSpace.op_hamiltonian_annihilate_create,
    'U': bh.HilbertSpace.op_hamiltonian_interaction,
    'W': bh.HilbertSpace.op_potential_disorder
}
HAMILTONIAN_K_DICT = {
    't-pbc': bh.HilbertSpace.op_hamiltonian_tunnel_k,
    'g2-pbc': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_k,
    'g1': bh.HilbertSpace.op_hamiltonian_annihilate_create_k,
    'U': bh.HilbertSpace.op_hamiltonian_interaction
}
HAMILTONIAN_PK_DICT = {
    't-pbc': bh.HilbertSpace.op_hamiltonian_tunnel_pk,
    'g2-pbc': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pk,
    'g1': bh.HilbertSpace.op_hamiltonian_annihilate_create_pk,
    'U': bh.HilbertSpace.op_hamiltonian_interaction
}
PRECISION = 12
BINS = 50


def main():
    L = 6
    M = 2
    test_bh(L, M)
    test_bhW(L, M)
    test_bhg1(L, M)
    test_bhg1W(L, M)
    test_bhg2(L, M)
    test_bhg2W(L, M)
    test_bhg1g2(L, M)
    test_bhg1g2W(L, M)


def test_bh(L, M):
    terms = [('t-pbc', 1.0), ('U', 1.0)]
    test_symmetries(terms, L, M)
    test_decomposition_n(terms, L, M)
    test_decomposition_z2(terms, L, M)
    test_decomposition_k(terms, L, M)
    test_decomposition_kn(terms, L, M)
    test_decomposition_kz2(terms, L, M)
    test_symmetries_k(terms, L, M)
    for N in range(L * M + 1):
        test_symmetries_kn(terms, L, M, N)
    for Z2 in (1, -1):
        test_symmetries_kz2(terms, L, M, Z2)
    test_decomposition_pk(terms, L, M)
    for N in range(L * M + 1):
        test_decomposition_pkn(terms, L, M, N)
    for Z2 in (1, -1):
        test_decomposition_pkz2(terms, L, M, Z2)


def test_bhW(L, M):
    terms = [('t-obc', 1.0), ('U', 1.0), ('W', 1.0)]
    test_symmetries(terms, L, M)
    test_decomposition_n(terms, L, M)
    test_decomposition_z2(terms, L, M)


def test_bhg1(L, M):
    terms = [('t-pbc', 1.0), ('U', 1.0), ('g1', 1.0)]
    test_symmetries(terms, L, M)
    test_decomposition_k(terms, L, M)
    test_symmetries_k(terms, L, M)
    test_decomposition_pk(terms, L, M)


def test_bhg1W(L, M):
    terms = [('t-obc', 1.0), ('U', 1.0), ('g1', 1.0), ('W', 1.0)]
    test_symmetries(terms, L, M)


def test_bhg2(L, M):
    terms = [('t-pbc', 1.0), ('U', 1.0), ('g2-pbc', 1.0)]
    test_symmetries(terms, L, M)
    test_decomposition_z2(terms, L, M)
    test_decomposition_k(terms, L, M)
    test_decomposition_kz2(terms, L, M)
    test_symmetries_k(terms, L, M)
    for Z2 in (1, -1):
        test_symmetries_kz2(terms, L, M, Z2)
    test_decomposition_pk(terms, L, M)
    for Z2 in (1, -1):
        test_decomposition_pkz2(terms, L, M, Z2)


def test_bhg2W(L, M):
    terms = [('t-obc', 1.0), ('U', 1.0), ('g2-obc', 1.0), ('W', 1.0)]
    test_symmetries(terms, L, M)
    test_decomposition_z2(terms, L, M)


def test_bhg1g2(L, M):
    terms = [('t-pbc', 1.0), ('U', 1.0), ('g2-pbc', 1.0), ('g1', 1.0)]
    test_symmetries(terms, L, M)
    test_decomposition_k(terms, L, M)
    test_symmetries_k(terms, L, M)
    test_decomposition_pk(terms, L, M)


def test_bhg1g2W(L, M):
    terms = [('t-obc', 1.0), ('U', 1.0), ('g2-obc', 1.0), ('g1', 1.0), ('W', 1.0)]
    test_symmetries(terms, L, M)


def test_symmetries(terms: list, num_sites: int, n_max: int):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max)
    h = np.sum([strength * HAMILTONIAN_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 0].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_n(h)
    h = s.T @ h_copy @ s
    axes[0, 1].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_spect_n_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 1].hist(w, BINS)
    s = hs.basis_transformation_z2(h)
    h = s.T @ h_copy @ s
    axes[0, 2].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_spect_z2_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_k(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 3].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_spect_k_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 3].hist(w, BINS)
    s = hs.basis_transformation_kn(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 4].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_spect_kn_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 4].hist(w, BINS)
    s = hs.basis_transformation_kz2(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 5].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_spect_kz2_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 5].hist(w, BINS)

    fig.tight_layout()
    fig.savefig(f'test/plots/symmetries_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_n(terms: list, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='N')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_DICT[term](hss) for term, strength in terms], axis=0)

        axes[i].imshow(np.abs(h))
        w_sectors[f'({hss.n_tot})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_n_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
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
    with open(f'test/data/decomposition_n_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_n_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_z2(terms: list, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='Z2')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_DICT[term](hss) for term, strength in terms], axis=0)

        axes[i].imshow(np.abs(h))
        w_sectors[f'({hss.n_tot_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_z2_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
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
    with open(f'test/data/decomposition_z2_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_z2_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_k(terms: list, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='K')
    fig, axes = plt.subplots(1, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_K_DICT[term](hss) for term, strength in terms], axis=0)

        axes[i].imshow(np.abs(h))
        w_sectors[f'({hss.crystal_momentum})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_k_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
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
    with open(f'test/data/decomposition_k_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_k_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_kn(terms: list, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='KN')
    fig, axes = plt.subplots(num_sites, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, num_sites * 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        for j, hsss in enumerate(hss.subspaces):
            h = np.sum([strength * HAMILTONIAN_K_DICT[term](hsss) for term, strength in terms], axis=0)
    
            axes[j, i].imshow(np.abs(h))
            w_sectors[f'({hsss.n_tot}|{hsss.crystal_momentum})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_kn_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
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
    with open(f'test/data/decomposition_kn_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_kn_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_kz2(terms: list, num_sites: int, n_max: int):
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, sym='KZ2')
    fig, axes = plt.subplots(num_sites, len(hs.subspaces), figsize=(len(hs.subspaces) * 2.5, num_sites * 2.5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors = {}
    for i, hss in enumerate(hs.subspaces):
        for j, hsss in enumerate(hss.subspaces):
            h = np.sum([strength * HAMILTONIAN_K_DICT[term](hsss) for term, strength in terms], axis=0)
    
            axes[j, i].imshow(np.abs(h))
            w_sectors[f'({hsss.n_tot_parity}|{hsss.crystal_momentum})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_kz2_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
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
    with open(f'test/data/decomposition_kz2_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_kz2_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_symmetries_k(terms: list, num_sites: int, n_max: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max, space='K', crystal_momentum=0)
    h = np.sum([strength * HAMILTONIAN_K_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 0].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_k_spect_k=zero_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h_copy @ s
    axes[0, 1].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_k_spect_pk_k=zero_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(num_sites, n_max, space='K', crystal_momentum=num_sites//2)
    h = np.sum([strength * HAMILTONIAN_K_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 2].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_k_spect_k=bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.T @ h_copy @ s
    axes[0, 3].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_k_spect_pk_k=bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')
    
    axes[1, 3].hist(w, BINS)

    fig.tight_layout()
    fig.savefig(f'test/plots/symmetries_k_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_pk(terms: list, num_sites: int, n_max: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors_list = [{}, {}]
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='K', sym='PK', crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_PK_DICT[term](hss) for term, strength in terms], axis=0)

        axes[0, i].imshow(np.abs(h))
        w_sectors_list[0][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='K', sym='PK', crystal_momentum=num_sites//2)
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_PK_DICT[term](hss) for term, strength in terms], axis=0)

        axes[1, i].imshow(np.abs(h))
        w_sectors_list[1][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_pk_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
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
        with open(f'test/data/decomposition_pk_spect_{k}_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_pk_spect_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_symmetries_kn(terms: list, num_sites: int, n_max: int, n_tot: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max, space='KN', n_tot=n_tot, crystal_momentum=0)
    h = np.sum([strength * HAMILTONIAN_K_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 0].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kn_spect_N={n_tot}_k=zero_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 1].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kn_spect_pk_N={n_tot}_k=zero_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(num_sites, n_max, space='KN', n_tot=n_tot, crystal_momentum=num_sites//2)
    h = np.sum([strength * HAMILTONIAN_K_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 2].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kn_spect_N={n_tot}_k=bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 3].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kn_spect_pk_N={n_tot}_k=bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 3].hist(w, BINS)

    fig.tight_layout()
    fig.savefig(f'test/plots/symmetries_kn_N={n_tot}_k=zero-bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_pkn(terms: list, num_sites: int, n_max: int, n_tot: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors_list = [{}, {}]
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='KN', sym='PKN', n_tot=n_tot, crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_PK_DICT[term](hss) for term, strength in terms], axis=0)

        axes[0, i].imshow(np.abs(h))
        w_sectors_list[0][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='KN', sym='PKN', n_tot=n_tot, crystal_momentum=num_sites//2)
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_PK_DICT[term](hss) for term, strength in terms], axis=0)

        axes[1, i].imshow(np.abs(h))
        w_sectors_list[1][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_pkn_N={n_tot}_k=zero-bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')

    for i, k in enumerate(('zero', 'bragg')):
        spect = {}
        for sector, w_sector in w_sectors_list[i].items():
            for energy in w_sector:
                if energy not in spect:
                    spect[energy] = [sector, 1]
                else:
                    spect[energy][0] += sector
                    spect[energy][1] += 1
        spect = {energy: spect[energy] for energy in sorted(spect.keys())}
        with open(f'test/data/decomposition_pkn_spect_N={n_tot}_k={k}_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_pkn_spect_N={n_tot}_k=zero-bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_symmetries_kz2(terms: list, num_sites: int, n_max: int, n_tot_parity: int):
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    hs = bh.HilbertSpace(num_sites, n_max, space='KZ2', n_tot_parity=n_tot_parity, crystal_momentum=0)
    h = np.sum([strength * HAMILTONIAN_K_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 0].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kz2_spect_Z2={n_tot_parity}_k=zero_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 0].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 1].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kz2_spect_pk_Z2={n_tot_parity}_k=zero_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 1].hist(w, BINS)
    hs = bh.HilbertSpace(num_sites, n_max, space='KZ2', n_tot_parity=n_tot_parity, crystal_momentum=num_sites//2)
    h = np.sum([strength * HAMILTONIAN_K_DICT[term](hs) for term, strength in terms], axis=0)
    h_copy = np.copy(h)
    axes[0, 2].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kz2_spect_Z2={n_tot_parity}_k=bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 2].hist(w, BINS)
    s = hs.basis_transformation_pk(h)
    h = s.conj().T @ h_copy @ s
    axes[0, 3].imshow(np.abs(h))
    w = np.round(np.linalg.eigvalsh(h), PRECISION)

    spect = {}
    for energy in w:
        if energy not in spect:
            spect[energy] = 1
        else:
            spect[energy] += 1
    spect = {energy: spect[energy] for energy in sorted(spect.keys())}
    with open(f'test/data/symmetries_kz2_spect_pk_Z2={n_tot_parity}_k=bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
        for energy, degeneracy in spect.items():
            file.write(f'{energy:.14f}[{degeneracy}]\n')

    axes[1, 3].hist(w, BINS)

    fig.tight_layout()
    fig.savefig(f'test/plots/symmetries_kz2_Z2={n_tot_parity}_k=zero-bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


def test_decomposition_pkz2(terms: list, num_sites: int, n_max: int, n_tot_parity: int):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    w_sectors_list = [{}, {}]
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='KZ2', sym='PKZ2', n_tot_parity=n_tot_parity, crystal_momentum=0)
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_PK_DICT[term](hss) for term, strength in terms], axis=0)

        axes[0, i].imshow(np.abs(h))
        w_sectors_list[0][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)
    hs = bh.DecomposedHilbertSpace(num_sites, n_max, space='KZ2', sym='PKZ2', n_tot_parity=n_tot_parity, crystal_momentum=num_sites//2)
    for i, hss in enumerate(hs.subspaces):
        h = np.sum([strength * HAMILTONIAN_PK_DICT[term](hss) for term, strength in terms], axis=0)

        axes[1, i].imshow(np.abs(h))
        w_sectors_list[1][f'({hss.reflection_parity})'] = np.round(np.linalg.eigvalsh(h), PRECISION)

    fig.tight_layout()
    fig.savefig(f'test/plots/decomposition_pkz2_Z2={n_tot_parity}_k=zero-bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')

    for i, k in enumerate(('zero', 'bragg')):
        spect = {}
        for sector, w_sector in w_sectors_list[i].items():
            for energy in w_sector:
                if energy not in spect:
                    spect[energy] = [sector, 1]
                else:
                    spect[energy][0] += sector
                    spect[energy][1] += 1
        spect = {energy: spect[energy] for energy in sorted(spect.keys())}
        with open(f'test/data/decomposition_pkz2_spect_Z2={n_tot_parity}_k={k}_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.txt', 'w') as file:
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
    fig.savefig(f'test/plots/decomposition_pkz2_spect_Z2={n_tot_parity}_k=zero-bragg_{'_'.join(str(term) + '-' + str(strength) for term, strength in terms)}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
