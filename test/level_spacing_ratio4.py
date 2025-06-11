import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def main():
    L = 8
    M = 2
    K = 0
    P = 1
    tunneling_rate = 1.0
    interaction_strength = 1.0
    g1s = np.linspace(0.01, 10.0, 30)
    g2s = np.linspace(0.01, 10.0, 30)
    keepvals = 100
    r = np.zeros((len(g1s), len(g2s)), dtype=float)
    hs = bh.HilbertSpace(L, M, 'PK', crystal_momentum=K, reflection_parity=P)
    dim = hs.dim
    print(hs.dim)
    H_tun = hs.op_hamiltonian_tunnel_pbc()
    H_int = hs.op_hamiltonian_interaction()
    H_b = hs.op_hamiltonian_annihilate_create()
    H_bb = hs.op_hamiltonian_annihilate_create_pair_pbc()
    for i, g1 in enumerate(g1s):
        for j, g2 in enumerate(g2s):
            hamiltonian = tunneling_rate * H_tun + interaction_strength * H_int + g1 * H_b + g2 * H_bb
            eigen_energies = np.linalg.eigvalsh(hamiltonian)[(dim - keepvals) // 2: (dim + keepvals) // 2]
            s = np.diff(eigen_energies)
            r[i, j] += np.mean([min(s1/s2, s2/s1) for s1, s2 in zip(s[1:], s[:-1])])
    plt.figure(dpi=300)
    pcm = plt.pcolormesh(*np.meshgrid(g1s, g2s), r.T, rasterized=True)
    plt.colorbar(pcm, ticks=[0.386, 0.5307], label='$r$')
    plt.xlabel('$g_1$')
    plt.ylabel('$g_2$')
    plt.tight_layout()
    plt.savefig('test/plots/level_spacing_ratio4.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
