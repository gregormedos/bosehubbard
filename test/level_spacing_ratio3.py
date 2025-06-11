import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def main():
    L = 10
    M = 2
    N = L
    K = 0
    P = 1
    tunneling_rates = np.linspace(0.01, 10.0, 30)
    interaction_strengths = np.linspace(0.01, 10.0, 30)
    keepvals = 100
    r = np.zeros((len(tunneling_rates), len(interaction_strengths)), dtype=float)
    hs = bh.HilbertSpace(L, M, 'PKN', n_tot=N, crystal_momentum=K, reflection_parity=P)
    dim = hs.dim
    print(hs.dim)
    H_tun = hs.op_hamiltonian_tunnel_pbc()
    H_int = hs.op_hamiltonian_interaction()
    for i, tunneling_rate in enumerate(tunneling_rates):
        for j, interaction_strength in enumerate(interaction_strengths):
            hamiltonian = tunneling_rate * H_tun + interaction_strength * H_int
            eigen_energies = np.linalg.eigvalsh(hamiltonian)[(dim - keepvals) // 2: (dim + keepvals) // 2]
            s = np.diff(eigen_energies)
            r[i, j] += np.mean([min(s1/s2, s2/s1) for s1, s2 in zip(s[1:], s[:-1])])
    plt.figure(dpi=300)
    pcm = plt.pcolormesh(*np.meshgrid(tunneling_rates, interaction_strengths), r.T, rasterized=True)
    plt.colorbar(pcm, ticks=[0.386, 0.5307], label='$r$')
    plt.xlabel('$t$')
    plt.ylabel('$U$')
    plt.tight_layout()
    plt.savefig('test/plots/level_spacing_ratio3.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
