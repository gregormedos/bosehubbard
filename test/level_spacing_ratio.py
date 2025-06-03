import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def main():
    num_sites = 6
    tunneling_rate = 1.0
    interaction_strengths = np.logspace(-2, 3, 20)
    disorder_strengths = np.linspace(0.01, 10.0, 20)
    keepvals = 100
    num_realizations = 100
    r = np.zeros((len(interaction_strengths), len(disorder_strengths)), dtype=float)
    hs = bh.HilbertSpace(num_sites, num_sites, 'N', num_sites)
    dim = hs.dim
    print(hs.dim)
    H_tun = hs.op_hamiltonian_tunnel_obc()
    H_int = hs.op_hamiltonian_interaction()
    for i, interaction_strength in enumerate(interaction_strengths):
        for _ in range(num_realizations):
            H_dis = hs.op_potential_disorder()
            for j, disorder_strength in enumerate(disorder_strengths):
                hamiltonian = tunneling_rate * H_tun + interaction_strength * H_int + disorder_strength * H_dis
                eigen_energies = np.linalg.eigvalsh(hamiltonian)[(dim - keepvals) // 2: (dim + keepvals) // 2]
                s = np.diff(eigen_energies)
                r[i, j] += np.mean([min(s1/s2, s2/s1) for s1, s2 in zip(s[1:], s[:-1])])
    r /= num_realizations
    plt.figure(dpi=300)
    pcm = plt.pcolormesh(*np.meshgrid(interaction_strengths, disorder_strengths), r.T, rasterized=True)
    plt.colorbar(pcm, ticks=[0.386, 0.5307], label='$r$')
    plt.xlabel('$U$')
    plt.xscale('log')
    plt.ylabel('$W$')
    plt.tight_layout()
    plt.savefig('test/plots/level_spacing_ratio.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
