import sys
import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

plt.rcParams.update({'font.size': 18})


def main():
    num_sites = 1000
    tunneling_rate = 1.0
    disorder_strengths = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]
    hs = bh.HilbertSpace(num_sites, 1, 'N', 1)
    H_tun = hs.op_hamiltonian_tunnel_pbc()
    H_dis = hs.op_potential_disorder()
    energy_step = 0.01
    energies = np.arange(-2.0 * tunneling_rate + energy_step, 2.0 * tunneling_rate, energy_step)
    dos = 1.0 / np.sqrt(4.0 * tunneling_rate ** 2 - energies ** 2)
    dos /= np.sum(dos) * energy_step
    plt.figure(dpi=300)
    for disorder_strength in disorder_strengths:
        hamiltonian_tunnel = tunneling_rate * H_tun + disorder_strength * H_dis
        eigen_energies = np.linalg.eigvalsh(hamiltonian_tunnel)
        plt.hist(eigen_energies, bins=100, density=True, alpha=0.7, label='$W={}$'.format(disorder_strength))
    plt.plot(energies, dos, linewidth=2, color='black', linestyle='dashed')
    plt.xlim(-4.0 * tunneling_rate, 4.0 * tunneling_rate)
    plt.xlabel('$E$')
    plt.ylabel('DOS($E$)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('test/plots/disorder.pdf')


if __name__ == '__main__':
    main()
