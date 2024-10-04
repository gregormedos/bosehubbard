import sys
import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

plt.rcParams.update({'text.usetex': False,
                     'font.size': 18})


def main():
    num_sites = 1000
    tunneling_rate = 0.5
    hs = bh.HilbertSpace(num_sites, 1, 'N', 1)
    hamiltonian_tunnel = tunneling_rate * hs.op_hamiltonian_tunnel_pbc()
    eigen_energies = np.linalg.eigvalsh(hamiltonian_tunnel)
    energy_step = 0.01
    energies = np.arange(-2.0 * tunneling_rate + energy_step, 2.0 * tunneling_rate, energy_step)
    dos = 1.0 / np.sqrt(4.0 * tunneling_rate ** 2 - energies ** 2)
    dos /= np.sum(dos) * energy_step
    plt.figure(dpi=300)
    plt.hist(eigen_energies, bins=100, density=True)
    plt.plot(energies, dos, linewidth=2, color='black', linestyle='dashed')
    plt.xlabel('$E$')
    plt.ylabel('DOS($E$)')
    plt.tight_layout()
    plt.savefig('test/plots/tight_binding.pdf')


if __name__ == '__main__':
    main()
