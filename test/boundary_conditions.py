import sys
import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

plt.rcParams.update({'font.size': 18})


def main():
    num_sites = 1000
    tunneling_rate = 0.5
    hs = bh.HilbertSpace(num_sites, 1, 'N', 1)
    plt.figure(dpi=300)
    hamiltonian_tunnel = tunneling_rate * hs.op_hamiltonian_tunnel_pbc()
    eigen_energies = np.linalg.eigvalsh(hamiltonian_tunnel)
    plt.hist(eigen_energies, bins=100, density=True, alpha=0.7, label='PBC')
    hamiltonian_tunnel = tunneling_rate * hs.op_hamiltonian_tunnel_obc()
    eigen_energies = np.linalg.eigvalsh(hamiltonian_tunnel)
    plt.hist(eigen_energies, bins=100, density=True, alpha=0.7, label='OBC')
    plt.xlabel('$E$')
    plt.ylabel('DOS($E$)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('test/plots/boundary_conditions.pdf')


if __name__ == '__main__':
    main()
