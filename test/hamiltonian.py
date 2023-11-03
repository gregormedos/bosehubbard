import bosehubbard as bh
import matplotlib.pyplot as plt


def plot_hamiltonian(hs: bh.HilbertSpace):
    h = bh.op_hamiltonian_tunnel_pbc(hs)
    plt.figure(dpi=100)
    plt.imshow(h)


hs = bh.HilbertSpace(5, 2)
plot_hamiltonian(hs)
for n_tot in range(5 * 2 + 1):
    hs = bh.HilbertSpace(5, 2, space='N', n_tot=n_tot)
    plot_hamiltonian(hs)
plt.show()
