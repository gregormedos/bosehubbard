import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

PRECISION = 14

fig, ax = plt.subplots(3, 2)

hs = bh.HilbertSpace(5, 2)
h = hs.op_hamiltonian_annihilate_create_pair_pbc()
ax[0, 0].imshow(h, norm=CenteredNorm())
w = np.linalg.eigvalsh(h)
w = np.diag(np.round(w, PRECISION))
ax[0, 1].imshow(w, norm=CenteredNorm())
s = hs.basis_transformation_n(h)
h = s.T @ h @ s
ax[1, 0].imshow(h, norm=CenteredNorm())
w = np.linalg.eigvalsh(h)
w = np.diag(np.round(w, PRECISION))
ax[1, 1].imshow(w, norm=CenteredNorm())
h = hs.op_hamiltonian_annihilate_create_pair_pbc()
s = hs.basis_transformation_k(h)
h = s.conj().T @ h @ s
ax[2, 0].imshow(np.abs(h), norm=CenteredNorm())
w = np.linalg.eigvalsh(h)
w = np.diag(np.round(w, PRECISION))
ax[2, 1].imshow(w, norm=CenteredNorm())
plt.tight_layout()

fig, ax = plt.subplots(2, 2)

hs = bh.HilbertSpace(5, 2, 'K', crystal_momentum=0)
h = hs.op_hamiltonian_annihilate_create_k()
ax[0, 0].imshow(np.abs(h), norm=CenteredNorm())
w = np.linalg.eigvalsh(h)
w = np.diag(np.round(w, PRECISION))
ax[0, 1].imshow(w, norm=CenteredNorm())
s = hs.basis_transformation_pk(h)
h = s.T @ h @ s
ax[1, 0].imshow(np.abs(h), norm=CenteredNorm())
w = np.linalg.eigvalsh(h)
w = np.diag(np.round(w, PRECISION))
ax[1, 1].imshow(w, norm=CenteredNorm())
plt.tight_layout()

plt.show()
