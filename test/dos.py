import os
import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def main():
    PKN_sector(10, 10, 10, 0, 1)
    PKN_sector(10, 10, 10, 0, -1)
    PKN_sector(10, 10, 10, 5, 1)
    PKN_sector(10, 10, 10, 5, -1)


def PKN_sector(L, M, N, K, P):
    file_id = f'L={L}_M={M}_N={N}_K={K}_P={P}'
    file_name = f'test/data/{file_id}.npy'
    if os.path.isfile(file_name):
        w = np.load(file_name)
    else:
        hs = bh.HilbertSpace(L, M, 'PKN', N, K, P)
        h = hs.op_hamiltonian_tunnel_k() + hs.op_hamiltonian_interaction_k()
        w = np.linalg.eigvalsh(h)
        np.save(file_name, w)
    bins = 500
    bin_size = (np.max(w) - np.min(w)) / bins
    w_mean = np.mean(w)
    mid = np.argmin(np.abs(w - w_mean))
    mid_spectrum_states = 500
    window = w[mid - mid_spectrum_states // 2: mid + mid_spectrum_states // 2]
    mid2 = len(w) // 2
    window2 = w[mid2 - mid_spectrum_states // 2: mid2 + mid_spectrum_states // 2]
    plt.figure()
    plt.xlabel('$E$')
    plt.ylabel('DOES($E$)')
    plt.hist(w, bins, alpha=0.7)
    plt.hist(window, round((np.max(window) - np.min(window)) / bin_size), alpha=0.7)
    plt.hist(window2, round((np.max(window2) - np.min(window2)) / bin_size), alpha=0.7)
    plt.axvline(w[mid], color='tab:orange', linestyle='dashed')
    plt.axvline(w[mid2], color='tab:green', linestyle='dashed')
    plt.tight_layout()
    plt.savefig(f'test/plots/{file_id}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
