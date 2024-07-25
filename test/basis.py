import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt


def print_basis(hs: bh.HilbertSpace):
    if 'K' in hs.space:
        print(f"Representative basis:\n{hs.representative_basis}")
        print(f"Representative dim: {hs.representative_dim}")
        print(f"Translation periods: {hs.translation_periods}")
        if 'P' in hs.space:
            print(f"Number of translations after reflection: {hs.nums_translations_reflection}")
    else:
        print(f"Basis:\n{hs.basis}")
        print(f"Dim: {hs.dim}")


def main():
    L = 3
    for trans in range(L):
        state = np.roll([1, 2, 3], trans)
        print(state)
        r_state, r = bh.fock_representative(state, L)
        print(r_state, r)
        R = bh.fock_checkstate(state, L, 0)
        print(R)
        r_state, r, l = bh.fock_representative_reflection(state, L)
        print(r_state, r, l)
        R, m = bh.fock_checkstate_reflection(state, L, 0)
        print(R, m)


if __name__ == '__main__':
    main()
