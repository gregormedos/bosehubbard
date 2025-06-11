import numpy as np
import bosehubbard as bh


def print_basis(hs: bh.HilbertSpace):
    if 'K' in hs.space:
        print(f"Translation periods: {hs.translation_periods}")
        if 'P' in hs.space:
            print(f"Number of translations after reflection: {hs.nums_translations_reflection}")
    print(f"Basis:\n{hs.basis}")
    print(f"Super basis:\n{hs.super_basis}")
    print(f"Dim: {hs.dim}")
    print(f"Super dim: {hs.super_dim}")


def main():
    L = 3
    M = 2
    N = 3
    Z2 = 1
    K = 0
    P = 1
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
    hs = bh.HilbertSpace(L, M)
    print("### --- full --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'N', n_tot=N)
    print("### --- N --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'Z2', n_tot_parity=Z2)
    print("### --- Z2 --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'K', crystal_momentum=K)
    print("### --- K --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'KN', n_tot=N, crystal_momentum=K)
    print("### --- KN --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'KZ2', n_tot_parity=Z2, crystal_momentum=K)
    print("### --- KZ2 --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'PKN', n_tot=N, crystal_momentum=K, reflection_parity=P)
    print("### --- PKN --- ###")
    print_basis(hs)
    hs = bh.HilbertSpace(L, M, 'PKZ2', n_tot_parity=Z2, crystal_momentum=K, reflection_parity=P)
    print("### --- PKZ2 --- ###")
    print_basis(hs)


if __name__ == '__main__':
    main()
