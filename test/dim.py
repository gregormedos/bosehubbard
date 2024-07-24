import numpy as np
import bosehubbard as bh


def main():
    L = 4
    M = 4
    dim = bh.dim_full(L, M)
    dim_N = []
    for N in range(M * L + 1):
        dim_N.append(bh.dim_nblock(L, N, M))
        basis_N = bh.gen_basis_nblock(L, N, dim_N[-1], M)
        dim_K = []
        for K in range(L):
            representative_basis_K, trans, dim_K_ = bh.gen_representative_basis_kblock(basis_N, L, K)
            dim_K.append(dim_K_)
            if (K == 0) or (L % 2 == 0 and K == L // 2):
                dim_P = []
                for P in (1, -1):
                    representative_basis_P, trans, trans_refl, dim_P_ = bh.gen_representative_basis_pkblock(basis_N, L, K, P)
                    dim_P.append(dim_P_)
                print(dim_K[-1], sum(dim_P))
        print(dim_N[-1], sum(dim_K))
    print(dim, sum(dim_N))


if __name__ == '__main__':
    main()
