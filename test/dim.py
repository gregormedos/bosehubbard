import numpy as np
import bosehubbard as bh


def main1(L: int, M: int):
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


def main2(L: int, M: int, N: int):
    sector_to_dimension = {}
    dim_N = bh.dim_nblock(L, N, M)
    sector_N = sector_to_dimension[('N', N, dim_N)] = {}
    basis_N = bh.gen_basis_nblock(L, N, dim_N, M)
    for K in range(L):
        representative_basis_K, trans, dim_K = bh.gen_representative_basis_kblock(basis_N, L, K)
        sector_K = sector_N[('KN', K, N,  dim_K)] = {}
        if (K == 0) or (L % 2 == 0 and K == L // 2):
            for P in (1, -1):
                representative_basis_P, trans, trans_refl, dim_P = bh.gen_representative_basis_pkblock(basis_N, L, K, P)
                sector_P = sector_K[('PKN', P, K, N, dim_P)] = {}
    print(sector_to_dimension)
    def get_end_nodes(map: dict, end_nodes: list = None):
        if end_nodes is None:
            end_nodes = []
        for node, next_node in map.items():
            if len(next_node) == 0:
                end_nodes.append(node)
            else:
                get_end_nodes(next_node, end_nodes)
        return end_nodes
    sectors = get_end_nodes(sector_to_dimension)
    print(sectors)


def main():
    main1(4, 4)
    main2(4, 4, 4)


if __name__ == '__main__':
    main()
