import numpy as np
import bosehubbard as bh


def main():
    for sym in (None, 'N', 'K', 'KN', 'PK', 'PKN'):
        test(sym)
    for sym in ('N', 'KN', 'PKN'):
        test(sym, space='N', N=6)


def foo(hs: bh.DecomposedHilbertSpace, dims: list):
    print(hs.space)
    if hs.subspaces is not None:
        for subspace in hs.subspaces:
            foo(subspace, dims)
    else:
        dims.append(hs.dim)


def test(sym, space='full', L=6, M=2, N=None):
    hs = bh.DecomposedHilbertSpace(L, M, space=space, sym=sym, n_tot=N)
    dims = []
    foo(hs, dims)
    print(dims, ":", sum(dims), '\n')


if __name__ == '__main__':
    main()
