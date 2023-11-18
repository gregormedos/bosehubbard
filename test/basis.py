import bosehubbard as bh


def print_basis(hs: bh.HilbertSpace):
    print(f"Basis:\n{hs.basis}")
    print(f"Dim: {hs.dim}")
    if 'K' in hs.space:
        print(f"Representative basis:\n{hs.representative_basis}")
        print(f"Representative dim: {hs.representative_dim}")


def main():
    hs = bh.DecomposedHilbertSpace(4, 2)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='K', crystal_momentum=2)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='PK', crystal_momentum=2, reflection_parity=1)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='PK', crystal_momentum=2, reflection_parity=-1)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='N', n_tot=4)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='KN', n_tot=4, crystal_momentum=2)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='PKN', n_tot=4, crystal_momentum=2, reflection_parity=1)
    print_basis(hs)
    hs = bh.DecomposedHilbertSpace(4, 2, space='PKN', n_tot=4, crystal_momentum=2, reflection_parity=-1)
    print_basis(hs)


if __name__ == '__main__':
    main()
