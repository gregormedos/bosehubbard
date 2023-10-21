import bosehubbard as bh


def print_basis(hs: bh.HilbertSpace):
    print(f"Basis:\n{hs.basis}")
    print(f"Dim: {hs.dim}")
    if 'K' in hs.space:
        print(f"Representative basis:\n{hs.representative_basis}")
        print(f"Representative dim: {hs.representative_dim}")


hs = bh.HilbertSpace(5, 2)
print_basis(hs)
hs = bh.HilbertSpace(5, 2, space='N', n_tot=3)
print_basis(hs)
hs = bh.HilbertSpace(5, 2, space='K', crystal_momentum=0)
print_basis(hs)
hs = bh.HilbertSpace(5, 2, space='KN', n_tot=3, crystal_momentum=0)
print_basis(hs)
hs = bh.HilbertSpace(5, 2, space='PK', crystal_momentum=0, reflection_parity=1)
print_basis(hs)
hs = bh.HilbertSpace(5, 2, space='PKN', n_tot=3, crystal_momentum=0, reflection_parity=1)
print_basis(hs)
