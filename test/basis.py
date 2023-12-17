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
    hs = bh.HilbertSpace(6, 2, space='PKN', n_tot=6, crystal_momentum=0, reflection_parity=-1)
    print_basis(hs)
    h = hs.op_hamiltonian_tunnel_pk()
    plt.figure()
    plt.imshow(h)
    plt.show()
    
    hs = bh.HilbertSpace(4, 2, space='PK', crystal_momentum=0, reflection_parity=-1)
    print_basis(hs)
    h = hs.op_hamiltonian_annihilate_create_pair_pk()
    plt.figure()
    plt.imshow(h)
    plt.show()


if __name__ == '__main__':
    main()
