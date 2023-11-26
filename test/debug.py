import numpy as np
import bosehubbard as bh


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
    d = (1, -1)
    r = (-1, 1)
    for a in range(hs.representative_dim):
        representative_state_a = hs.representative_basis[a]
        for i in range(hs.num_sites):
            t_state = np.copy(representative_state_a)
            t_state[i] += r[0]
            if 0 <= t_state[i] <= hs.n_max:
                for d_j in d:
                    j = i + d_j
                    j = j % hs.num_sites  # PBC IF NEEDED
                    state_b = np.copy(t_state)
                    state_b[j] += r[1]
                    if 0 <= state_b[j] <= hs.n_max:
                        (
                            representative_state_b,
                            num_translations_b,
                            num_reflections_b
                        ) = bh.fock_representative_reflection(state_b, hs.num_sites)
                        if tuple(representative_state_b) in hs.representative_findstate:
                            b = hs.representative_findstate[tuple(representative_state_b)]
                            print(a, representative_state_a, '->', state_b, '->', b, representative_state_b, num_reflections_b, num_translations_b)


if __name__ == '__main__':
    main()

