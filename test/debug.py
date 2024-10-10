import numpy as np
import bosehubbard as bh

np.set_printoptions(linewidth=100)


def print_basis(hs: bh.HilbertSpace):
    if 'K' in hs.space:
        print(f"Translation periods: {hs.translation_periods}")
        if 'P' in hs.space:
            print(f"Number of translations after reflection: {hs.nums_translations_reflection}")
    else:
        print(f"Basis:\n{hs.basis}")
        print(f"Super basis:\n{hs.super_basis}")
        print(f"Dim: {hs.dim}")
        print(f"Super dim: {hs.super_dim}")


def test1():
    hs = bh.HilbertSpace(6, 1, space='PKN', n_tot=3, crystal_momentum=0, reflection_parity=1)
    print_basis(hs)

    mat = np.zeros((hs.dim, hs.dim), dtype=float)
    d = (1, -1)
    r = (-1, 1)
    for a in range(hs.dim):
        representative_state_a = hs.basis[a]
        translation_period_a = hs.translation_periods[a]
        num_translations_reflection_a = hs.nums_translations_reflection[a]
        if num_translations_reflection_a >= 0:
            factor_a = 2.0
        else:
            factor_a = 1.0
        for i in range(hs.num_sites):
            n_i = representative_state_a[i]
            t_state = np.copy(representative_state_a)
            t_state[i] += r[0]
            if 0 <= t_state[i] <= hs.n_max:
                for d_j in d:
                    j = i + d_j
                    j = j % hs.num_sites  # PBC IF NEEDED
                    n_j = t_state[j]
                    state_b = np.copy(t_state)
                    state_b[j] += r[1]
                    if 0 <= state_b[j] <= hs.n_max:
                        (
                            representative_state_b,
                            num_translations_b,
                            num_reflections_b
                        ) = bh.fock_representative_reflection(state_b, hs.num_sites)
                        if tuple(representative_state_b) in hs.findstate:
                            b = hs.findstate[tuple(representative_state_b)]
                            translation_period_b = hs.translation_periods[b]
                            num_translations_reflection_b = hs.nums_translations_reflection[b]
                            if num_translations_reflection_b >= 0:
                                factor_b = 2.0
                            else:
                                factor_b = 1.0
                            phase_arg = 2.0 * np.pi / hs.num_sites * hs.crystal_momentum * num_translations_b
                            elem = np.sqrt(
                                (n_i + (r[0]+1)/2.0) * (n_j + (r[1]+1)/2.0)
                                * translation_period_a * factor_b
                                / (translation_period_b * factor_a)
                            ) * hs.reflection_parity ** num_reflections_b * np.cos(phase_arg)
                            mat[a, b] += elem
                            print(a, representative_state_a, '->', state_b, ':', i, j, '->', b, representative_state_b, ':', elem)
    print(mat)


def test2():
    hs = bh.HilbertSpace(4, 1, space='PK', crystal_momentum=0, reflection_parity=1)
    print_basis(hs)

    mat = np.zeros((hs.dim, hs.dim), dtype=float)
    for r in (-1, 1):
        for a in range(hs.dim):
            representative_state_a = hs.basis[a]
            translation_period_a = hs.translation_periods[a]
            num_translations_reflection_a = hs.nums_translations_reflection[a]
            if num_translations_reflection_a >= 0:
                factor_a = 2.0
            else:
                factor_a = 1.0
            for i in range(hs.num_sites):
                n_i = representative_state_a[i]
                state_b = np.copy(representative_state_a)
                state_b[i] += r
                if 0 <= state_b[i] <= hs.n_max:
                    (
                        representative_state_b,
                        num_translations_b,
                        num_reflections_b
                    ) = bh.fock_representative_reflection(state_b, hs.num_sites)
                    if tuple(representative_state_b) in hs.findstate:
                        b = hs.findstate[tuple(representative_state_b)]
                        translation_period_b = hs.translation_periods[b]
                        num_translations_reflection_b = hs.nums_translations_reflection[b]
                        if num_translations_reflection_b >= 0:
                            factor_b = 2.0
                        else:
                            factor_b = 1.0
                        phase_arg = 2.0 * np.pi / hs.num_sites * hs.crystal_momentum * num_translations_b
                        elem = np.sqrt(
                            (n_i + (r+1)/2.0)
                            * translation_period_a * factor_b
                            / (translation_period_b * factor_a)
                        ) * hs.reflection_parity ** num_reflections_b * np.cos(phase_arg)
                        mat[a, b] += elem
                        print(a, representative_state_a, '->', state_b, ':', i, '->', b, representative_state_b, ':', elem)
    print(mat)


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()

