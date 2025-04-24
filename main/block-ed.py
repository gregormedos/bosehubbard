import numpy as np
import json, h5py
import bosehubbard as bh


HAMILTONIAN_DICT = {
    'PK': {
        't': bh.HilbertSpace.op_hamiltonian_tunnel_pk,
        'U': bh.HilbertSpace.op_hamiltonian_interaction,
        'V1': bh.HilbertSpace.op_hamiltonian_annihilate_create_pk,
        'V2': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pk
    },
    'K': {
        't': bh.HilbertSpace.op_hamiltonian_tunnel_k,
        'U': bh.HilbertSpace.op_hamiltonian_interaction,
        'V1': bh.HilbertSpace.op_hamiltonian_annihilate_create_k,
        'V2': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_k
    },
    'None': {
        't': bh.HilbertSpace.op_hamiltonian_tunnel_pbc,
        'U': bh.HilbertSpace.op_hamiltonian_interaction,
        'V1': bh.HilbertSpace.op_hamiltonian_annihilate_create,
        'V2': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc
    }
}


def main():
    with open('input.json', 'r') as f:
        params = json.load(f)
    block_exact_diagonalization(**params)


def block_exact_diagonalization(
        L: int,
        M: int,
        terms: dict,
        keepvecs: int,
        space: str = 'full',
        sym: str = None,
        N: int = None,
        K: int = None,
        P: int = None
):
    with h5py.File(f'output.h5', 'w') as file:
        group = file.create_group('data')
        hs = bh.DecomposedHilbertSpace(num_sites=L, n_max=M, space=space, sym=sym, n_tot=N, crystal_momentum=K, reflection_parity=P)
        _block_exact_diagonalization(file, group, hs, terms, keepvecs)


def _block_exact_diagonalization(
        file: h5py.File,
        group: h5py.Group,
        hs: bh.DecomposedHilbertSpace,
        terms: dict,
        keepvecs: int
):
    if hs.subspaces is not None:
        subspaces = group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            if subspace.dim > 0:
                _block_exact_diagonalization(
                    file,
                    subspaces.create_group(f'{i:04d}'),
                    subspace,
                    terms,
                    keepvecs
                )
    else:
        param = group.create_group('param')
        param.create_dataset('num_sites', data=hs.num_sites)
        param.create_dataset('n_max', data=hs.n_max)
        param.create_dataset('space', data=hs.space)
        if hs.sym is not None:
            param.create_dataset('sym', data=hs.sym)
        if hs.n_tot is not None:
            param.create_dataset('n_tot', data=hs.n_tot)
        if hs.crystal_momentum is not None:
            param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        if hs.reflection_parity is not None:
            param.create_dataset('reflection_parity', data=hs.reflection_parity)
        file.flush()

        spectrum = group.create_group('spectrum')
        keepvecs = min(hs.dim, keepvecs)
        if hs.space in {'PK', 'PKN'}:
            hamiltonians = HAMILTONIAN_DICT['PK']
        elif hs.space in ('K', 'KN'):
            hamiltonians = HAMILTONIAN_DICT['K']
        else:
            hamiltonians = HAMILTONIAN_DICT['None']
        hamiltonian = np.sum([rate * hamiltonians[term](hs) for term, rate in terms.items()], axis=0)
        vals, r_vecs = np.linalg.eigh(hamiltonian)
        kept_vals = vals[(hs.dim - keepvecs) // 2: (hs.dim + keepvecs) // 2]
        kept_r_vecs = r_vecs[:, (hs.dim - keepvecs) // 2: (hs.dim + keepvecs) // 2]
        if hs.space in {'KN', 'K'}:
            kept_vecs = np.zeros((hs.super_dim, keepvecs), dtype=complex)
            for a in range(hs.dim):
                representative_state_a = hs.basis[a]
                translation_period_a = hs.translation_periods[a]
                amplitudes_a = kept_r_vecs[a]
                normalization_a = np.sqrt(translation_period_a) / hs.num_sites
                for r in range(hs.num_sites):
                    phase_arg = -2.0 * np.pi / hs.num_sites * hs.crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    kept_vecs[hs.super_findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
        elif hs.space in {'PKN', 'PK'}:
            kept_vecs = np.zeros((hs.super_dim, keepvecs), dtype=complex)
            for a in range(hs.dim):
                representative_state_a = hs.basis[a]
                translation_period_a = hs.translation_periods[a]
                num_translations_reflection_a = hs.nums_translations_reflection[a]
                amplitudes_a = kept_r_vecs[a]
                if num_translations_reflection_a == -1:
                    normalization_a = np.sqrt(2.0 * translation_period_a) / (2.0 * hs.num_sites)
                else:
                    normalization_a = np.sqrt(translation_period_a) / hs.num_sites
                for r in range(hs.num_sites):
                    phase_arg = -2.0 * np.pi / hs.num_sites * hs.crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    kept_vecs[hs.super_findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                    if num_translations_reflection_a == -1:
                        t_representative_state_a = np.flipud(t_representative_state_a)
                        kept_vecs[hs.super_findstate[tuple(t_representative_state_a)], :] += normalization_a * hs.reflection_parity * bloch_wave * amplitudes_a
            else:
                kept_vecs = kept_r_vecs
        spectrum.create_dataset('dim', data=hs.dim)
        spectrum.create_dataset('keepvecs', data=keepvecs)
        spectrum.create_dataset('all_vals', data=vals)
        spectrum.create_dataset('kept_vals', data=kept_vals)
        spectrum.create_dataset('kept_vecs', data=kept_vecs)
        file.flush()


if __name__ == '__main__':
    main()
