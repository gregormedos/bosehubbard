import numpy as np
import h5py
from .model import *


HAMILTONIAN_DICT = {
    'PK': {
        't': HilbertSpace.op_hamiltonian_tunnel_pk,
        'U': HilbertSpace.op_hamiltonian_interaction_k,
        'V1': HilbertSpace.op_hamiltonian_annihilate_create_pk,
        'V2': HilbertSpace.op_hamiltonian_annihilate_create_pair_pk
    },
    'K': {
        't': HilbertSpace.op_hamiltonian_tunnel_k,
        'U': HilbertSpace.op_hamiltonian_interaction_k,
        'V1': HilbertSpace.op_hamiltonian_annihilate_create_k,
        'V2': HilbertSpace.op_hamiltonian_annihilate_create_pair_k
    },
    'None': {
        't': HilbertSpace.op_hamiltonian_tunnel_pbc,
        'U': HilbertSpace.op_hamiltonian_interaction,
        'V1': HilbertSpace.op_hamiltonian_annihilate_create,
        'V2': HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc
    }
}


def run_decomposed(
        dir_name: str,
        file_name: str,
        L: int,
        M: int,
        terms: tuple,
        space: str = 'full',
        sym: str = None,
        N: int = None,
        K: int = None,
        P: int = None
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = DecomposedHilbertSpace(num_sites=L, n_max=M, space=space, sym=sym, n_tot=N, crystal_momentum=K, reflection_parity=P)
        print(hs.dim)
        exact_diagonalization_decomposed(file, group, hs, terms)


def run(
        dir_name: str,
        file_name: str,
        L: int,
        M: int,
        terms: tuple,
        space: str = 'full',
        N: int = None,
        K: int = None,
        P: int = None
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = HilbertSpace(num_sites=L, n_max=M, space=space, n_tot=N, crystal_momentum=K, reflection_parity=P)
        print(hs.dim)
        exact_diagonalization(file, group, hs, terms)


def read_eigen_energies_decomposed(
        dir_name: str,
        file_name: str
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        eigen_energies = np.empty(dim, dtype=float)
        counter = [0]
        get_eigen_energies_decomposed(group, eigen_energies, counter)
        return eigen_energies
    

def read_eigen_states_decomposed(
        dir_name: str,
        file_name: str
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        basis = group['basis'][()]
        findstate = dict()
        for a in range(dim):
            findstate[tuple(basis[a])] = a
        eigen_states = np.empty((dim, dim), dtype=complex)
        counter = [0]
        get_eigen_states_decomposed(group, eigen_states, findstate, counter, dim)
        return eigen_states


def read_mid_spectrum_eigen_energies_decomposed(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_energies = []
        dim_sectors = []
        get_mid_spectrum_eigen_energies_decomposed(group, eigen_energies, dim_sectors, num)
        return eigen_energies, dim_sectors


def read_mid_spectrum_eigen_states_decomposed(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        basis = group['basis'][()]
        findstate = dict()
        for a in range(dim):
            findstate[tuple(basis[a])] = a
        eigen_states = []
        dim_sectors = []
        get_mid_spectrum_eigen_states_decomposed(group, eigen_states, dim_sectors, findstate, dim, num)
        return eigen_states, dim_sectors


def read_eigen_energies(
        dir_name: str,
        file_name: str      
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_energies = get_eigen_energies(group)
        return eigen_energies


def read_eigen_states(
        dir_name: str,
        file_name: str
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_states = get_eigen_states(group)
        return eigen_states


def read_mid_spectrum_eigen_energies(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_energies = get_mid_spectrum_eigen_energies(group, num)
        return eigen_energies


def read_mid_spectrum_eigen_states(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_states = get_mid_spectrum_eigen_states(group, num)
        return eigen_states


def exact_diagonalization_decomposed(
        file: h5py.File,
        group: h5py.Group,
        hs: DecomposedHilbertSpace,
        terms: tuple
):
    param = group.create_group('param')
    param.create_dataset('num_sites', data=hs.num_sites)
    param.create_dataset('n_max', data=hs.n_max)
    param.create_dataset('space', data=hs.space)
    group.create_dataset('dim', data=hs.dim)
    group.create_dataset('basis', data=hs.basis)
    group.create_dataset('super_dim', data=hs.super_dim)
    group.create_dataset('super_basis', data=hs.super_dim)
    if hs.sym is not None:
        param.create_dataset('sym', data=hs.sym)
    if hs.n_tot is not None:
        param.create_dataset('n_tot', data=hs.n_tot)
    if hs.crystal_momentum is not None:
        param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        group.create_dataset('translation_periods', data=hs.translation_periods)
    if hs.reflection_parity is not None:
        param.create_dataset('reflection_parity', data=hs.reflection_parity)
        group.create_dataset('num_translations_reflection', data=hs.nums_translations_reflection)
    file.flush()
    if hs.subspaces is not None:
        group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            if subspace.dim > 0:
                exact_diagonalization_decomposed(
                    file,
                    group.create_group(f'subspaces/{i:04d}'),
                    subspace,
                    terms
                )
    else:
        spectrum = group.create_group('spectrum')
        if hs.space in {'PK', 'PKN'}:
            hamiltonians = HAMILTONIAN_DICT['PK']
        elif hs.space in ('K', 'KN'):
            hamiltonians = HAMILTONIAN_DICT['K']
        else:
            hamiltonians = HAMILTONIAN_DICT['None']
        hamiltonian = np.sum([rate * hamiltonians[term](hs) for term, rate in terms], axis=0)
        print(hamiltonian.shape)
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def exact_diagonalization(
        file: h5py.File,
        group: h5py.Group,
        hs: HilbertSpace,
        terms: tuple
):
    param = group.create_group('param')
    param.create_dataset('num_sites', data=hs.num_sites)
    param.create_dataset('n_max', data=hs.n_max)
    param.create_dataset('space', data=hs.space)
    group.create_dataset('dim', data=hs.dim)
    group.create_dataset('basis', data=hs.basis)
    group.create_dataset('super_dim', data=hs.super_dim)
    group.create_dataset('super_basis', data=hs.super_basis)
    if hs.n_tot is not None:
        param.create_dataset('n_tot', data=hs.n_tot)
    if hs.crystal_momentum is not None:
        param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        group.create_dataset('translation_periods', data=hs.translation_periods)
    if hs.reflection_parity is not None:
        param.create_dataset('reflection_parity', data=hs.reflection_parity)
        group.create_dataset('num_translations_reflection', data=hs.nums_translations_reflection)
    file.flush()
    spectrum = group.create_group('spectrum')
    if hs.space in {'PK', 'PKN'}:
        hamiltonians = HAMILTONIAN_DICT['PK']
    elif hs.space in ('K', 'KN'):
        hamiltonians = HAMILTONIAN_DICT['K']
    else:
        hamiltonians = HAMILTONIAN_DICT['None']
    hamiltonian = np.sum([rate * hamiltonians[term](hs) for term, rate in terms], axis=0)
    print(hamiltonian.shape)
    eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
    spectrum.create_dataset('eigen_energies', data=eigen_energies)
    spectrum.create_dataset('eigen_states', data=eigen_states)
    file.flush()


def get_eigen_energies_decomposed(group: h5py.Group, eigen_energies: np.ndarray, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_energies_decomposed(group[f'subspaces/{subspace_name}'], eigen_energies, counter)
    else:
        j = counter[0]
        dim_sector = group['dim'][()]
        counter[0] += dim_sector
        eigen_energies_sector = group['spectrum/eigen_energies'][()]
        eigen_energies[j: j + dim_sector] = eigen_energies_sector


def get_eigen_states_decomposed(group: h5py.Group, eigen_states: np.ndarray, findstate: dict, counter: list, dim: int):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_states_decomposed(group[f'subspaces/{subspace_name}'], eigen_states, findstate, counter, dim)
    else:
        j = counter[0]
        subspace = group['param/space'][()].decode()
        if subspace in {'KN', 'K'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            representative_basis = group['basis'][()]
            translation_periods = group['translation_periods'][()]
            representative_dim = group['dim'][()]
            counter[0] += representative_dim
            representative_eigen_states = group['spectrum/eigen_states'][()]
            eigen_states_sector = np.zeros((dim, representative_dim), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                amplitudes_a = representative_eigen_states[a]
                normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
            eigen_states[:, j: j + representative_dim] = eigen_states_sector
        elif subspace in {'PKN', 'PK'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            reflection_parity = group['param/reflection_parity'][()]
            representative_basis = group['basis'][()]
            translation_periods = group['translation_periods'][()]
            num_translations_reflection = group['num_translations_reflection'][()]
            representative_dim = group['dim'][()]
            counter[0] += representative_dim
            representative_eigen_states = group['spectrum/eigen_states'][()]
            eigen_states_sector = np.zeros((dim, representative_dim), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                num_translations_reflection_a = num_translations_reflection[a]
                amplitudes_a = representative_eigen_states[a]
                if num_translations_reflection_a == -1:
                    normalization_a = np.sqrt(2.0 * translation_period_a) / (2.0 * num_sites)
                else:
                    normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                    if num_translations_reflection_a == -1:
                        t_representative_state_a = np.flipud(t_representative_state_a)
                        eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a
            eigen_states[:, j: j + representative_dim] = eigen_states_sector
        else:
            dim_sector = group['dim'][()]
            counter[0] += dim_sector
            eigen_states_sector = group['spectrum/eigen_states'][()]
            eigen_states[:, j: j + dim_sector] = eigen_states_sector


def get_mid_spectrum_eigen_energies_decomposed(group: h5py.Group, eigen_energies: list, dim_sectors: list, num: int):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_mid_spectrum_eigen_energies_decomposed(group[f'subspaces/{subspace_name}'], eigen_energies, dim_sectors, num)
    else:
        dim_sector = group['dim'][()]
        eigen_energies_sector = group['spectrum/eigen_energies'][(dim_sector - num) // 2: (dim_sector + num) // 2]
        eigen_energies.append(eigen_energies_sector)
        dim_sectors.append(dim_sector)


def get_mid_spectrum_eigen_states_decomposed(group: h5py.Group, eigen_states: list, dim_sectors: list, findstate: dict, dim: int, num: int):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_mid_spectrum_eigen_states_decomposed(group[f'subspaces/{subspace_name}'], eigen_states, dim_sectors, findstate, dim, num)
    else:
        subspace = group['param/space'][()].decode()
        if subspace in {'KN', 'K'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            representative_basis = group['basis'][()]
            translation_periods = group['translation_periods'][()]
            representative_dim = group['dim'][()]
            representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
            eigen_states_sector = np.zeros((dim, num), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                amplitudes_a = representative_eigen_states[a]
                normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
            eigen_states.append(eigen_states_sector)
            dim_sectors.append(representative_dim)
        elif subspace in {'PKN', 'PK'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            reflection_parity = group['param/reflection_parity'][()]
            representative_basis = group['basis'][()]
            translation_periods = group['translation_periods'][()]
            num_translations_reflection = group['num_translations_reflection'][()]
            representative_dim = group['dim'][()]
            representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
            eigen_states_sector = np.zeros((dim, num), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                num_translations_reflection_a = num_translations_reflection[a]
                amplitudes_a = representative_eigen_states[a]
                if num_translations_reflection_a == -1:
                    normalization_a = np.sqrt(2.0 * translation_period_a) / (2.0 * num_sites)
                else:
                    normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                    if num_translations_reflection_a == -1:
                        t_representative_state_a = np.flipud(t_representative_state_a)
                        eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a
            eigen_states.append(eigen_states_sector)
            dim_sectors.append(representative_dim)
        else:
            dim_sector = group['dim'][()]
            eigen_states_sector = group['spectrum/eigen_states'][:, (dim_sector - num) // 2: (dim_sector + num) // 2]
            eigen_states.append(eigen_states_sector)
            dim_sectors.append(dim_sector)


def get_eigen_energies(group: h5py.Group):
    eigen_energies = group['spectrum/eigen_energies'][()]
    return eigen_energies


def get_eigen_states(group: h5py.Group):
    space = group['param/space'][()].decode()
    dim = group['super_dim'][()]
    basis = group['super_basis'][()]
    findstate = dict()
    for a in range(dim):
        findstate[tuple(basis[a])] = a
    if space in {'KN', 'K'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        representative_basis = group['basis'][()]
        translation_periods = group['translation_periods'][()]
        representative_dim = group['dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][()]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, representative_dim), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
    elif space in {'PKN', 'PK'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        reflection_parity = group['param/reflection_parity'][()]
        representative_basis = group['basis'][()]
        translation_periods = group['translation_periods'][()]
        num_translations_reflection = group['num_translations_reflection'][()]
        representative_dim = group['dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][()]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, representative_dim), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            num_translations_reflection_a = num_translations_reflection[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            if num_translations_reflection_a == -1:
                normalization_a /= np.sqrt(2.0)
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                if num_translations_reflection_a == -1:
                    t_representative_state_a = np.flipud(t_representative_state_a)
                    eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a
    else:
        eigen_states = group['spectrum/eigen_states'][()]
    return eigen_states


def get_mid_spectrum_eigen_energies(group: h5py.Group, num: int):
    dim = group['dim'][()]
    eigen_energies = group['spectrum/eigen_energies'][(dim - num) // 2: (dim + num) // 2]
    return eigen_energies


def get_mid_spectrum_eigen_states(group: h5py.Group, num: int):
    space = group['param/space'][()].decode()
    dim = group['super_dim'][()]
    basis = group['super_basis'][()]
    findstate = dict()
    for a in range(dim):
        findstate[tuple(basis[a])] = a
    if space in {'KN', 'K'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        representative_basis = group['basis'][()]
        translation_periods = group['translation_periods'][()]
        representative_dim = group['dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, num), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
    elif space in {'PKN', 'PK'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        reflection_parity = group['param/reflection_parity'][()]
        representative_basis = group['basis'][()]
        translation_periods = group['translation_periods'][()]
        num_translations_reflection = group['num_translations_reflection'][()]
        representative_dim = group['dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, num), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            num_translations_reflection_a = num_translations_reflection[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            if num_translations_reflection_a == -1:
                normalization_a /= np.sqrt(2.0)
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                if num_translations_reflection_a == -1:
                    t_representative_state_a = np.flipud(t_representative_state_a)
                    eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a

    else:
        eigen_states = group['spectrum/eigen_states'][:, (dim - num) // 2: (dim + num) // 2]
    return eigen_states
