import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
import cmasher as cmr
import h5py

plt.rcParams.update({'text.usetex': True,
                     'font.size': 18})

PRECISION = 10


def exact_diagonalization(
        file: h5py.File,
        group: h5py.Group,
        hs: bh.DecomposedHilbertSpace,
        tunneling_rate: float,
        repulsion_strength: float,
        particle_transfer_rate: float,
        pair_production_rate: float
):
    param = group.create_group('param')
    param.create_dataset('num_sites', data=hs.num_sites)
    param.create_dataset('n_max', data=hs.n_max)
    param.create_dataset('space', data=hs.space)
    group.create_dataset('basis', data=hs.basis)
    group.create_dataset('dim', data=hs.dim)
    if hs.sym is not None:
        param.create_dataset('sym', data=hs.sym)
    if hs.n_tot is not None:
        param.create_dataset('n_tot', data=hs.n_tot)
    if hs.crystal_momentum is not None:
        param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        group.create_dataset('representative_basis', data=hs.representative_basis)
        group.create_dataset('translation_periods', data=hs.translation_periods)
        group.create_dataset('representative_dim', data=hs.representative_dim)
    if hs.reflection_parity is not None:
        param.create_dataset('reflection_parity', data=hs.reflection_parity)
        group.create_dataset('reflection_translation_periods', data=hs.nums_translations_reflection)
    file.flush()
    if hs.subspaces is not None:
        group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            exact_diagonalization(
                file,
                group.create_group(f'subspaces/{i:04d}'),
                subspace,
                tunneling_rate,
                repulsion_strength,
                particle_transfer_rate,
                pair_production_rate
            )
    else:
        spectrum = group.create_group('spectrum')
        if hs.space in ('PK', 'PKN'):
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pk()
            hamiltonian_interaction = hs.op_hamiltonian_interaction_k()
            hamiltonian_annihilate_create = hs.op_hamiltonian_annihilate_create_pk()
            hamiltonian_annihilate_create_pair = hs.op_hamiltonian_annihilate_create_pair_pk()
        elif hs.space in ('K', 'KN'):
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_k()
            hamiltonian_interaction = hs.op_hamiltonian_interaction_k()
            hamiltonian_annihilate_create = hs.op_hamiltonian_annihilate_create_k()
            hamiltonian_annihilate_create_pair = hs.op_hamiltonian_annihilate_create_pair_k()
        else:
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pbc()
            hamiltonian_interaction = hs.op_hamiltonian_interaction()
            hamiltonian_annihilate_create = hs.op_hamiltonian_annihilate_create()
            hamiltonian_annihilate_create_pair = hs.op_hamiltonian_annihilate_create_pair_pbc()
        hamiltonian = (- tunneling_rate * hamiltonian_tunnel
                       + repulsion_strength * hamiltonian_interaction
                       + particle_transfer_rate * hamiltonian_annihilate_create
                       + pair_production_rate * hamiltonian_annihilate_create_pair)
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def get_eigen_energies(group: h5py.Group, eigen_energies: np.ndarray, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_energies(group[f'subspaces/{subspace_name}'], eigen_energies, counter)
    else:
        sub_space = group['param/space'][()].decode()
        if sub_space in ('K', 'KN', 'PK', 'PKN'):
            sub_dim = group['representative_dim'][()]
        else:
            sub_dim = group['dim'][()]
        sub_eigen_energies = group['spectrum/eigen_energies'][()]
        j = counter[0]
        eigen_energies[j: j + sub_dim] = sub_eigen_energies
        counter[0] += sub_dim


def get_eigen_states(group: h5py.Group, eigen_states: np.ndarray, findstate: dict, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_states(group[f'subspaces/{subspace_name}'], eigen_states, findstate, counter)
    else:
        sub_space = group['param/space'][()].decode()
        if sub_space in ('K', 'KN', 'PK', 'PKN'):
            sub_dim = group['representative_dim'][()]
            sub_basis = group['representative_basis'][()]
        else:
            sub_dim = group['dim'][()]
            sub_basis = group['basis'][()]
        sub_eigen_states = group['spectrum/eigen_states'][()]
        j = counter[0]
        for a in range(sub_dim):
            sub_state_a = sub_basis[a]
            sub_values_a = sub_eigen_states[a]
            if sub_space in ('K', 'KN'):
                num_sites = group['param/num_sites'][()]
                sub_crystal_momentum = group['param/crystal_momentum'][()]
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * sub_crystal_momentum * r
                    t_sub_state_a = bh.fock_translation(sub_state_a, r)
                    eigen_states[findstate[tuple(t_sub_state_a)], j: j + sub_dim] += (
                            sub_values_a * np.exp(1.0j * phase_arg))
            elif sub_space in ('PK', 'PKN'):
                num_sites = group['param/num_sites'][()]
                sub_crystal_momentum = group['param/crystal_momentum'][()]
                sub_reflection_parity = group['param/reflection_parity'][()]
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * sub_crystal_momentum * r
                    t_sub_state_a = bh.fock_translation(sub_state_a, r)
                    eigen_states[findstate[tuple(t_sub_state_a)], j: j + sub_dim] += (
                            sub_values_a * np.exp(1.0j * phase_arg))
                    t_sub_state_a = bh.fock_reflection(sub_state_a)
                    t_sub_state_a = bh.fock_translation(t_sub_state_a, r)
                    eigen_states[findstate[tuple(t_sub_state_a)], j: j + sub_dim] += (
                            sub_values_a * sub_reflection_parity * np.exp(1.0j * phase_arg))
            else:
                eigen_states[findstate[tuple(sub_state_a)], j: j + sub_dim] += sub_values_a
        counter[0] += sub_dim


def plot_dos(file_name: str, eigen_energies: np.ndarray):
    plt.figure(dpi=300)
    plt.hist(np.round(eigen_energies, PRECISION), 60)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.savefig(f'test/{file_name}_dos.png')
    plt.close()


def plot_states(file_name: str, dim_x: int, dim_y: int, eigen_energies: np.ndarray, eigen_states: np.ndarray):
    x = np.arange(1, dim_x + 1, dtype=int)
    colors = cmr.take_cmap_colors('viridis', dim_y, cmap_range=(0.14, 0.86))
    plt.figure(dpi=300)
    for i, color in enumerate(colors):
        plt.fill_between(x, np.real(eigen_states[:, i]) + eigen_energies[i], eigen_energies[i], color=color, alpha=0.5)
        plt.plot(x, np.real(eigen_states[:, i]) + eigen_energies[i], color=color, alpha=0.75)
    plt.xlabel(r'$\underline{n}$')
    plt.ylabel(r'$Re(\psi)$')
    plt.tight_layout()
    plt.savefig(f'test/{file_name}_states_real.png')
    plt.close()
    plt.figure(dpi=300)
    for i, color in enumerate(colors):
        plt.fill_between(x, np.imag(eigen_states[:, i]) + eigen_energies[i], eigen_energies[i], color=color, alpha=0.5)
        plt.plot(x, np.imag(eigen_states[:, i]) + eigen_energies[i], color=color, alpha=0.75)
    plt.xlabel(r'$\underline{n}$')
    plt.ylabel(r'$Im(\psi)$')
    plt.tight_layout()
    plt.savefig(f'test/{file_name}_states_imag.png')
    plt.close()


def run(
        file_name: str,
        tunneling_rate: float = 1.0,
        repulsion_strength: float = 1.0,
        particle_transfer_rate: float = 1.0,
        pair_production_rate: float = 1.0,
        num_sites: int = 4,
        n_max: int = 2,
        space: str = 'full',
        sym: str = None,
        n_tot: int = None,
        crystal_momentum: int = None,
        reflection_parity: int = None
):
    with h5py.File(f'test/{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = bh.DecomposedHilbertSpace(num_sites, n_max, space, sym, n_tot, crystal_momentum, reflection_parity)
        exact_diagonalization(file, group, hs, tunneling_rate, repulsion_strength, particle_transfer_rate, pair_production_rate)
    with h5py.File(f'test/{file_name}.h5', 'r') as file:
        group = file['data']
        basis = group['basis'][()]
        dim = group['dim'][()]
        findstate = dict()
        for a in range(dim):
            findstate[tuple(basis[a])] = a
        if space in ('K', 'KN', 'PK', 'PKN'):
            representative_dim = group['representative_dim'][()]
            eigen_energies = np.empty(representative_dim, dtype=float)
            eigen_states = np.zeros((dim, representative_dim), dtype=complex)
        else:
            eigen_energies = np.empty(dim, dtype=float)
            eigen_states = np.zeros((dim, dim), dtype=complex)
        counter = [0]
        get_eigen_energies(group, eigen_energies, counter)
        counter = [0]
        get_eigen_states(group, eigen_states, findstate, counter)
        eigen_states /= np.linalg.norm(eigen_states, axis=0)
        eigen_states = eigen_states[:, eigen_energies.argsort()]
        eigen_energies.sort()
        plot_dos(file_name, eigen_energies)
        if space in ('K', 'KN', 'PK', 'PKN'):
            plot_states(file_name, dim, representative_dim, eigen_energies, eigen_states)
        else:
            plot_states(file_name, dim, dim, eigen_energies, eigen_states)


def main():
    run('data1')
    run('data2', sym='N')
    run('data3', sym='K')
    run('data4', sym='KN')
    run('data5', sym='PK')
    run('data6', sym='PKN')
    run('data7', space='N', n_tot=4)
    run('data8', space='N', sym='KN', n_tot=4)
    run('data9', space='N', sym='PKN', n_tot=4)
    run('data10', space='K', crystal_momentum=0)
    run('data11', space='K', sym='PK', crystal_momentum=2)
    run('data12', space='KN', n_tot=5, crystal_momentum=2)
    run('data13', space='KN', sym='PKN', n_tot=5, crystal_momentum=2)
    run('data14', space='PK', crystal_momentum=2, reflection_parity=-1)
    run('data15', space='PKN', sym='PKN', n_tot=5, crystal_momentum=2, reflection_parity=-1)


if __name__ == '__main__':
    main()