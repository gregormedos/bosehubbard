import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
import cmasher as cmr
import h5py

plt.rcParams.update({'text.usetex': True,
                     'font.size': 18})


def exact_diagonalization(file: h5py.File,
                          group: h5py.Group,
                          hs: bh.HilbertSpace,
                          tunneling_rate: float,
                          repulsion_strength: float):
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
    if hs.periods is not None:
        group.create_dataset('periods', data=hs.periods)
    if hs.reflection_parity is not None:
        param.create_dataset('reflection_parity', data=hs.reflection_parity)
    if hs.reflection_periods is not None:
        group.create_dataset('reflection_periods', data=hs.reflection_periods)
    group.create_dataset('dim', data=hs.dim)
    group.create_dataset('basis', data=hs.basis)
    file.flush()
    if hs.subspaces is not None:
        group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            exact_diagonalization(file, group.create_group(f'subspaces/{i:04d}'), subspace, tunneling_rate, repulsion_strength)
    else:
        spectrum = group.create_group('spectrum')
        if hs.space == 'PKN':
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pk()
        elif hs.space == 'KN':
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_k()
        else:
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pbc()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def get_eigen_energies(group: h5py.Group, eigen_energies: np.ndarray, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_energies(group[f'subspaces/{subspace_name}'], eigen_energies, counter)
    else:
        sub_dim = group['dim'][()]
        sub_eigen_energies = group['spectrum/eigen_energies'][()]
        j = counter[0]
        eigen_energies[j: j + sub_dim] = sub_eigen_energies
        counter[0] += sub_dim


def get_eigen_states(group: h5py.Group, findstate: dict, eigen_states: np.ndarray, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_states(group[f'subspaces/{subspace_name}'], findstate, eigen_states, counter)
    else:
        sub_space = group['param/space'][()]
        sub_dim = group['dim'][()]
        sub_basis = group['basis'][()]
        sub_eigen_states = group['spectrum/eigen_states'][()]
        j = counter[0]
        for a in range(sub_dim):
            sub_state_a = sub_basis[a]
            sub_values_a = sub_eigen_states[a]
            if sub_space == 'PKN':
                num_sites = group['num_sites'][()]
                sub_crystal_momentum = group['crystal_momentum'][()]
                sub_reflection_parity = group['reflection_parity'][()]
                for r in range(num_sites):
                    phase_arg = -sub_crystal_momentum * r
                    t_sub_state_a = bh.fock_translation(sub_state_a, r)
                    eigen_states[findstate[tuple(t_sub_state_a)], j: j + sub_dim] += (
                            sub_values_a * complex(np.cos(phase_arg), np.sin(phase_arg)))
                    t_sub_state_a = bh.fock_reflection(t_sub_state_a)
                    eigen_states[findstate[tuple(t_sub_state_a)], j: j + sub_dim] += (
                            sub_values_a * sub_reflection_parity * complex(np.cos(phase_arg), np.sin(phase_arg)))
            elif sub_space == 'KN':
                num_sites = group['num_sites'][()]
                sub_crystal_momentum = group['crystal_momentum'][()]
                for r in range(num_sites):
                    phase_arg = -sub_crystal_momentum * r
                    t_sub_state_a = bh.fock_translation(sub_state_a, r)
                    eigen_states[findstate[tuple(t_sub_state_a)], j: j + sub_dim] += (
                            sub_values_a * complex(np.cos(phase_arg), np.sin(phase_arg)))
            else:
                eigen_states[findstate[tuple(sub_state_a)], j: j + sub_dim] += sub_values_a
        counter[0] += sub_dim


def plot_dos(file_name: str, eigen_energies: np.ndarray):
    plt.figure(dpi=300)
    plt.hist(eigen_energies, 100)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.savefig(f'{file_name}_dos.png')


def plot_states(file_name: str, dim: int, eigen_states: np.ndarray):
    plt.figure(dpi=300)
    width = 1 / dim
    colors_real = list(cmr.take_cmap_colors('viridis', dim, cmap_range=(0.14, 0.86)))
    colors_imag = list(cmr.take_cmap_colors('plasma', dim, cmap_range=(0.14, 0.86)))
    integral_numbers = np.arange(1, dim + 1, dtype=int)
    for i in range(dim):
        plt.bar(integral_numbers + (width * i - 0.5),
                np.real(eigen_states[:, i]),
                width=width,
                color=colors_real[i],
                alpha=0.5)
        plt.bar(integral_numbers + (width * i - 0.5),
                np.imag(eigen_states[:, i]),
                width=width,
                color=colors_imag[i],
                alpha=0.5)
    plt.xlabel(r'$\underline{n}$')
    plt.ylabel(r'$Re\psi,Im\psi$')
    plt.tight_layout()
    plt.savefig(f'{file_name}_states.png')


def run(file_name: str,
        tunneling_rate: float = 1.0,
        repulsion_strength: float = 1.0,
        num_sites: int = 3,
        n_max: int = 2,
        space: str = 'full',
        sym: str = None,
        n_tot: int = None,
        crystal_momentum: int = None,
        reflection_parity: int = None):
    with h5py.File(f'{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = bh.HilbertSpace(num_sites, n_max, space, sym, n_tot, crystal_momentum, reflection_parity)
        exact_diagonalization(file, group, hs, tunneling_rate, repulsion_strength)
    with h5py.File(f'{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        basis = group['basis'][()]
        eigen_energies = np.empty(dim, dtype=float)
        counter = [0]
        get_eigen_energies(group, eigen_energies, counter)
        plot_dos(file_name, eigen_energies)
        findstate = dict()
        for a in range(dim):
            findstate[tuple(basis[a])] = a
        eigen_states = np.zeros((dim, dim), dtype=complex)
        counter = [0]
        get_eigen_states(group, findstate, eigen_states, counter)
        eigen_states /= np.linalg.norm(eigen_states, axis=0)
        plot_states(file_name, dim, eigen_states)


def main():
    run('data1')
    run('data2', sym='N')
    run('data3', sym='KN')
    run('data4', sym='PKN')
    run('data5', space='N', sym='N', n_tot=3)
    run('data6', space='N', sym='KN', n_tot=3)
    run('data7', space='N', sym='PKN', n_tot=3)
    run('data8', space='KN', sym='KN', n_tot=3, crystal_momentum=0)
    run('data9', space='KN', sym='PKN', n_tot=3, crystal_momentum=0)
    run('data10', space='PKN', sym='PKN', n_tot=3, crystal_momentum=0, reflection_parity=1)


def calc1(tunneling_rate, repulsion_strength, num_sites, n_max):
    hs = bh.HilbertSpace(num_sites, n_max, sym='KN')
    with h5py.File('calc1.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        param.create_dataset('space', data=hs.space)
        param.create_dataset('sym', data=hs.sym)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        for i, shs in enumerate(hs.subspaces):
            for j, sshs in enumerate(shs.subspaces):
                spectrum = spectra.create_group(f'{i:03d}{j:03d}')
                spectrum.create_dataset('N', data=sshs.n_tot)
                spectrum.create_dataset('K', data=sshs.crystal_momentum)
                spectrum.create_dataset('dim', data=sshs.dim)
                spectrum.create_dataset('basis', data=sshs.basis)
                hamiltonian_tunnel = sshs.op_hamiltonian_tunnel_k()
                hamiltonian_interaction = sshs.op_hamiltonian_interaction()
                hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
                eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
                spectrum.create_dataset('eigen_energies', data=eigen_energies)
                spectrum.create_dataset('eigen_states', data=eigen_states)
                file.flush()


def calc2(tunneling_rate, repulsion_strength, num_sites, n_max):
    hs = bh.HilbertSpace(num_sites, n_max, sym='N')
    with h5py.File('calc2.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        for i, shs in enumerate(hs.subspaces):
            spectrum = spectra.create_group(f'{i:03d}')
            spectrum.create_dataset('N', data=shs.n_tot)
            spectrum.create_dataset('dim', data=shs.dim)
            spectrum.create_dataset('basis', data=shs.basis)
            hamiltonian_tunnel = shs.op_hamiltonian_tunnel_pbc()
            hamiltonian_interaction = shs.op_hamiltonian_interaction()
            hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
            eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
            spectrum.create_dataset('eigen_energies', data=eigen_energies)
            spectrum.create_dataset('eigen_states', data=eigen_states)
            file.flush()


def calc3(tunneling_rate, repulsion_strength, num_sites, n_max):
    hs = bh.HilbertSpace(num_sites, n_max)
    with h5py.File('calc3.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        spectrum = spectra.create_group(f'{0:03d}')
        spectrum.create_dataset('dim', data=hs.dim)
        spectrum.create_dataset('basis', data=hs.basis)
        hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pbc()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def calc4(tunneling_rate, repulsion_strength, num_sites, n_max, n_tot):
    hs = bh.HilbertSpace(num_sites, n_max, space='N', sym='KN', n_tot=n_tot)
    with h5py.File('calc4.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        param.create_dataset('N', data=hs.n_tot)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        for i, shs in enumerate(hs.subspaces):
            spectrum = spectra.create_group(f'{i:03d}')
            spectrum.create_dataset('K', data=shs.crystal_momentum)
            spectrum.create_dataset('dim', data=shs.dim)
            spectrum.create_dataset('basis', data=shs.basis)
            hamiltonian_tunnel = shs.op_hamiltonian_tunnel_k()
            hamiltonian_interaction = shs.op_hamiltonian_interaction()
            hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
            eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
            spectrum.create_dataset('eigen_energies', data=eigen_energies)
            spectrum.create_dataset('eigen_states', data=eigen_states)
            file.flush()


def calc5(tunneling_rate, repulsion_strength, num_sites, n_max, n_tot):
    hs = bh.HilbertSpace(num_sites, n_max, space='N', sym='N', n_tot=n_tot)
    with h5py.File('calc5.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        param.create_dataset('N', data=hs.n_tot)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        spectrum = spectra.create_group(f'{0:03d}')
        spectrum.create_dataset('dim', data=hs.dim)
        spectrum.create_dataset('basis', data=hs.basis)
        hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pbc()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def calc6(tunneling_rate, repulsion_strength, num_sites, n_max, n_tot, crystal_momentum):
    hs = bh.HilbertSpace(num_sites, n_max, space='KN', sym='PKN', n_tot=n_tot, crystal_momentum=crystal_momentum)
    with h5py.File('calc6.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        param.create_dataset('N', data=hs.n_tot)
        param.create_dataset('K', data=hs.crystal_momentum)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        for i, shs in enumerate(hs.subspaces):
            spectrum = spectra.create_group(f'{i:03d}')
            spectrum.create_dataset('P', data=shs.reflection_parity)
            spectrum.create_dataset('dim', data=shs.dim)
            spectrum.create_dataset('basis', data=shs.basis)
            hamiltonian_tunnel = shs.op_hamiltonian_tunnel_pk()
            hamiltonian_interaction = shs.op_hamiltonian_interaction()
            hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
            eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
            spectrum.create_dataset('eigen_energies', data=eigen_energies)
            spectrum.create_dataset('eigen_states', data=eigen_states)
            file.flush()


def calc7(tunneling_rate, repulsion_strength, num_sites, n_max, n_tot, crystal_momentum):
    hs = bh.HilbertSpace(num_sites, n_max, space='KN', sym='KN', n_tot=n_tot, crystal_momentum=crystal_momentum)
    with h5py.File('calc7.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        param.create_dataset('N', data=hs.n_tot)
        param.create_dataset('K', data=hs.crystal_momentum)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        spectrum = spectra.create_group(f'{0:03d}')
        spectrum.create_dataset('dim', data=hs.dim)
        spectrum.create_dataset('basis', data=hs.basis)
        hamiltonian_tunnel = hs.op_hamiltonian_tunnel_k()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def calc8(tunneling_rate, repulsion_strength, num_sites, n_max, n_tot, crystal_momentum, reflection_parity):
    hs = bh.HilbertSpace(num_sites, n_max, space='PKN', sym='PKN', n_tot=n_tot, crystal_momentum=crystal_momentum,
                         reflection_parity=reflection_parity)
    with h5py.File('calc8.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
        param.create_dataset('N', data=hs.n_tot)
        param.create_dataset('K', data=hs.crystal_momentum)
        param.create_dataset('P', data=hs.reflection_parity)
        file.create_dataset('dim', data=hs.dim)
        spectra = file.create_group('spectra')
        spectrum = spectra.create_group(f'{0:03d}')
        spectrum.create_dataset('dim', data=hs.dim)
        spectrum.create_dataset('basis', data=hs.basis)
        hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pk()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonian = - tunneling_rate * hamiltonian_tunnel + repulsion_strength * hamiltonian_interaction
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def plot_dos_1(file_name):
    eigen_energies = list()
    with h5py.File(f'{file_name}.h5', 'r') as file:
        for name in file['spectra']:
            spectrum = file[f'spectra/{name}']
            eigen_energies.append(spectrum['eigen_energies'][()])
    eigen_energies = np.concatenate(eigen_energies)

    plt.figure(dpi=300)
    plt.hist(eigen_energies, 100)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.savefig(f'{file_name}.png')


def plot_eigen_states_n(file_name):
    with h5py.File(f'{file_name}.h5', 'r') as file:
        num_sites = file['param/L'][()]
        n_max = file['param/M'][()]
        n_tot = file['param/N'][()]
        crystal_momentum = None
        dim = file['dim'][()]
        hs = bh.HilbertSpace(num_sites, n_max, space='N', sym='N', n_tot=n_tot)
        expanded_eigen_states = np.zeros((hs.dim, dim), dtype=complex)
        j = 0
        for spectrum_name in file[f'spectra']:
            if 'P' in file[f'spectra/{spectrum_name}']:
                reflection_parity = file[f'spectra/{spectrum_name}/P'][()]
            dim = file[f'spectra/{spectrum_name}/dim'][()]
            basis = file[f'spectra/{spectrum_name}/basis'][()]
            eigen_states = file[f'spectra/{spectrum_name}/eigen_states'][()]
            for a in range(dim):
                values = eigen_states[a]
                state_a = basis[a]
                for r in range(num_sites):
                    phase_arg = -crystal_momentum * r
                    t_state_a = bh.fock_translation(state_a, r)
                    expanded_eigen_states[hs.findstate[tuple(t_state_a)], j: j + dim] += values * complex(np.cos(phase_arg), np.sin(phase_arg))
                    if reflection_parity is not None:
                        t_state_a = bh.fock_reflection(t_state_a)
                        expanded_eigen_states[hs.findstate[tuple(t_state_a)], j: j + dim] += values * reflection_parity * complex(np.cos(phase_arg), np.sin(phase_arg))
            j += dim

    expanded_eigen_states /= np.linalg.norm(expanded_eigen_states, axis=0)
    wave_functions = hs.basis.T @ expanded_eigen_states
    wave_functions /= np.linalg.norm(wave_functions, axis=0)
    plt.figure(dpi=300)
    sites = np.arange(1, num_sites + 1, dtype=int)
    width = 1 / expanded_eigen_states.shape[1]
    colors = cmr.take_cmap_colors('plasma', dim, cmap_range=(0.14, 0.86))
    for i, color in enumerate(colors):
        plt.bar(sites + (width * i - 0.5), np.abs(wave_functions[:, i]) ** 2, width=width, color=color, alpha=0.75)
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\vert\psi\vert^{2}$')
    plt.tight_layout()
    plt.savefig(f'{file_name}_states.png')


def plot_eigen_states_kn(file_name):
    with h5py.File(f'{file_name}.h5', 'r') as file:
        num_sites = file['param/L'][()]
        n_max = file['param/M'][()]
        n_tot = file['param/N'][()]
        crystal_momentum = file['param/K'][()]
        reflection_parity = None
        dim = file['dim'][()]
        hs = bh.HilbertSpace(num_sites, n_max, space='N', sym='N', n_tot=n_tot)
        expanded_eigen_states = np.zeros((hs.dim, dim), dtype=complex)
        j = 0
        for spectrum_name in file[f'spectra']:
            if 'P' in file[f'spectra/{spectrum_name}']:
                reflection_parity = file[f'spectra/{spectrum_name}/P'][()]
            dim = file[f'spectra/{spectrum_name}/dim'][()]
            basis = file[f'spectra/{spectrum_name}/basis'][()]
            eigen_states = file[f'spectra/{spectrum_name}/eigen_states'][()]
            for a in range(dim):
                values = eigen_states[a]
                state_a = basis[a]
                for r in range(num_sites):
                    phase_arg = -crystal_momentum * r
                    t_state_a = bh.fock_translation(state_a, r)
                    expanded_eigen_states[hs.findstate[tuple(t_state_a)], j: j + dim] += values * complex(np.cos(phase_arg), np.sin(phase_arg))
                    if reflection_parity is not None:
                        t_state_a = bh.fock_reflection(t_state_a)
                        expanded_eigen_states[hs.findstate[tuple(t_state_a)], j: j + dim] += values * reflection_parity * complex(np.cos(phase_arg), np.sin(phase_arg))
            j += dim

    expanded_eigen_states /= np.linalg.norm(expanded_eigen_states, axis=0)
    wave_functions = hs.basis.T @ expanded_eigen_states
    wave_functions /= np.linalg.norm(wave_functions, axis=0)
    plt.figure(dpi=300)
    sites = np.arange(1, num_sites + 1, dtype=int)
    width = 1 / expanded_eigen_states.shape[1]
    colors = cmr.take_cmap_colors('plasma', dim, cmap_range=(0.14, 0.86))
    for i, color in enumerate(colors):
        plt.bar(sites + (width * i - 0.5), np.abs(wave_functions[:, i]) ** 2, width=width, color=color, alpha=0.75)
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\vert\psi\vert^{2}$')
    plt.tight_layout()
    plt.savefig(f'{file_name}_states.png')


def plot_eigen_states_pkn(file_name):
    with h5py.File(f'{file_name}.h5', 'r') as file:
        num_sites = file['param/L'][()]
        n_max = file['param/M'][()]
        n_tot = file['param/N'][()]
        crystal_momentum = file['param/K'][()]
        reflection_parity = file['param/P']
        dim = file['dim'][()]
        hs = bh.HilbertSpace(num_sites, n_max, space='N', sym='N', n_tot=n_tot)
        expanded_eigen_states = np.zeros((hs.dim, dim), dtype=complex)
        j = 0
        for spectrum_name in file[f'spectra']:
            dim = file[f'spectra/{spectrum_name}/dim'][()]
            basis = file[f'spectra/{spectrum_name}/basis'][()]
            eigen_states = file[f'spectra/{spectrum_name}/eigen_states'][()]
            for a in range(dim):
                values = eigen_states[a]
                state_a = basis[a]
                for r in range(num_sites):
                    phase_arg = -crystal_momentum * r
                    t_state_a = bh.fock_translation(state_a, r)
                    expanded_eigen_states[hs.findstate[tuple(t_state_a)], j: j + dim] += values * complex(np.cos(phase_arg), np.sin(phase_arg))
                    t_state_a = bh.fock_reflection(t_state_a)
                    expanded_eigen_states[hs.findstate[tuple(t_state_a)], j: j + dim] += values * reflection_parity * complex(np.cos(phase_arg), np.sin(phase_arg))
            j += dim

    expanded_eigen_states /= np.linalg.norm(expanded_eigen_states, axis=0)
    wave_functions = hs.basis.T @ expanded_eigen_states
    wave_functions /= np.linalg.norm(wave_functions, axis=0)
    plt.figure(dpi=300)
    sites = np.arange(1, num_sites + 1, dtype=int)
    width = 1 / expanded_eigen_states.shape[1]
    colors = cmr.take_cmap_colors('plasma', dim, cmap_range=(0.14, 0.86))
    for i, color in enumerate(colors):
        plt.bar(sites + (width * i - 0.5), np.abs(wave_functions[:, i]) ** 2, width=width, color=color, alpha=0.75)
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\vert\psi\vert^{2}$')
    plt.tight_layout()
    plt.savefig(f'{file_name}_states.png')


def main1():
    calc1(1.0, 1.0, 5, 2)
    plot_dos_1('calc1')
    calc2(1.0, 1.0, 5, 2)
    plot_dos_1('calc2')
    calc3(1.0, 1.0, 5, 2)
    plot_dos_1('calc3')
    calc4(1.0, 1.0, 5, 2, 3)
    plot_dos_1('calc4')
    #plot_eigen_states_n('calc4')
    calc5(1.0, 1.0, 5, 2, 3)
    plot_dos_1('calc5')
    #plot_eigen_states_n('calc5')
    calc6(1.0, 1.0, 5, 2, 3, 0)
    plot_dos_1('calc6')
    plot_eigen_states_kn('calc6')
    calc7(1.0, 1.0, 5, 2, 3, 0)
    plot_dos_1('calc7')
    plot_eigen_states_kn('calc7')
    calc8(1.0, 1.0, 5, 2, 3, 0, 1)
    plot_dos_1('calc8')
    plot_eigen_states_pkn('calc8')


if __name__ == '__main__':
    main()
