import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
import h5py

plt.rcParams.update({'text.usetex': True,
                     'font.size': 18})


def calc1(tunneling_rate, repulsion_strength, num_sites, n_max):
    hs = bh.HilbertSpace(num_sites, n_max, sym='KN')
    with h5py.File('calc1.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=hs.num_sites)
        param.create_dataset('M', data=hs.n_max)
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


def plot_dos(file_name):
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


def calc(t: float, U: float, num_sites: int, n_max: int, n_tot: int, crystal_momentum: int, reflection_parity: int = None):
    if crystal_momentum == 0 or num_sites % 2 == 0 and crystal_momentum == num_sites // 2:
        space = 'PKN'
        sym = 'PKN'
    else:
        space = 'KN'
        sym = 'KN'
    hs = bh.HilbertSpace(num_sites, n_max, space=space, sym=sym, n_tot=n_tot, crystal_momentum=crystal_momentum, reflection_parity=reflection_parity)
    with h5py.File('calc.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=num_sites)
        param.create_dataset('M', data=n_max)
        param.create_dataset('N', data=n_tot)
        param.create_dataset('K', data=crystal_momentum)
        if reflection_parity is not None:
            param.create_dataset('P', data=reflection_parity)
        file.create_dataset('dim', data=hs.dim)
        file.create_dataset('basis', data=hs.basis)
        spectrum = file.create_group('spectrum')
        hamiltonian_tunnel = hs.op_hamiltonian_tunnel_k()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonain = -t * hamiltonian_tunnel + U * hamiltonian_interaction
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonain)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)


def plot_eigen_energies():
    with h5py.File('calc.h5', 'r') as file:
        dim = file['dim'][()]
        spectrum = file['spectrum']
        eigen_energies = spectrum['eigen_energies'][()]

    plt.figure(dpi=100)
    plt.hist(eigen_energies, 20)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.show()


def plot_eigen_state(ii):
    with h5py.File('calc.h5', 'r') as file:
        num_sites = file['param/L'][()]
        n_max = file['param/M'][()]
        n_tot = file['param/N'][()]
        crystal_momentum = file['param/K'][()]
        if 'param/P' in file:
            reflection_parity = file['param/P'][()]
        else:
            reflection_parity = None
        dim = file['dim'][()]
        basis = file['basis'][()]
        spectrum = file['spectrum']
        eigen_state = spectrum['eigen_states'][()][:, ii]

    hs = bh.HilbertSpace(num_sites, n_max, space='N', sym='N', n_tot=n_tot)
    state = np.zeros(hs.dim, dtype=complex)
    for a in range(dim):
        value = eigen_state[a]
        state_a = basis[a]
        for r in range(num_sites):
            phase_arg = -crystal_momentum * r
            t_state_a = bh.fock_translation(state_a, r)
            state[hs.findstate[tuple(t_state_a)]] += value * complex(np.cos(phase_arg),
                                                                     np.sin(phase_arg))
            if reflection_parity is not None:
                t_state_a = bh.fock_reflection(t_state_a)
                state[hs.findstate[tuple(t_state_a)]] += value * reflection_parity * complex(np.cos(phase_arg),
                                                                                             np.sin(phase_arg))
    state /= np.linalg.norm(state)
    wave_function = state @ hs.basis
    wave_function /= np.linalg.norm(wave_function)

    plt.figure(dpi=100)
    sites = np.arange(1, num_sites + 1, dtype=int)
    plt.bar(sites, np.real(wave_function))
    plt.bar(sites, np.imag(wave_function))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$Re(\psi),Im(\psi)$')
    plt.tight_layout()
    plt.show()


def main4():
    calc(1.0, 0.0, 12, 1, 1, 6, -1)
    plot_eigen_energies()
    plot_eigen_state(0)


def main():
    calc1(1.0, 1.0, 5, 2)
    plot_dos('calc1')
    calc2(1.0, 1.0, 5, 2)
    plot_dos('calc2')
    calc3(1.0, 1.0, 5, 2)
    plot_dos('calc3')
    calc4(1.0, 1.0, 5, 2, 3)
    plot_dos('calc4')
    calc5(1.0, 1.0, 5, 2, 3)
    plot_dos('calc5')
    calc6(1.0, 1.0, 5, 2, 3, 0)
    plot_dos('calc6')
    calc7(1.0, 1.0, 5, 2, 3, 0)
    plot_dos('calc7')
    calc8(1.0, 1.0, 5, 2, 3, 0, 1)
    plot_dos('calc8')


if __name__ == '__main__':
    main()
