import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
import h5py

plt.rcParams.update({'text.usetex': True,
                     'font.size': 18})


def calc1():
    num_sites = 10
    n_max = 1
    sym = 'KN'
    hs = bh.HilbertSpace(num_sites, n_max, sym=sym)
    with h5py.File('calc1.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=num_sites)
        param.create_dataset('M', data=n_max)
        spectra = file.create_group('spectra')
        for i, shs in enumerate(hs.subspaces):
            for j, sshs in enumerate(shs.subspaces):
                spectrum = spectra.create_group(f'{i:02d}{j:04d}')
                spectrum.create_dataset('N', data=sshs.n_tot)
                spectrum.create_dataset('K', data=sshs.crystal_momentum)
                hamiltonian_tunnel = sshs.op_hamiltonian_tunnel_k()
                hamiltonian_interaction = sshs.op_hamiltonian_interaction()
                hamiltonain = hamiltonian_tunnel + hamiltonian_interaction
                eigen_energies = np.linalg.eigvalsh(hamiltonain)
                spectrum.create_dataset('eigen_energies', data=eigen_energies)
                file.flush()


def plot1():
    eigen_energies = list()
    with h5py.File('calc1.h5', 'r') as file:
        for name in file['spectra']:
            spectrum = file[f'spectra/{name}']
            eigen_energies.append(spectrum['eigen_energies'][()])
    eigen_energies = np.concatenate(eigen_energies)

    plt.figure(dpi=100)
    plt.hist(eigen_energies, 20)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.show()


def main1():
    calc1()
    plot1()


def calc2():
    num_sites = 10
    n_max = 1
    space = 'N'
    sym = 'KN'
    n_tot = 5
    hs = bh.HilbertSpace(num_sites, n_max, space=space, sym=sym, n_tot=n_tot)
    with h5py.File('calc2.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=num_sites)
        param.create_dataset('M', data=n_max)
        param.create_dataset('N', data=n_tot)
        spectra = file.create_group('spectra')
        for i, shs in enumerate(hs.subspaces):
            spectrum = spectra.create_group(f'{i:04d}')
            spectrum.create_dataset('K', data=shs.crystal_momentum)
            hamiltonian_tunnel = shs.op_hamiltonian_tunnel_k()
            hamiltonian_interaction = shs.op_hamiltonian_interaction()
            hamiltonain = hamiltonian_tunnel + hamiltonian_interaction
            eigen_energies = np.linalg.eigvalsh(hamiltonain)
            spectrum.create_dataset('eigen_energies', data=eigen_energies)
            file.flush()


def plot2():
    eigen_energies = list()
    with h5py.File('calc2.h5', 'r') as file:
        for name in file['spectra']:
            spectrum = file[f'spectra/{name}']
            eigen_energies.append(spectrum['eigen_energies'][()])
    eigen_energies = np.concatenate(eigen_energies)

    plt.figure(dpi=100)
    plt.hist(eigen_energies, 20)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.show()


def main2():
    calc2()
    plot2()


def calc3():
    num_sites = 10
    n_max = 1
    space = 'KN'
    sym = 'KN'
    n_tot = 5
    crystal_momentum = 0
    hs = bh.HilbertSpace(num_sites, n_max, space=space, sym=sym, n_tot=n_tot, crystal_momentum=crystal_momentum)
    with h5py.File('calc3.h5', 'w') as file:
        param = file.create_group('param')
        param.create_dataset('L', data=num_sites)
        param.create_dataset('M', data=n_max)
        param.create_dataset('N', data=n_tot)
        param.create_dataset('K', data=crystal_momentum)
        spectrum = file.create_group('spectrum')
        hamiltonian_tunnel = hs.op_hamiltonian_tunnel_k()
        hamiltonian_interaction = hs.op_hamiltonian_interaction()
        hamiltonain = hamiltonian_tunnel + hamiltonian_interaction
        eigen_energies = np.linalg.eigvalsh(hamiltonain)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        file.flush()


def plot3():
    eigen_energies = list()
    with h5py.File('calc3.h5', 'r') as file:
        spectrum = file['spectrum']
        eigen_energies.append(spectrum['eigen_energies'][()])
    eigen_energies = np.concatenate(eigen_energies)

    plt.figure(dpi=100)
    plt.hist(eigen_energies, 20)
    plt.xlabel(r'$E$')
    plt.ylabel(r'$DOS$')
    plt.tight_layout()
    plt.show()


def main3():
    calc3()
    plot3()


def main():
    main1()
    main2()
    main3()

if __name__ == '__main__':
    main()
