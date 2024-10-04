from setuptools import setup, find_packages

setup(
    name='bosehubbard',
    version='0.1',
    description='1D Bose-Hubbard model',
    author='Gregor Medoš',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "h5py"],
)
