from setuptools import setup

setup(
    name='PAvisualize',
    description='python scripts for visualizing MARTINI simulations',
    version='0.0',
    packages=['PAvisualize'],
    package_data={'PAvisualize': ['*']},
    include_package_data=True,
    install_requires=['numpy', 'mdtraj'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
)
