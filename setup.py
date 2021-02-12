from setuptools import setup, find_packages

# Go!
setup(
    # Module name
    name='pintsfunctest',
    version='0.0.1dev0',

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@cs.ox.ac.uk',

    # Packages and data to include
    packages=find_packages(include=('pintsfunctest', 'pintsfunctest.*')),

    # List of dependencies
    install_requires=[
        'bayesian_changepoint_detection',
        'matplotlib',
        'numpy',
        'pints',
        'tqdm',
        'ipywidgets'
    ],
)
