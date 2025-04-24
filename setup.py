from setuptools import setup, find_packages
import os 

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requirements = list(f.read().splitlines())

setup(
    name='hessml',

    author='Gereon Feldmann',

    author_email='feldmann@pc.rwth-aachen.de',

    description='A tool for learning and predicting Hessian matrices based on semiemprical electronic structure theory atomic quantities derived from xtb',

    url='https://git.rwth-aachen.de/bannwarthlab/Hessian_ML',

    version='0.0.1',

    install_requires=install_requirements,

    packages = find_packages(),

    python_requires='>=3.6',

    entry_points={
        'console_scripts': [
            'hessml=hess_ml.__main__:main'
        ]
    }
)