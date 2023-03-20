from setuptools import setup, find_packages


setup(
    name='hessml',

    author='Gereon Feldmann',

    author_email='feldmann@pc.rwth-aachen.de',

    description='A tool for learning and predicting Hessian matrices based on semiemprical electronic structure theory atomic quantities derived from xtb',

    url='https://git.rwth-aachen.de/bannwarthlab/Hessian_ML',

    version='0.0.1',

    packages=find_packages(include = ['src']),

    entry_points={
        'console_scripts': [
            'hessml=src.main:main'
        ]
    }
)