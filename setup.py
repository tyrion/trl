# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='trl',
    version='0.1.0',

    description='lib to run multiple RL algorithms',
    long_description=long_description,
    url='https://github.com/tyrion/trl',

    # Author details
    author='Germano Gabbianelli',
    author_email='tyrion.mx@gmail.com',

    license='GPLv3+',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='ai rl reinforcement learning fqi openai',
    packages=['trl'],
    install_requires=[
        'numpy>=1.11.0,<1.12.0',
        'scipy>=0.18.1,<1.0', # <1.0 required for pyBrain
        'keras>=1.2.1,<2.0',
        'gym>=0.7.2',
        'scikit-learn>=0.18.1',
        'pybrain>=0.3.3',
        'matplotlib>=2.0.0',
        'h5py>=2.6.0',
        'click>=6.7',
        'colorlog>=3.1.0',
        'ifqi',
        'hyperopt',
    ],
    extras_require={
        'test': [
            'pytest>=3.0.6',
            'Numdifftools>=0.9.20',
        ],
    },
    entry_points='''
        [console_scripts]
        trl=trl.cli:main
    ''',
)
