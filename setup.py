from setuptools import find_packages, setup

DESC = 'A lightweight python radition transport code.'

setup(
    name='BART-Lite',
    version='0.0',
    description=DESC,
    url='http://github.com/mzweig/gallo',
    author='Marissa Ramirez Zweiger',
    author_email='mzweig@berkeley.edu',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    packages=find_packages(),
)
