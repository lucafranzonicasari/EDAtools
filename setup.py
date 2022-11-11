from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='EDAtools',
    url='https://github.com/lucfranzoni/EDAtools.git',
    author='Luca Franzoni',
    author_email='luca.franzoni.casari@gmail.com',
    # Needed to actually package something
    # packages=['feature_importance'],   ## qui vanno inclusi solo ed esclusivamente i sotto-pacchetti, i moduli (ovvero i files python: 1 file --> 1 1 modulo) no
    # Needed for dependencies
    install_requires=[
        'pandas',
        'sklearn',
        'numpy',
        'matplotlib',
        'numpy',
        'ppscore',
        'scipy',
        'tqdm'
    ],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An pandas.DataFrame accessor to make easier Exploratory Data Analysis',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.md').read(),
)