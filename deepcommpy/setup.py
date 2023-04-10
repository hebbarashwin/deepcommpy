from setuptools import setup, find_packages

setup(
    name='deepcommpy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.5.1',
        'numpy==1.21.5',
        'six==1.16.0',
        'torch==1.10.2',
        'tqdm==4.62.3'
    ],
    extras_require={
        'matlab': ['matlab'],  # Add optional MATLAB dependency
    },
    entry_points={
        'console_scripts': [
            'tinyturbo-train=deepcommpy.tinyturbo.tinyturbo:main',
            # Add other entry points for crisp and nams if necessary
        ],
    },
)
