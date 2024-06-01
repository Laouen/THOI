from setuptools import setup, find_packages
import subprocess

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

cf_remote_version = subprocess.run(['git', 'describe', '--tags'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

setup(
    name='thoi',
    version=cf_remote_version,
    description='A Python package to compute O information in Higher Order Interactions uing batch processing',
    author='Laouen Mayal Louan Belloli, Ruben Herzog',
    author_email='laouen.belloli@gmail.com',
    url='https://github.com/Laouen/THOI',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'scipy',
        'pandas',
        'networkx'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.6',
)