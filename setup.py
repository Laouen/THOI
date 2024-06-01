from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='thoi',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='Torch - Higher Order Interactions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Laouen Mayal Louan Belloli, Ruben Herzog',
    author_email='laouen.belloli@gmail.com',
    url='https://github.com/Laouen/THOI',
    packages=find_packages(),
    install_requires=[
        'numpy',
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