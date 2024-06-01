from setuptools import setup, find_packages

setup(
    name='thoi',
    version='v0.1.0',
    description='Torch - Higher Order Interactions',
    long_description='A Python package to compute O information in Higher Order Interactions uing batch processing',
    long_description_content_type='text/x-rst',
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