from setuptools import setup, find_packages

setup(
    name='scUNAGI',
    version='0.5.1',
    long_description=open('README.md').read(),
    packages=find_packages(),
    package_data={'UNAGI': ['data/*.npy','data/*.txt']},
    long_description_content_type='text/markdown',
    install_requires=[
        'pyro-ppl>=1.8.6',
        'scanpy>=1.9.5',
        'anndata==0.8.0',
        'matplotlib>=3.7.1',
    ],
    include_package_data=True,
    author='Yumin Zheng',
    author_email='yumin.zheng@mail.mcgill.ca',
    description='UNAGI: Deep Generative Model for Deciphering Cellular Dynamics and In-Silico Drug Discovery in Complex Diseases',
    url='https://github.com/mcgilldinglab/UNAGI',
)
