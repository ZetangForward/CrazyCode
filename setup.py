from setuptools import setup, find_packages

setup(
    name='modelzipper',
    version='0.1',
    packages=find_packages(),
    description='Quick Command Line Tools for Model Deployment',
    author='Zecheng-Tang',
    author_email='zctang2000@gmail.com',
    url='https://github.com/ZetangForward/ZipCode.git', 
    install_requires=[
        'termcolor',
        'matplotlib',
        'pyyaml',
        'fire',
        'transformers>=4.34.0',
        'matplotlib',
        'gpustat',
    ],
    python_requires='>=3.9',
)
