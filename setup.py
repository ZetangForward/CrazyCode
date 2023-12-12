from setuptools import setup, find_packages

setup(
    name='modelzipper',
    version='0.2.6',
    package_dir={'': 'src'},
    packages=find_packages(),
    description='Quick Command Line Tools for Model Deployment',
    author='Zecheng-Tang',
    author_email='zctang2000@gmail.com',
    url='https://github.com/ZetangForward/ZipCode.git', 
    license='MIT', 
    install_requires=[
        'termcolor',
        'matplotlib',
        'pyyaml',
        'fire',
        'transformers>=4.34.0',
        'matplotlib',
        'gpustat',
        'pytz',
        'nltk',
        'torchmetrics',
        'bert_score',
    ],
    python_requires='>=3.8',
)
