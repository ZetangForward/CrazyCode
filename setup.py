from setuptools import setup, find_packages

with open('./src/modelzipper/__init__.py', 'r') as f:  
    lines = f.readlines()  
version = [line for line in lines if line.startswith('__version__')][0].split('=')[1].strip().strip('"').strip("'") 

setup(
    name='modelzipper',
    version=version,
    package_dir={'': 'src'},
    packages=find_packages("src"),
    description='Quick Command Line Tools for Model Deployment',
    author='Zecheng-Tang',
    author_email='zctang2000@gmail.com',
    url='https://github.com/ZetangForward/ZipCode.git', 
    license='MIT', 
    install_requires=[  # FIXME: add more dependencies following huggingface setup.py
        'termcolor',
        'matplotlib',
        'pyyaml',
        'fire',
        'transformers',
        'matplotlib',
        'gpustat',
        'pytz',
        'nltk',
        'torchmetrics',
        'bert_score',
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
