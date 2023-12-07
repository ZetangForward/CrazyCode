from setuptools import setup, find_packages

setup(
    name='ZipCode-ZetangForward',
    version='0.2',
    packages=find_packages(),
    description='Quick Command Line Tools for Deep Learning',
    author='Zecheng-Tang',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/myproject',  # 可选
    install_requires=[
        'termcolor',
        'matplotlib',
        'pyyaml',
        'fire',
        'transformers>=4.34.0',
        'matplotlib',
        # 其他依赖，如果有的话
    ],
    # 如果你的包需要特定版本的Python，则可以使用python_requires来指定
    python_requires='>=3.8',
)
