from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='BibExpressVisualization',
    version='1.0.0',
    description='A Python library to quickly visualize scopus based bibliographic data',
    author='Jan-Felix Klein',
    author_email='janfelixklein@googlemail.com',
    packages=find_packages(),
    install_requires=required_packages,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
