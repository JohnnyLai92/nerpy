# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('nerpy/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='nerpy',
    version=__version__,
    description='nerpy: Named Entity Recognition toolkit using Python',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/nerpy',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires=">=3.8",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='NER,nerpy,Chinese Named Entity Recognition Tool,ner,bert,bert2tag',
    install_requires=[
        "loguru",
        "transformers>=4.6.0",
        "datasets",
        "tqdm",
        "scipy",
        "numpy",
        "pandas",
        "seqeval",
        "tensorboard",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'nerpy': 'nerpy'},
    package_data={'nerpy': ['*.*', 'data/*.txt']}
)
