# -*- coding: utf-8 -*-

# Header ...

import os
from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()
    
with open("LICENSE") as f:
    license = f.read()
    
with open("inspection2/_version.py") as f:
    exec(f.read())

with open("requirements.txt") as f:
    requirements = f.read().split("\n")
    
    
setup(
    name="inspection2",
    version=__version__,
    description="Deep learning package customized to industrial inspection",
    long_description=readme,
    author="Shuai",
    author_email="shuaih7@gmail.com",
    url="https://github.com/shuaih7/inspection2",
    license=license,
    classifiers=["Programming Language :: Python :: 3.6.6", 
                 "Programming Language :: Python :: Implementation :: CPython",],
    packages=find_packages(exclude=("docs")),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "cpu": [
            "tensorflow==2.2.0",
        ],
        "gpu": [
            "tensorflow-gpu==2.2.0",
        ],
    },
    test_suits="nose.collector",
    tests_require=["nose"],
)