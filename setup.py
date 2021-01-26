from setuptools import setup, find_packages

setup(
    name="covpred",
    version="0.1",
    author="Shane Barratt, Stephen Boyd",
    packages=find_packages(),
    setup_requires=[
        "numpy",
        "scipy",
        "torch"
    ],
    url="http://github.com/cvxgrp/covpred/",
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
