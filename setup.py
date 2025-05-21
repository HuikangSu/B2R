#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="b2r",
    version="0.1.0",
    description="B2R: Boundary-to-Region Supervision for Offline Safe Reinforcement Learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Anonymous",
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="offline safe reinforcement learning, transformers, RL",
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "torch>=1.13.0",
        "gymnasium>=0.26.0",
        "pybullet>=3.0.6",
        "bullet_safety_gym==1.4.0",
        "safety-gymnasium==0.4.0",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "pyyaml",
        "h5py",
        "wandb",
        "matplotlib",
    ],
    extras_require={
        "metadrive": [
            "metadrive-simulator@git+https://github.com/HenryLHH/metadrive_clean.git@main"
        ],
    },
)
