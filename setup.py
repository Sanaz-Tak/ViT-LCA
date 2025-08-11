#!/usr/bin/env python3
"""
Setup script for ViT-LCA package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vit-lca",
    version="1.0.0",
    author="Sanaz M. Takaghaj",
    author_email="sanaz.takaghaj@example.com",
    description="Vision Transformer with Local Competition Algorithm for sparse feature representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sanaz-Tak/ViT-LCA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vit-lca=vit_lca_experiment:main",
        ],
    },
    keywords="vision-transformer, lca, sparse-coding, computer-vision, deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/Sanaz-Tak/ViT-LCA/issues",
        "Source": "https://github.com/Sanaz-Tak/ViT-LCA",
        "Documentation": "https://github.com/Sanaz-Tak/ViT-LCA#readme",
    },
)
