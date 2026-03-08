"""DACA - DaVinci Accelerated Compute Architecture.

A compute platform library for Huawei Ascend 910ProA NPUs.
"""

from setuptools import setup, find_packages

setup(
    name="daca",
    version="0.1.0",
    author="DACA Contributors",
    author_email="changcheng967@users.noreply.github.com",
    description="DaVinci Accelerated Compute Architecture - Ascend NPU Platform Library",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/changcheng967/DACA",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*", "benchmarks", "benchmarks.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "mindspore": [
            "mindspore>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ascend npu huawei davinCi ai machine-learning deep-learning mindspore",
    entry_points={
        "console_scripts": [
            "daca-probe=daca.tools.probe:main",
            "daca-doctor=daca.tools.doctor:main",
        ],
    },
)
