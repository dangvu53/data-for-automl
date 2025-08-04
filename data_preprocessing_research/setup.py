"""
Setup script for the Data Preprocessing Research Framework.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="data-preprocessing-research",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="A comprehensive framework for studying data preprocessing impact on AutoML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/data-preprocessing-research",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.1.0",
            "notebook>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-baseline=experiments.scripts.run_baseline:main",
            "run-preprocessing=experiments.scripts.run_preprocessing_study:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "docs/*.md"],
    },
)
