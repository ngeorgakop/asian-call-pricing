"""
Setup script for the Asian Options Pricing Library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="asian-options",
    version="1.0.0",
    author="Asian Options Development Team",
    author_email="contact@asianoptions.dev",
    description="A comprehensive Python library for pricing Asian call options",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asianoptions/asian-options-pricing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "asian-options=asian_options.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)