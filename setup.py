#!/usr/bin/env python3
"""Setup script for Audio Evaluation Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-evals",
    version="1.0.0",
    author="Audio Evaluation Team",
    author_email="contact@example.com",
    description="Audio evaluation pipeline for call center quality analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/audio-evals",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"],
        "advanced": ["torch>=1.9.0", "pyannote.audio==3.1.1", "torchaudio>=0.9.0"],
    },
    entry_points={
        "console_scripts": [
            "audio-eval=audio_evals.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)