from setuptools import setup, find_packages

setup(
    name="bluff",
    version="1.0.0",
    description="BLUFF: Benchmark for Linguistic Understanding of Fake-news Forensics",
    author="Jason Lucas, Dongwon Lee",
    author_email="jsl5710@psu.edu",
    url="https://github.com/jsl5710/BLUFF",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
