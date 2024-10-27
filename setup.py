from setuptools import setup, find_packages

setup(
    name="cot-decoding",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "accelerate>=0.24.0",
    ]
)
