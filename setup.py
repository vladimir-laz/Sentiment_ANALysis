from setuptools import setup

setup(
    name="sentiment",
    packages=[
        "src",
    ],
    install_requires=[
        "torch",
        "transformers~=4.26.0",
        "pandas~=1.4.4",
        "yaml",
        "pyyaml~=6.0",
        "numpy~=1.23.5",
        "scikit-learn~=1.0.2",
        "tqdm~=4.64.1",
        "omegaconf~=2.3.0",
    ],
    author="",
)