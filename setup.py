from setuptools import setup

setup(
    name="sentiment",
    packages=[
        "src",
    ],
    install_requires=[
        "torch",
        "transformers",
        "pandas",
        "pyyaml",
        "numpy",
        "scikit-learn",
        "tqdm",
        "omegaconf",
    ],
    author="",
)