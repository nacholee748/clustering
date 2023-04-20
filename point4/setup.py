from setuptools import setup, find_packages

setup(
    name="unsupervised_jim",
    version="0.0.1",
    description="Library for calculating matrix operations ",
    url="https://github.com/nacholee748/unsupervised-package",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    long_description_content_type="text/markdown",
    author="JORGE IGNACIO MORALES",
    author_email="jorge.morales1@udea.edu.co",
)