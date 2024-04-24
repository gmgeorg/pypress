from setuptools import setup, find_packages

setup(
    name="pypress",
    version="0.0.2",
    url="https://github.com/gmgeorg/pypress.git",
    author="Georg M. Goerg",
    author_email="im@gmge.org",
    description="Predictive State Smoothing (PRESS) in Python (keras)",
    packages=find_packages(),
    install_requires=["numpy >= 1.11.0", "tensorflow >= 2.11.0", "pandas >= 1.0.0"],
)
