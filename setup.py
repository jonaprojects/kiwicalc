from setuptools import setup, find_packages

setup(
    name="kiwicalc",
    version="1.0.0",
    description="A comprehensive mathematical library for Python",
    author="Kiwicalc Team",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "sympy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
