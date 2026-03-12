from setuptools import setup, find_packages

setup(
    name="vortex-kfold-engine",
    version="0.1.0",
    author="BELBIN BENO R M",
    author_email="belbin.datascientist@gmail.com",
    description="A high-performance K-Fold cross-validation engine with automated Cloudpickle persistence for Classification and Regression.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BELBINBENORM/vortex-kfold-engine",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "cloudpickle",
    ],
)
