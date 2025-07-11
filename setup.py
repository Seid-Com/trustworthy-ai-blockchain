from setuptools import setup, find_packages

setup(
    name="trustworthy-ai-blockchain",
    version="1.0.0",
    description="Blockchain-Integrated Federated Learning for Trustworthy AI",
    author="Seid Mehammed",
    author_email="seidmda@gmail.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    keywords="blockchain, federated learning, trustworthy ai, machine learning, security",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/trustworthy-ai-blockchain/issues",
        "Source": "https://github.com/yourusername/trustworthy-ai-blockchain",
    },
)