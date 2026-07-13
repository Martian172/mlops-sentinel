from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlops-sentinel",
    version="0.4.0",
    author="Martian172",
    author_email="martian172@users.noreply.github.com",
    description="Production ML Model Monitoring & Alerting Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Martian172/mlops-sentinel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
        "prometheus-client>=0.18.0",
        "rich>=13.0.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "sklearn": ["scikit-learn>=1.3.0"],
        "sqlite": ["sqlalchemy>=2.0.0"],
        "dev": [
            "pytest", "pytest-cov", "pytest-asyncio", "flake8", "black",
            "scikit-learn>=1.3.0", "sqlalchemy>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentinel=sentinel.cli.commands:cli",
        ],
    },
    package_data={
        "sentinel": ["dashboard/templates/*.html", "dashboard/static/*.js"],
    },
    include_package_data=True,
)
