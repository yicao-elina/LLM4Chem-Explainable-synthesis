from setuptools import setup, find_packages

setup(
    name="jhu_colors",
    version="1.0.0",
    description="Johns Hopkins University official color palette for matplotlib",
    packages=find_packages(),
    package_data={
        'jhu_colors': ['data/*.json'],
    },
    install_requires=[
        'matplotlib>=3.0.0',
        'numpy',
    ],
    python_requires='>=3.6',
)