import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synthrouting",
    use_scm_version=False,
    author="Weather Navigation",
    description="Zermelo navigation problem via hybrid search",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "black",
        "imageio",
        "jax",
        "jaxlib",
        "matplotlib",
        "numpy",
        "pip-tools",
        "pytest",
        "scipy",
        "streamlit",
        "typer",
    ],
)
