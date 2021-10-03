import setuptools

description = """a 3D electromagnetic FDTD simulator written in Python"""

with open("readme.md", "r") as file:
    long_description = file.read()

reqs = [
    "tqdm",
    "numpy",
    "scipy",
    "matplotlib",
]

extras = {
    "dev": [
        "pytest",
        "black",
        "nbstripout",
        "pre-commit",
        "ipykernel",
        "line_profiler",
    ],
    "test": [
        "pytest",
    ],
    "docs": [
        "sphinx",
        "nbsphinx",
        "sphinx-rtd-theme",
    ],
}
author = "Floris laporte"
version = "0.0.3"

setuptools.setup(
    name="fdtd",
    version=version,
    author=author,
    author_email="floris.laporte@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/flaport/fdtd",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
