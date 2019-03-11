import fdtd
import setuptools

description = """a 3D electromagnetic FDTD simulator written in Python"""

with open("readme.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name=fdtd.__name__,
    version=fdtd.__version__,
    author=fdtd.__author__,
    author_email="floris.laporte@gmail.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/flaport/fdtd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
