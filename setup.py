import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()
requirements = [req for req in requirements if req and not req.startswith("#")]

setup(
    name="weaver",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/de-lachende-cavalier/weaver",
    packages=find_packages(exclude=["*.egg-info", "tests*", "docs*", "build*"]),
    install_requires=requirements,
    extras_require={
        "modal": ["modal"],
    },
    python_requires=">=3.13",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
