import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="gym-classics2",
    version="1.0.0",
    author="Michael Hahsler",
    author_email="mhahsler@smu.edu",
    description="Classic environments for reinforcement learning and dynamic"
                " programming, implemented in Gymnasium. Reimplementation of the original gym-classics package by Brett Daley.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mhahsler/gym-classics2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
