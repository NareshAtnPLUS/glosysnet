import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glosysnet",
    version="0.0.1",
    author="Naresh Kumar",
    author_email="nareshkumarAtnPLUS@gmail.com",
    description="Package containing all Machine Learning and Deep Learning Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/nareshkumarAtnPLUS/glosysnet.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)