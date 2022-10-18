import setuptools
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()
        
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyGUS",
    version="0.0.1",
    author="TreesAI",
    author_email="{team@trees.ai, team@lucidminds.ai}",
    description="Green infrastructures scenario and impact analysis tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucidmindsai/gus",
    project_urls={
        "Bug Tracker": "https://github.com/lucidmindsai/gus/issues",
    },
    license='GPLv3',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Topic :: Communications :: Email",
    ],
    package_dir={"": "src"},
    package_data={"": ["inputs/*.json", "inputs/*.csv"]},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=install_requires,
)
