[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![PyPI version](https://badge.fury.io/py/pygus.svg)](https://badge.fury.io/py/pygus) [![Versions](https://img.shields.io/pypi/pyversions/pygus)]()

![GUS-IMAGE](https://i-p.rmcdn.net/61d40c7b94627d001f2e8309/4445961/image-86e86df2-51c6-4f23-893d-6beaeb528260.png?w=176&e=webp&nll=true)

## Green Unified Scenarios 
A digital twin representation, simulation of forests and their impact analysis.

### Installation

Install GUS from PyPi:

```
$ pip install pyGus==2.1.2
```

You can use, Poetry as well:

```
$ poetry add pyGus
```

### Development

You can create and use a virtualenv easily with `pyenv` and `poetry`

#### PyENV

See: [pyenv on github](https://github.com/pyenv/pyenv)

##### MacOS + Homebrew
```
brew update &&
brew install readline xz &&
brew install pyenv
```

Add those to your `~/.bashrc` or `~/.zshrc` (or any profiler you use)

```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Install a specific python version:

```
$ PY_VERSION = 3.9.17 # or 3.8.13 or 3.10.12, whichever you like :)
$ pyenv install $PY_VERSION
```

Then set this as your system global (or local) version

```
$ pyenv global $PY_VERSION
```

and install poetry

```
$ pip install poetry
```

Sidenote: We recommend to store your poetry virtualenvs within the project directory for ease of access to source code, etc.

```
$ poetry config virtualenvs.in-project true
```

Finally, install dependencies using

```
$ poetry install
```

Stick to PEP8 rules for code development. To do the checks, you can run code checks with [Black](https://black.readthedocs.io/en/stable/index.html)

Once you're done with developing on your branch, before pushing your changes, please run:

`$ poetry run black .`

### Testing

The code is tested by through `pytest`, which is included in the `requirements.txt`. You can manually install it with the command:
`$ pip3 install pytest`

And run the tests:

`$ pytest`

All tests are in the `tests/` folder

### Who maintains GUS?
The GUS is currently developed and maintained by [Lucidminds](https://lucidminds.ai/) 

### Notes
* The GUS is open for PRs.
* PRs will be reviewed by the current maintainers of the project.
* Extensive development guidelines will be provided soon.
* To report bugs, fixes, and questions, please use the [GitHub issues](https://github.com/lucidmindsai/gus/issues).
