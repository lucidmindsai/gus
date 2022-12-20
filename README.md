[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# gus
![GUS-IMAGE](https://miro.medium.com/max/1400/1*fMM7rnq1RJCh-nFBGLUvyA.png)
Green Urban Scenarios - A digital twin representation, simulation of urban forests and their impact analysis.

### Installation

Install GUS from PyPi:

```
$ pip install pyGus==0.1.8
```

You can use, Poetry as well:

```
$ poetry add pyGus
```

### Development

Create a virtualenv by using pyenv, install it first:

```
$ brew install readline xz
$ brew install pyenv pyenv-virtualenv
```

Add those to your `~/.bashrc` or `~/.zshrc` (or any profiler you use)

```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Install a specific python version:

```
$ pyenv install 3.9.12 -- Pick your Python version (3.8 is available too)
```

Create a virtualenv:

```
$ pyenv virtualenv 3.9.12 gus
```

To enable virtualenv:

```
$ pyenv shell gus
```

Now run poetry to setup GUS:

```
$ poetry build
```


Stick to PEP8 rules for code development. To do the checks, install `flake8` to your local machine:

`$ pip3 install flake8`

Once you're done with developing on your branch, before pushing your changes, please run:

`$ flake8 <file_you_changed_or_added>`

Please fix the errors and warnings if they appear.

### Testing

The code is tested by through `pytest`, which is included in the `requirements.txt`. You can manually install it with the command:
`$ pip3 install pytest`

And run the tests:

`$ pytest`

All tests are in the `tests/` folder

### Who maintains GUS?
The GUS is currently developed and maintained by [Lucidminds](https://lucidminds.ai/) and [Dark Matter Labs](https://darkmatterlabs.org/) members as part of their joint project [TreesAI](https://treesasinfrastructure.com/#/).

### Notes
* The GUS is open for PRs.
* PRs will be reviewed by the current maintainers of the project.
* Extensive development guidelines will be provided soon.
* To report bugs, fixes, and questions, please use the [GitHub issues](https://github.com/lucidmindsai/gus/issues).