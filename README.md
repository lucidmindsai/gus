# gus
Green Urban Scenarios - A digital twin representation, simulation of urban forests and their impact analysis.

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
$ pyenv install 3.9.12 -- Pick your Python version
```

Create a virtualenv:

```
$ pyenv virtualenv 3.9.12 gus
```

To enable virtualenv:

```
$ pyenv shell gus
```

Install GUS Python Deps:

```
$ pip install -r requirements.txt
```

Stick to PEP8 rules for code development. To do the checks, install `flake8` to your local machine:

`$ pip3 install flake8`

Once you're done with developing on your branch, before pushing your changes, please run:

`$ flake8 <file_you_changed_or_added>`

Please fix the errors and warnings if they appear.

### Documentation

To read the documentation, navigate to `html/src` directory and open the index file.

To generate documentation based on the docstr of the code
first install

`$ pip3 install pdoc3`

and then run

`pdoc --html gus/src`

### Testing

The code is tested by through `pytest`, which is included in the `requirements.txt`. You can manually install it with the command:
`$ pip3 install pytest`

And run the tests:

`$ pytest`

All tests are in the `tests/` folder
