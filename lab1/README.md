# How to run this code

## manage python environments using `uv`

We will be using `uv` to manage the Python environment.

### installing `uv`

``` shell
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

### setup the environment

```shell
# creates a virtual environment in the current folder
uv venv 

# activate the virtual environment
source .venv/bin/activate

# install everything
uv sync
```

You should see all the dependencies installed:

```shell
uv pip list
```
