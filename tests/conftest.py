import pytest
from fdtd.backend import backend_names

# Pylint with 'pylint fdtd' in the .git root
# To run tests in a conda env, use `python -m pytest` in the .git root
# to specify a test, use -k "test_current_detector"
# To view output, -rA
# to run with all backends, --all_backends.
# with --all_backends, --maxfail=1 is recommended to avoid blowing up the scrollback!
# to view the output of long tests live,
# use -s or --capture=no

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

# Perform tests over all backends when pytest called with --all_backends
# and function name has "all_bends" in it.
# Function must have (, backends) in args and
# rename to all_bds for consistency?
# but "all_bends" is, frankly, hilarious
def backend_parametrizer(metafunc):
    # called once per each test function
    # see https://docs.pytest.org/en/6.2.x/example/parametrize.html
    if("all_bends" in metafunc.function.__name__):
        if(metafunc.config.getoption("all_backends")):
            funcarglist = backend_names
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )
        else:
            funcarglist = [backend_names[0]]
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )


pytest_generate_tests = backend_parametrizer


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run_slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

def pytest_addoption(parser):
    parser.addoption("--all_backends", action="store_true", help="run all backends")
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run slow tests"
    )

from fixtures import backend_parametrizer

pytest_generate_tests = backend_parametrizer# perform tests over all backends
