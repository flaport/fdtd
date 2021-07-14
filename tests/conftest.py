import pytest

# To run tests in a conda env, use `python -m pytest` in the .git root
# to specify a test, use -k "test_current_detector"
# To view output, -rA
# to run with all backends, --all_backends.
# with --all_backends, --maxfail=1 is recommended to avoid blowing up the scrollback!



def pytest_addoption(parser):
    parser.addoption("--all_backends", action="store_true", help="run all backends")

from fixtures import backend_parametrizer

pytest_generate_tests = backend_parametrizer# perform tests over all backends
