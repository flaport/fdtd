import pytest

def pytest_addoption(parser):
    parser.addoption("--all_backends", action="store_true", help="run all backends")

# from fixtures import backend_parametrizer
#
# pytest_generate_tests = backend_parametrizer
