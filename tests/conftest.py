import pytest

def pytest_addoption(parser):
    parser.addoption("--all_backends", action="store_true", help="run all backends")
