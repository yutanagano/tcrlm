import pytest


@pytest.fixture(scope='session')
def get_path_to_project():
    return '/home/yuta/Projects/cdr3encoding'
