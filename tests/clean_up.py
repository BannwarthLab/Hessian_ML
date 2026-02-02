import os
import shutil
import pytest


@pytest.fixture
def cleanup_file():
    files = []

    def register(file_path):
        files.append(file_path)
        return file_path

    yield register

    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.fixture
def cleanup_folder():
    folders = []

    def register(path):
        folders.append(path)
        return path

    yield register

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
