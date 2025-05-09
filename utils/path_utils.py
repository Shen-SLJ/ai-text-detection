import os

from definitions import ROOT_DIR

def abs_path_from_project_path(path: str) -> str:
    return f"{ROOT_DIR}/{path}"