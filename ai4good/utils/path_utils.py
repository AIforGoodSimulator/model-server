import os

base = '../../fs'


def fig_path(name: str) -> str:
    return _path(f'{base}/figs', name)


def model_results_path(name: str) -> str:
    return _path(f'{base}/model_results', name)


def params_path(name: str) -> str:
    return _path(f'{base}/params', name)


def reports_path(name: str) -> str:
    return _path(f'{base}/reports', name)


def _path(suffix: str, name: str) -> str:
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    base_dir = os.path.join(__location__, suffix)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, name)