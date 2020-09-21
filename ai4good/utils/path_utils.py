import os
import glob

base = '../../fs'


def fig_path(name: str) -> str:
    return _path(f'{base}/figs', name)


def model_results_path(name: str) -> str:
    return _path(f'{base}/model_results', name)


def list_models(glob_spec: str):
    p = _path(f'{base}/model_results')
    return glob.glob(p+'/'+glob_spec)


def params_path(name: str) -> str:
    return _path(f'{base}/params', name)


def reports_path(name: str) -> str:
    return _path(f'{base}/reports', name)


def cache_path() -> str:
    return _path(f'{base}/cache')


def cm_params_path(name: str) -> str:
    return _path(f'{base}/params/cm_model', name)


def _path(suffix: str, name: str = None) -> str:
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    base_dir = os.path.join(__location__, suffix)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    if name:
        return os.path.join(base_dir, name)
    else:
        return base_dir

