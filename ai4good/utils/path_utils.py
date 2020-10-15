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


nm_base = '../../ai4good/models/nm/data'   # TODO: need to harmonize this for models to share params
am_base = '../../ai4good/models/abm/data'


def get_nm_aug_pop() -> str:
    return _path(f'{nm_base}', 'augmented_population.csv')


def get_am_aug_pop() -> str:
    return _path(f'{am_base}', 'age_and_sex.csv')


def _path(suffix: str, name: str = None) -> str:
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    base_dir = os.path.join(__location__, suffix)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    if name:
        return os.path.join(base_dir, name)
    else:
        return base_dir
