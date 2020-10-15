import argparse
import logging
from enum import Enum
from ai4good.models.model_registry import get_models, create_params
from ai4good.runner.facade import Facade
from ai4good.models.model import ModelResult


facade = Facade.simple()


def cache_info():

    def find_profile(m, c, rid):
        _mdl = get_models()[m](facade.ps)
        for p in facade.ps.get_profiles(m):
            params = create_params(facade.ps, m, p, c)
            res_id = _mdl.result_id(params)
            if res_id == rid:
                return p
        return None

    for m in get_models():
        cached_data = facade.rs.list(m)
        print(f"---- {m} ----")
        for cd in cached_data:
            rid = facade.rs.result_id_from_file_name(cd, m)
            mr: ModelResult = facade.rs.load(m, rid)
            p = mr.get('params')
            profile_info = find_profile(m, p.camp, rid)
            if profile_info is None:
                profile_info = str(p.control_dict)
            print(f'{rid}\t{p.camp}\t{profile_info}')


class Commands(Enum):
    CACHE_INFO = cache_info


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='AI4Good utils')
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('--cache-info', dest='command', action='store_const', const=Commands.CACHE_INFO,
                               help='Print cache info')

    args = parser.parse_args()

    cmd = args.command
    if cmd == Commands.CACHE_INFO:
        cache_info()
    else:
        raise ValueError('Unknown command: '+str(cmd))
