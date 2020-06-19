from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.models.cm.cm_model import CompartmentalModel
from typing import Dict, Any
import json


def get_models() -> Dict[str, Any]:
    return {
        CompartmentalModel.ID: lambda ps: CompartmentalModel(ps)
    }


def create_params(ps, _model, _profile, camp, overrides=None):  # model specific params
    """
    :param ps:
    :param _model:
    :param _profile: str or pd.Dataframe
    :param camp:
    :param overrides:
    :return:
    """
    if _model == CompartmentalModel.ID:
        override_dct = {} if overrides is None else json.loads(overrides)
        profile_df = ps.get_params(_model, _profile) if (type(_profile) is str) else _profile
        return Parameters(ps, camp, profile_df, override_dct)
    else:
        raise RuntimeError('Unsupported model: '+_model)