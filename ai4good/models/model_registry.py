from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.models.cm.cm_model import CompartmentalModel
from typing import Dict, Any
import json


def get_models() -> Dict[str, Any]:
    return {
        CompartmentalModel.ID: lambda ps: CompartmentalModel(ps)
    }


def create_params(ps, _model, _profile, camp, overrides=None):  # model specific params
    if _model == CompartmentalModel.ID:
        override_dct = {} if overrides is None else json.loads(overrides)
        return Parameters(ps, camp, _profile, override_dct)
    else:
        raise RuntimeError('Unsupported model: '+_model)