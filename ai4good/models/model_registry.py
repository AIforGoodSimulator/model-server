from ai4good.models.model import Model
from ai4good.models.cm.cm_model import CompartmentalModel
from typing import Dict, Any


def get_models() -> Dict[str, Any]:
    return {
        CompartmentalModel.ID: lambda ps: CompartmentalModel(ps)
    }