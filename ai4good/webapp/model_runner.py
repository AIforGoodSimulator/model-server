import logging
from ai4good.models.model import Model, ModelResult
from ai4good.models.model_registry import get_models


class ModelRunner:

    def __init__(self, facade):
        self.facade = facade

    def run_model(self, _model: str, _profile: str, camp: str) -> ModelResult: #TODO: prevent double exec while in progress
        logging.info('Running %s model with %s profile', _model, _profile)
        _mdl: Model = get_models()[_model](self.facade.ps)
        res_id = _mdl.result_id(camp, _profile)
        if self.facade.rs.exists(_mdl.id(), res_id):
            logging.info("Loading from model result cache")
            return self.facade.rs.load(_mdl.id(), res_id)
        else:
            logging.info("Running model for camp %s", camp)  #TODO: add async and progress
            mr = _mdl.run(camp, _profile)
            logging.info("Saving model result to cache")
            self.facade.rs.store(_mdl.id(), res_id, mr)
            return mr

    def results_exist(self, _model: str, _profile: str, camp: str) -> bool:
        _mdl: Model = get_models()[_model](self.facade.ps)
        res_id = _mdl.result_id(camp, _profile)
        return self.facade.rs.exists(_mdl.id(), res_id)

    def get_result(self, _model: str, _profile: str, camp: str) -> ModelResult:
        _mdl: Model = get_models()[_model](self.facade.ps)
        res_id = _mdl.result_id(camp, _profile)
        return self.facade.rs.load(_mdl.id(), res_id)