from ai4good.params.param_store import ParamStore, SimpleParamStore
from ai4good.models.model_result_store import ModelResultStore, SimpleModelResultStore


class Facade:

    def __init__(self, ps: ParamStore, rs: ModelResultStore):
        self.ps = ps
        self.rs = rs

    @staticmethod
    def simple():
        return Facade(SimpleParamStore(), SimpleModelResultStore())
