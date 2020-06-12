from typeguard import typechecked
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from ai4good.params.param_store import ParamStore


@typechecked
class ModelResult:

    def __init__(self, rid: str, result_data: Dict[str, Any], rtype: str = 'generic'):
        self.rid = rid
        self.rtype = rtype
        self.result_data = result_data

    def get(self, key: str):
        return self.result_data[key]


@typechecked
class Model(ABC):

    def __init__(self, ps: ParamStore):
        self.ps = ps

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def result_id(self, camp: str, profile: str) -> str:
        """Result id to enable caching"""
        pass

    @abstractmethod
    def run(self, camp: str, profile: str) -> ModelResult:
        pass


