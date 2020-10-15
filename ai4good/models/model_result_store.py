from typeguard import typechecked
from typing import Any, List
from abc import ABC, abstractmethod
import os
import re
import pickle
import ai4good.utils.path_utils as pu


@typechecked
class ModelResultStore(ABC):
    @abstractmethod
    def store(self, model_id: str, result_id: str, obj: Any):
        pass

    @abstractmethod
    def load(self, model_id: str, result_id: str) -> Any:
        pass

    @abstractmethod
    def exists(self, model_id: str, result_id: str) -> bool:
        pass

    @abstractmethod
    def list(self, model_id: str) -> List[str]:
        pass

    @abstractmethod
    def remove_all(self, model_id: str):
        pass

    @staticmethod
    def result_id_from_file_name(f: str, model_id: str) -> str:
        re_res = re.search(f'{model_id}_([\\dabcdef]+)\\.pkl', f)
        return re_res.group(1)


@typechecked
class SimpleModelResultStore(ModelResultStore):

    def store(self, model_id: str, result_id: str, obj: Any):
        p = self._path(f"{model_id}_{result_id}.pkl")
        with open(p, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, model_id: str, result_id: str) -> Any:
        p = self._path(f"{model_id}_{result_id}.pkl")
        with open(p, 'rb') as handle:
            return pickle.load(handle)

    def exists(self, model_id: str, result_id: str) -> bool:
        p = self._path(f"{model_id}_{result_id}.pkl")
        return os.path.exists(p)

    @staticmethod
    def _path(name: str) -> str:
        return pu.model_results_path(name)

    def list(self, model_id: str) -> List[str]:
        return pu.list_models(f"{model_id}_*")

    def remove_all(self, model_id: str):
        files = pu.list_models(f"{model_id}_*")
        for f in files:
            os.remove(f)

