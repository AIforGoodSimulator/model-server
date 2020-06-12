import logging
import collections
import traceback
import pandas as pd
from typing import List
from enum import Enum, auto
from dask.distributed import Client, Future
from ai4good.models.model import Model, ModelResult
from ai4good.models.model_registry import get_models
from datetime import datetime

MAX_CONCURRENT_MODELS = 5
HISTORY_SIZE = 10


class ModelScheduleRunResult(Enum):
    ALREADY_RUNNING = auto()
    CAPACITY = auto()
    SCHEDULED = auto()


class ModelRunResult(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    CANCELLED = auto()
    ERROR = auto()


class ModelRunHistory:
    def __init__(self):
        self.history = collections.deque([], HISTORY_SIZE)

    def record_scheduled(self, key):
        self.history.append((key, ModelRunResult.RUNNING, datetime.now(), None))

    def record_finished(self, key, mr):
        self.history.append((key, ModelRunResult.SUCCESS, datetime.now(), mr))

    def record_cancelled(self, key):
        self.history.append((key, ModelRunResult.CANCELLED, datetime.now(), None))

    def record_error(self, key, error_details):
        self.history.append((key, ModelRunResult.CANCELLED, datetime.now(), error_details))


class ModelRunner:

    def __init__(self, facade, dask_client_provider):
        self.facade = facade
        self.models_running_now = dict()
        self.history = ModelRunHistory()
        self.dask_client_provider = dask_client_provider

    def run_model(self, _model: str, _profile: str, camp: str) -> ModelScheduleRunResult:

        def on_future_done(f: Future):
            self.models_running_now.pop(key, None)
            if f.status == 'finished':
                logging.info("Model run %s success", str(key))
                self.history.record_finished(key, f.result())
            elif f.status == 'cancelled':
                logging.info("Model run %s cancelled", str(key))
                self.history.record_cancelled(key)
            else:
                tb = f.traceback()
                error_details = traceback.format_tb(tb)
                logging.error("Model run %s failed: %s", str(key), error_details)
                self.history.record_error(key, error_details)

        key = (_model, _profile, camp)
        if key in self.models_running_now.keys():
            return ModelScheduleRunResult.ALREADY_RUNNING
        elif len(self.models_running_now) >= MAX_CONCURRENT_MODELS:
            return ModelScheduleRunResult.CAPACITY
        else:
            client = self.dask_client_provider()
            future: Future = client.submit(self._sync_run_model, _model, _profile, camp)
            self.models_running_now[key] = future
            future.add_done_callback(on_future_done)
            self.history.record_scheduled(key)
            return ModelScheduleRunResult.SCHEDULED

    @staticmethod
    def history_columns() -> List[str]:
        return ['Key', 'Status', 'Time', 'Details']

    def history_df(self) -> pd.DataFrame:
        rows = []
        for r in self.history.history:
            rows.append({
                'Key': str(r[0]),
                'Status': str(r[1]),
                'Time': str(r[2]),
                'Details': str(r[3]),
            })
        return pd.DataFrame(rows)

    def _sync_run_model(self, _model: str, _profile: str, camp: str) -> ModelResult:
        logging.info('Running %s model with %s profile', _model, _profile)
        _mdl: Model = get_models()[_model](self.facade.ps)
        res_id = _mdl.result_id(camp, _profile)
        if self.facade.rs.exists(_mdl.id(), res_id):
            logging.info("Loading from model result cache")
            return self.facade.rs.load(_mdl.id(), res_id)
        else:
            logging.info("Running model for camp %s", camp)
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