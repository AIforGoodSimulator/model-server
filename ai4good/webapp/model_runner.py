import logging
import traceback
import redis
import pandas as pd
from typing import List
from enum import Enum, auto
from dask.distributed import Client, Future
from ai4good.models.model import Model, ModelResult
from ai4good.models.model_registry import get_models, create_params
from datetime import datetime
import pickle
import socket

MAX_CONCURRENT_MODELS = 3
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
    _CACHE_KEY = f'{socket.gethostname()}_model_run_history'

    def __init__(self, _redis: redis.Redis):
        self._redis = _redis

    def _append(self, t):
        with self._redis.pipeline() as pipe:
            pipe.lpush(self._CACHE_KEY, pickle.dumps(t))
            pipe.ltrim(self._CACHE_KEY, 0, HISTORY_SIZE)
            pipe.execute()

    def record_scheduled(self, key):
        self._append((key, ModelRunResult.RUNNING, datetime.now(), None))

    def record_finished(self, key, mr):
        self._append((key, ModelRunResult.SUCCESS, datetime.now(), "Success"))

    def record_cancelled(self, key):
        self._append((key, ModelRunResult.CANCELLED, datetime.now(), None))

    def record_error(self, key, error_details):
        self._append((key, ModelRunResult.CANCELLED, datetime.now(), error_details))

    def history(self):
        history = self._redis.lrange(self._CACHE_KEY, 0, HISTORY_SIZE)
        return map(pickle.loads, history)


class ModelsRunningNow:
    _CACHE_KEY = f'{socket.gethostname()}models_running_now'

    def __init__(self, _redis: redis.Redis):
        self.state = dict()
        self._redis = _redis

    def pop(self, key):
        self._redis.srem(self._CACHE_KEY,  '¬'.join(key))

    def start_run(self, key, f):
        _skey = '¬'.join(key)
        with self._redis.pipeline() as pipe:
            error_count = 0
            while error_count < 1000:
                try:
                    pipe.watch(self._CACHE_KEY)
                    n_running = self._redis.scard(self._CACHE_KEY)
                    print("N_running: "+str(n_running))
                    is_running = self._redis.sismember(self._CACHE_KEY, _skey)
                    print("is_running: " + str(is_running))
                    if n_running >= MAX_CONCURRENT_MODELS:
                        pipe.unwatch()
                        return ModelScheduleRunResult.CAPACITY
                    elif is_running:
                        pipe.unwatch()
                        return ModelScheduleRunResult.ALREADY_RUNNING
                    else:
                        pipe.multi()
                        pipe.sadd(self._CACHE_KEY, _skey)
                        pipe.execute()
                        f()
                        return ModelScheduleRunResult.SCHEDULED
                except redis.WatchError:
                    error_count += 1
                    logging.warning("ModelsRunningNow optimistic lock error #%d; retrying", error_count)
        raise RuntimeError("Failed to obtain lock")


class ModelRunner:

    def __init__(self, facade, _redis: redis.Redis, dask_client_provider):
        self.facade = facade
        self.history = ModelRunHistory(_redis)
        self.models_running_now = ModelsRunningNow(_redis)
        self.dask_client_provider = dask_client_provider

    def run_model(self, _model: str, _profile: str, camp: str) -> ModelScheduleRunResult:

        def on_future_done(f: Future):
            self.models_running_now.pop(key)
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

        def submit():
            client = self.dask_client_provider()
            self.history.record_scheduled(key)
            future: Future = client.submit(self._sync_run_model, self.facade, _model, _profile, camp)
            future.add_done_callback(on_future_done)

        key = (_model, _profile, camp)
        return self.models_running_now.start_run(key, submit)

    @staticmethod
    def history_columns() -> List[str]:
        return ['Key', 'Status', 'Time', 'Details']

    def history_df(self) -> pd.DataFrame:
        rows = []
        for r in self.history.history():
            rows.append({
                'Key': str(r[0]),
                'Status': str(r[1]),
                'Time': str(r[2]),
                'Details': str(r[3]),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _sync_run_model(facade, _model: str, _profile: str, camp: str) -> ModelResult:
        logging.info('Running %s model with %s profile', _model, _profile)
        _mdl: Model = get_models()[_model](facade.ps)
        params = create_params(facade.ps, _model, _profile, camp)
        res_id = _mdl.result_id(params)
        logging.info("Running model for camp %s", camp)
        mr = _mdl.run(params)
        logging.info("Saving model result to cache")
        facade.rs.store(_mdl.id(), res_id, mr)
        return mr

    def results_exist(self, _model: str, _profile: str, camp: str) -> bool:
        _mdl: Model = get_models()[_model](self.facade.ps)
        params = create_params(self.facade.ps, _model, _profile, camp)
        res_id = _mdl.result_id(params)
        return self.facade.rs.exists(_mdl.id(), res_id)

    def get_result(self, _model: str, _profile: str, camp: str) -> ModelResult:
        _mdl: Model = get_models()[_model](self.facade.ps)
        params = create_params(self.facade.ps, _model, _profile, camp)
        res_id = _mdl.result_id(params)
        return self.facade.rs.load(_mdl.id(), res_id)