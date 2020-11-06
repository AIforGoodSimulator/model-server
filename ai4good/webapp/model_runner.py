import traceback
import redis
import numpy as np
import pandas as pd
from typing import List
from enum import Enum, auto
from dask.distributed import Client, Future
from ai4good.models.model import Model, ModelResult
from ai4good.models.model_registry import get_models, create_params
from datetime import datetime
import pickle
import json
import socket
from ai4good.utils import path_utils as pu
from ai4good.webapp.commit_date import get_version_date
from ai4good.utils.logger_util import get_logger
from dask.distributed import wait

MAX_CONCURRENT_MODELS = 3
HISTORY_SIZE = 10
INPUT_PARAMETER_TIMEOUT = 60*30 # in seconds
logger = get_logger(__name__)

# _sid = np.random.randint(100000000, 1000000000)  # session id
_sid = 42 #fixed for local dev

class InputParameterCache:
    _CACHE_KEY = f'{socket.gethostname()}_{_sid}_input_parameter'

    def __init__(self, _redis: redis.Redis):
        self._redis = _redis

    @staticmethod
    def _decode_byte(value) -> List:
        value = [i.decode('utf-8') if i is not None else None for i in value]
        value = [i if i is None else None if i.strip() == '' else i for i in value]
        for i,j in enumerate(value):
            if j is not None:
                try:
                    value[i] = int(j)
                except:
                    value[i] = j
        return value        
        
    def cache_get_all(self):
        key_value_pair_dict = self._redis.hgetall(self._CACHE_KEY)
        key = list(key_value_pair_dict.keys())
        value = list(key_value_pair_dict.values())
        return self._decode_byte(key), self._decode_byte(value)
    
    def cache_get(self, input_param_key):
        if isinstance(input_param_key, str):
            input_param_key = [input_param_key]
        value = [self._redis.hget(self._CACHE_KEY, str(i)) for i in input_param_key]
        return self._decode_byte(value)

    def cache_set(self, input_param, page_number):
        with self._redis.pipeline() as pipe:
            error_count = 0
            while error_count < 1000:
                try:
                    pipe.watch(self._CACHE_KEY)
                    pipe.multi()
                    for i,j in input_param.items():
                        j_conv = '' if j is None else str(j)  # prevent string conversion of None
                        pipe.hset(self._CACHE_KEY, str(i), j_conv)
                    pipe.execute()
                    pipe.unwatch()
                    self._redis.expire(self._CACHE_KEY, INPUT_PARAMETER_TIMEOUT)
                    return None
                except redis.WatchError:
                    error_count += 1
                    logger.warning('Input page #%d optimistic lock error #%d; retrying', page_number, error_count)
        raise RuntimeError('Failed to obtain lock - input parameters')
    def cache_save(self):
        """Save cache locally before processing it with the models"""
        output_dict = dict(zip(self.cache_get_all))
        pickle.dump(output_dict, open(pu.cache_path(f"{_CACHE_KEY}.pickle"), "wb"))





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
        self._append((key, ModelRunResult.RUNNING, datetime.now(), None, str(get_version_date())))

    def record_finished(self, key, mr):
        self._append((key, ModelRunResult.SUCCESS, datetime.now(), "Success", str(get_version_date())))

    def record_cancelled(self, key):
        self._append((key, ModelRunResult.CANCELLED, datetime.now(), None, str(get_version_date())))

    def record_error(self, key, error_details):
        self._append((key, ModelRunResult.CANCELLED, datetime.now(), error_details, str(get_version_date())))

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
                    logger.warning("ModelsRunningNow optimistic lock error #%d; retrying", error_count)
        raise RuntimeError("Failed to obtain lock")

    def clear_run(self):
        for running_now in self._redis.smembers(self._CACHE_KEY):
            self._redis.srem(self._CACHE_KEY, running_now)

    def run_available(self):
        n_running = self._redis.scard(self._CACHE_KEY)
        if n_running >= MAX_CONCURRENT_MODELS:
            return False
        else:
            return True


class ModelRunner:

    def __init__(self, facade, _redis: redis.Redis, dask_client_provider, _sid):
        self.facade = facade
        self.history = ModelRunHistory(_redis)
        self.models_running_now = ModelsRunningNow(_redis)
        self.dask_client_provider = dask_client_provider
        self._redis = _redis
        self._sid = _sid
        self.user_input = None

    def run_model(self, _model: str, _profile: str) -> ModelScheduleRunResult:
        if self.user_input is None:
            self.user_input = self.load_input_params()
        def on_future_done(f: Future):
            self.models_running_now.pop(key)
            if f.status == 'finished':
                logger.info("Model run %s success", str(key))
                self.history.record_finished(key, f.result())
            elif f.status == 'cancelled':
                logger.info("Model run %s cancelled", str(key))
                self.history.record_cancelled(key)
            else:
                tb = f.traceback()
                error_details = traceback.format_tb(tb)
                logger.error("Model run %s failed: %s", str(key), error_details)
                self.history.record_error(key, error_details)

        def submit():
            client = self.dask_client_provider()
            self.history.record_scheduled(key)
            future: Future = client.submit(self._sync_run_model, self.facade, _model, _profile, self.user_input)
            future.add_done_callback(on_future_done)
        
        key = (_model, _profile)
        return self.models_running_now.start_run(key, submit)

    def batch_run_model(self, run_config: dict):
        if self.user_input is None:
            self.user_input = self.load_input_params()
        # initialise the client
        client = self.dask_client_provider()
        futures = []
        for model in run_config.keys():
            if len(run_config[model]) > 0:
                for profile in run_config[model]:
                    def on_future_done(f: Future):
                        if f.status == 'finished':
                            logger.info("Model run %s success", str(key))
                            self.history.record_finished(key, f.result())
                        elif f.status == 'cancelled':
                            logger.info("Model run %s cancelled", str(key))
                            self.history.record_cancelled(key)
                        else:
                            tb = f.traceback()
                            error_details = traceback.format_tb(tb)
                            logger.error("Model run %s failed: %s", str(key), error_details)
                            self.history.record_error(key, error_details)
                    key = (model, profile)
                    future: Future = client.submit(self._sync_run_model, self.facade, model, profile, self.user_input)
                    future.add_done_callback(on_future_done)
                    futures.append(future)
        wait(futures) # block operation until all futures are finished
        logger.info('all runs have finished')
        return True

    @staticmethod
    def history_columns() -> List[str]:
        return ['Key', 'Status', 'Time', 'Details', 'Version Date'] 

    def history_df(self) -> pd.DataFrame:
        rows = []
        for r in self.history.history():
            try:
                rows.append({
                    'Key': str(r[0]),
                    'Status': str(r[1]),
                    'Time': str(r[2]),
                    'Details': str(r[3]),
                    'Version Date': str(r[4]),
                })
            except IndexError: # avoids error when using a history for a model run before the version date parameter was added
                rows.append({
                    'Key': str(r[0]),
                    'Status': str(r[1]),
                    'Time': str(r[2]),
                    'Details': str(r[3]),
                })        
            
        return pd.DataFrame(rows)

    @staticmethod
    def _sync_run_model(facade, _model: str, _profile: str, user_input: str) -> ModelResult:
        logger.info('Running %s model with %s profile', _model, _profile)
        _mdl: Model = get_models()[_model](facade.ps)
        params = create_params(facade.ps, _model, _profile, user_input)
        res_id = _mdl.result_id(params)
        logger.info(f"Running model for camp with {_mdl.id()} %s")
        mr = _mdl.run(params)
        logger.info("Saving model result to cache")
        facade.rs.store(_mdl.id(), res_id, mr)
        return mr

    def results_exist(self, _model: str, _profile: str, camp: str) -> bool:
        _mdl: Model = get_models()[_model](self.facade.ps)
        params = create_params(self.facade.ps, _model, _profile, self.user_input)
        res_id = _mdl.result_id(params)
        return self.facade.rs.exists(_mdl.id(), res_id)

    def get_result(self, _model: str, _profile: str, camp: str) -> ModelResult:
        _mdl: Model = get_models()[_model](self.facade.ps)
        params = create_params(self.facade.ps, _model, _profile, self.user_input)
        res_id = _mdl.result_id(params)
        return self.facade.rs.load(_mdl.id(), res_id)

    def load_input_params(self):
        # access the particular cache from the cache ID
        input_param_cache = InputParameterCache(self._redis)
        input_param_cache._CACHE_KEY = f'{socket.gethostname()}_{self._sid}_input_parameter'
        saved_keys, saved_values = input_param_cache.cache_get_all()
        user_input = json.dumps(dict(zip(saved_keys, saved_values)))
        return user_input