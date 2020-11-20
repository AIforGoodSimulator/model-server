import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_caching import Cache
from dask.distributed import Client
from ai4good.runner.facade import Facade
from ai4good.webapp.model_runner import ModelRunner
from ai4good.utils.logger_util import get_logger
from dotenv import load_dotenv
import redis
import socket
import os

cache_timeout = 60*60*2  # In seconds

logger = get_logger(__name__,'DEBUG')
load_dotenv()

flask_app = Flask(__name__)

local_cache = Cache(flask_app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': cache_timeout
})

cache = Cache(flask_app, config={
    'DEBUG': True,
    'CACHE_DEFAULT_TIMEOUT': cache_timeout,
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get("REDIS_URL"),
    'CACHE_KEY_PREFIX': socket.gethostname()
})

_redis = redis.Redis.from_url(os.environ.get("REDIS_URL"))

dash_app = dash.Dash(
    __name__,
    server=flask_app,
    routes_pathname_prefix='/sim/',
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, '/static/css/ai4good.css']
)
dash_app.title = "AI4Good COVID-19 Model Server"

_client = None  # Needs lazy init

def dask_client() -> Client:    
    global _client

    if _client is None:
        if ("DASK_SCHEDULER_HOST" not in os.environ) :
            logger.warn("No Dask Sceduler host specified in .env, Running Dask locally ...")
            _client = Client()
        elif (os.environ.get("DASK_SCHEDULER_HOST")=="127.0.0.1") :
            logger.info("Running Dask locally ...")
            _client = Client()
        elif (os.environ.get("DASK_SCHEDULER_HOST")=='') :
            logger.warn("No Dask Sceduler host specified in .env, Running Dask locally ...")
            _client = Client()
        else :
            logger.info("Running Dask Distributed using Dask Scheduler ["+os.environ.get("DASK_SCHEDULER_HOST")+"] ...")
            _client = Client(os.environ.get("DASK_SCHEDULER_HOST")+":"+os.environ.get("DASK_SCHEDULER_PORT"))

    return _client

facade = Facade.simple()

model_runner = ModelRunner(facade, _redis, dask_client)
