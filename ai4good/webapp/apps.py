import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_caching import Cache
from dask.distributed import Client
from ai4good.runner.facade import Facade
from ai4good.webapp.model_runner import ModelRunner
import redis
import socket


flask_app = Flask(__name__)

cache_timeout = 60*60*2  # In seconds

local_cache = Cache(flask_app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': cache_timeout
})

REDIS_URL = 'rediss://:IrKLJLmfJaafc4sIop3hJAlUFNj3KesPvb+cABkDEnk=@ai4good-redis.redis.cache.windows.net:6380'

cache = Cache(flask_app, config={
    'DEBUG': True,
    'CACHE_DEFAULT_TIMEOUT': cache_timeout,
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': REDIS_URL,
    'CACHE_KEY_PREFIX': socket.gethostname()
})


_redis = redis.Redis.from_url(REDIS_URL)

dash_app = dash.Dash(
    __name__,
    server=flask_app,
    routes_pathname_prefix='/sim/',
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
dash_app.title = "AI4Good COVID-19 Model Server"

_client = None  # Needs lazy init


def dask_client() -> Client:
    global _client
    if _client is None:
        _client = Client(processes=False)
    return _client


facade = Facade.simple()
model_runner = ModelRunner(facade, _redis, dask_client)
