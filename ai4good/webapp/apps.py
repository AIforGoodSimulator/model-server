import dash
import dash_bootstrap_components as dbc
from flask import Flask
from flask_caching import Cache
from dask.distributed import Client
from ai4good.runner.facade import Facade
from ai4good.webapp.model_runner import ModelRunner
import ai4good.utils.path_utils as pu

flask_app = Flask(__name__)


fast_cache = Cache(flask_app, config={  # todo: Move to redis or memcached
    'CACHE_TYPE': 'simple'
})

cache = Cache(flask_app, config={  # todo: Move to redis or memcached
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR':  pu.cache_path()
})

cache_timeout = 60*60*2  # In seconds


dash_app = dash.Dash(
    __name__,
    server=flask_app,
    routes_pathname_prefix='/sim/',
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
dash_app.title = "AI4Good COVID-19 Model Server"

_client = None  #Needs lazy init


def dask_client() -> Client:
    global _client
    if _client is None:
        _client = Client(processes=False)
    return _client


facade = Facade.simple()
model_runner = ModelRunner(facade, dask_client)