import dash
import dash_bootstrap_components as dbc
from flask import Flask, Blueprint
from flask_caching import Cache
from flask_login import login_required
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask.helpers import get_root_path
from dask.distributed import Client
from ai4good.config import BaseConfig
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

# register flask components
def register_flask_components(server):
    # register flask extensions
    db_sqlalchemy.init_app(server)
    db_migrate.init_app(server, db_sqlalchemy)
    login.init_app(server)
    login.login_view = 'main.login'

def register_blueprints(server):
    from ai4good.webapp_file import server_bp

    server.register_blueprint(server_bp)

def register_dashapps(app):
    from ai4good.dashapp1.layout import layout
    from ai4good.dashapp1.callbacks import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    dashapp1 = dash.Dash(__name__,
                         server=app,
                         url_base_pathname='/dashboard/',
                         assets_folder=get_root_path(__name__) + '/dashboard/assets/',
                         meta_tags=[meta_viewport])

    with app.app_context():
        dashapp1.title = 'Dashapp 1'
        dashapp1.layout = layout
        register_callbacks(dashapp1)

    _protect_dashviews(dashapp1)


def _protect_dashviews(dashapp):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(dashapp.server.view_functions[view_func])


def _protect_simviews(dashapp):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(dashapp.server.view_functions[view_func])



flask_app = Flask(__name__)
flask_app.config.from_object(BaseConfig)

# start flask extensions
db_sqlalchemy = SQLAlchemy()
db_migrate = Migrate()
login = LoginManager()

register_dashapps(flask_app)
register_flask_components(flask_app)
register_blueprints(flask_app)


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
    #routes_pathname_prefix='/sim/',
    url_base_pathname='/sim/',
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, '/static/css/ai4good.css']
)
dash_app.title = "AI4Good COVID-19 Model Server"
_protect_dashviews(dash_app)

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
