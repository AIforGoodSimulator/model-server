import dash
import dash_bootstrap_components as dbc
from flask import Flask
from ai4good.runner.facade import Facade
from ai4good.webapp.model_runner import ModelRunner

flask_app = Flask(__name__)

dash_app = dash.Dash(
    __name__,
    server=flask_app,
    routes_pathname_prefix='/sim/',
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

facade = Facade.simple()
model_runner = ModelRunner(facade)
