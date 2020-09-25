import dash_core_components as dcc
import dash_html_components as html
from ai4good.models.nm.nm_model import NetworkModel
from ai4good.webapp.apps import dash_app, facade, model_runner
import plotly.graph_objects as go
from ai4good.models.cm.plotter import figure_generator, age_structure_plot, stacked_bar_plot, uncertainty_plot
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def charts(mr):

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.H6('Result Graph: TODO'),
            ]), width=6)
        ], style={'margin': 10}),
    ])


def layout(camp, profile):
    mr = model_runner.get_result(NetworkModel.ID, profile, camp)
    assert mr is not None
    return html.Div(
        [
            html.H3('Model Results'),
            charts(mr),
        ], style={'margin': 10}
    )



