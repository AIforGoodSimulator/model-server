import dash_core_components as dcc
import dash_html_components as html
from ai4good.models.nm.nm_model import NetworkModel
from ai4good.webapp.apps import dash_app, facade, model_runner
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def charts(mr):
    result_bm = mr.get('result_base_model')
    result_sq = mr.get('result_single_fq')
    fig = go.Figure()
    fig_sq = go.Figure()
    for col in result_bm.drop(columns=['Time', 'Susceptible', 'T_index']).columns:
        fig.add_trace(go.Scatter(
            x=result_bm['Time'], y=result_bm[col], name=col))
    for col in result_sq.drop(columns=['Time', 'Susceptible', 'T_index']).columns:
        fig_sq.add_trace(go.Scatter(
            x=result_sq['Time'], y=result_sq[col], name=col))

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.H6('Base Model'),
                dcc.Graph(id='fig_bm', figure=fig)
            ]), width=6),
            dbc.Col(html.Div([
                html.H6('Interventions with single food queue'),
                dcc.Graph(id='fig_sq', figure=fig_sq)
            ]), width=6)
        ], style={'margin': 10})])


def layout(camp, profile):
    mr = model_runner.get_result(NetworkModel.ID, profile, camp)
    assert mr is not None
    return html.Div(
        [
            html.H3('Network Model Results'),
            charts(mr),
        ], style={'margin': 10}
    )
