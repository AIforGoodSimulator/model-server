import dash_core_components as dcc
import dash_html_components as html
from ai4good.models.nm.nm_model import NetworkModel
from ai4good.webapp.apps import dash_app, facade, model_runner
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def charts(mr):
    result = mr.get('result')
    fig1 = get_figure(result, columns=['Exposed', 'Infected_Presymptomatic',
                                       'Infected_Symptomatic', 'Infected_Asymptomatic',
                                       'Hospitalized', 'Fatalities'])
    fig2 = get_figure(result, columns=['Recovered', 'Susceptible'])

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.H6('Infected over time'),
                dcc.Graph(id='fig_bm', figure=fig1)
            ]), width=10),
            dbc.Col(html.Div([
                html.H6('Susceptible and Recovered over time'),
                dcc.Graph(id='fig_sq', figure=fig2)
            ]), width=10)
        ], style={'margin': 10})])


def get_figure(result, columns):
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(
            x=result['Time'], y=result[col], name=col))
    return fig


def layout(camp, profile):
    mr = model_runner.get_result(NetworkModel.ID, profile, camp)
    assert mr is not None
    return html.Div(
        [
            html.H3('Network Model Results'),
            html.H4('Profile: ' + profile),
            charts(mr),
        ], style={'margin': 10}
    )
