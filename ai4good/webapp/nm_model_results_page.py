import dash_core_components as dcc
import dash_html_components as html
from ai4good.models.nm.nm_model import NetworkModel
from ai4good.webapp.apps import dash_app, facade, model_runner
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def charts(mr):
    result = mr.get('result')
    params = mr.get('params')

    total = params.total_population

    fig1 = get_figure(result, x_label='Time (Days)', y_label='Number of people', columns=['Exposed', 'Infected_Presymptomatic',
                                                                                          'Infected_Symptomatic', 'Infected_Asymptomatic', 'Hospitalized', 'Fatalities'])
    fig2 = get_figure(result, x_label='Time (Days)', y_label='Number of people',
                      columns=['Recovered', 'Susceptible'])

    # Get result in percentage format
    for col in result.drop(columns=['Time', 'T_index']):
        result[col] = (result[col]*100)/total

    fig3 = get_figure(result,  x_label='Time (Days)', y_label='Percent of Population', columns=['Exposed', 'Infected_Presymptomatic',
                                                                                                'Infected_Symptomatic', 'Infected_Asymptomatic',
                                                                                                'Hospitalized', 'Fatalities', 'Recovered', 'Susceptible'])

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.H6('Infected over time', style={'text-align': 'center'}),
                dcc.Graph(id='fig_bm', figure=fig1)
            ]), width=10),
            dbc.Col(html.Div([
                html.H6('Susceptible and Recovered over time',
                        style={'text-align': 'center'}),
                dcc.Graph(id='fig_sq', figure=fig2)
            ]), width=10),
            dbc.Col(html.Div([
                html.H6('Percentage change over time',
                        style={'text-align': 'center'}),
                dcc.Graph(id='fig_bm', figure=fig3)
            ]), width=10)
        ], style={'margin': 10})])


def get_figure(result, x_label, y_label, columns):
    fig = go.Figure(layout=dict(
        xaxis=dict(
            title=x_label,
            automargin=True,
        ),
        yaxis=dict(mirror=True,
                   title=y_label,
                   automargin=True,
                   type='linear'
                   )))
    for col in columns:
        fig.add_trace(go.Scatter(
            x=result['Time'], y=result[col], name=col,
        ))
    return fig


def layout(camp, profile):
    mr = model_runner.get_result(NetworkModel.ID, profile, camp)
    assert mr is not None
    return html.Div(
        [
            html.H3('Network Model Results'),
            html.H3('-------------------------------------'),
            html.H4('Camp: ' + camp),
            html.H3('-------------------------------------'),
            html.H4('Profile: ' + profile),
            html.H3('-------------------------------------'),
            charts(mr),
        ], style={'margin': 10}
    )
