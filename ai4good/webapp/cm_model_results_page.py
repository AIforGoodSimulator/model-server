import dash_core_components as dcc
import dash_html_components as html
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.webapp.apps import dash_app, facade, model_runner
import plotly.graph_objects as go
from ai4good.models.cm.plotter import figure_generator, age_structure_plot, stacked_bar_plot, uncertainty_plot
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


def charts(mr):
    sol = mr.get('standard_sol')
    percentiles = mr.get('percentiles')
    p = mr.get('params')

    multiple_categories_to_plot = ['E', 'A', 'I', 'R', 'H', 'C', 'D', 'O', 'Q', 'U']  # categories to plot
    single_category_to_plot = 'C'  # categories to plot in final 3 plots #TODO: make selectable

    fig_multi_lines = go.Figure(figure_generator(sol, p, multiple_categories_to_plot))  # plot with lots of lines
    fig_age_structure = go.Figure(age_structure_plot(sol, p, single_category_to_plot))
    fig_bar_chart = go.Figure(stacked_bar_plot(sol, p, single_category_to_plot))  # bar chart (age structure)
    fig_uncertainty = go.Figure(uncertainty_plot(sol, p, single_category_to_plot, percentiles))  # uncertainty

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.H6('Disease progress'),
                dcc.Graph(id='fig_multi_lines', figure=fig_multi_lines)
            ]), width=6),
            dbc.Col(html.Div([
                html.H6('Age Structure'),
                dcc.Graph(id='fig_age_structure', figure=fig_age_structure)
            ]), width=6)
        ], style={'margin': 10}),
        dbc.Row([
            dbc.Col(html.Div([
                html.H6('Age Structure (Bar chart)'),
                dcc.Graph(id='fig_bar_chart', figure=fig_bar_chart)
            ]), width=6),
            dbc.Col(html.Div([
                html.H6('Uncertainty'),
                dcc.Graph(id='fig_uncertainty', figure=fig_uncertainty)
            ]), width=6)
        ], style={'margin': 10}),
    ])


def layout(camp, profile):
    mr = model_runner.get_result(CompartmentalModel.ID, profile, camp)
    assert mr is not None
    return html.Div(
        [
            html.H3('Model Results'),
            charts(mr),
        ], style={'margin': 10}
    )



