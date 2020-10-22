import argparse
import logging
import plotly.graph_objects as go
from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.models.model_registry import get_models, create_params
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.runner.facade import Facade
from ai4good.models.cm.plotter import figure_generator, age_structure_plot, stacked_bar_plot, uncertainty_plot
import ai4good.utils.path_utils as pu
from functools import reduce
import logging
import textwrap

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.webapp.apps import dash_app, facade, model_runner, cache, local_cache, cache_timeout
from ai4good.webapp.cm_model_report_page import get_model_result
from ai4good.webapp.cm_model_report_utils import *
#==============< POC changes >==================================
def layout():
    return html.Div(
        [
            html.H3('Model Results'),
            report_poc(),
        ]
    )


def report_poc():
    mr, profile_df, params, report = get_model_result('Moria', "better_hygiene_one_month")
    logging.info(f"Plotting ")

    columns_to_plot = ['Infected (symptomatic)', 'Hospitalised', 'Critical', 'Deaths']
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        vertical_spacing=0.10,
                        horizontal_spacing=0.06,
                        subplot_titles=columns_to_plot)

    for i, col in enumerate(columns_to_plot):
        row_idx = int(i % 2 + 1)
        col_idx = int(i / 2 + 1)
        p1, p2 = plot_iqr(report, col)
        fig.add_trace(p1, row=row_idx, col=col_idx)
        fig.add_trace(p2, row=row_idx, col=col_idx)
        fig.update_yaxes(title_text=col, row=row_idx, col=col_idx)

    x_title = 'better_hygiene_one_month'
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    fig.update_xaxes(title_text=x_title, row=2, col=2)

    fig.update_traces(mode='lines')
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        showlegend=False
    )

    fig.show()
    # return [
    #     dcc.Graph(
    #         id='plot_all_fig',
    #         figure=fig,
    #         style={'width': '100%'}
    #     )
    # ]


color_scheme_main = ['rgba(0, 176, 246, 0.2)', 'rgba(255, 255, 255,0)', 'rgb(0, 176, 246)']
color_scheme_secondary = ['rgba(245, 186, 186, 0.5)', 'rgba(255, 255, 255,0)', 'rgb(255, 0, 0)']

def plot_iqr(df: pd.DataFrame, y_col: str,
             x_col='Time', estimator=np.median, estimator_name='median', ci_name_prefix='',
             iqr_low=0.25, iqr_high=0.75,
             color_scheme=color_scheme_main):
    grouped = df.groupby(x_col)[y_col]
    est = grouped.agg(estimator)
    cis = pd.DataFrame(np.c_[grouped.quantile(iqr_low), grouped.quantile(iqr_high)], index=est.index,
                       columns=["low", "high"]).stack().reset_index()

    x = est.index.values.tolist()
    x_rev = x[::-1]

    y_upper = cis[cis['level_1'] == 'high'][0].values.tolist()
    y_lower = cis[cis['level_1'] == 'low'][0].values.tolist()
    y_lower = y_lower[::-1]

    p1 = go.Scatter(
        x=x + x_rev, y=y_upper + y_lower, fill='toself', fillcolor=color_scheme[0],
        line_color=color_scheme[1], name=f'{ci_name_prefix}{iqr_low*100}% to {iqr_high*100}% interval')
    p2 = go.Scatter(x=x, y=est, line_color=color_scheme[2], name=f'{y_col} {estimator_name}')
    return p1, p2



#==============< End of Ruchita changes >==================================
