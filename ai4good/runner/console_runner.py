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
from ai4good.webapp.metadata_report import GenerateMetadataDict, GenerateMetadataHTML
facade = Facade.simple()


@typechecked
def run_model(_model: str, _profile: str, camp: str, load_from_cache: bool,
              save_to_cache: bool, is_save_plots: bool, is_show_plots: bool,
              is_save_report: bool, overrides) -> ModelResult:
    logging.info('Running %s model with %s profile', _model, _profile)
    _mdl: Model = get_models()[_model](facade.ps)
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    res_id = _mdl.result_id(params)
    if load_from_cache and facade.rs.exists(_mdl.id(), res_id):
        logging.info("Loading from model result cache")
        mr: ModelResult = facade.rs.load(_mdl.id(), res_id)
    else:
        logging.info("Running model for camp %s", camp)
        mr: ModelResult = _mdl.run(params)
        if save_to_cache:
            logging.info("Saving model result to cache")
            facade.rs.store(_mdl.id(), res_id, mr)
    if is_save_plots:
        # if _mdl.id() == 'agent-based-model':
        save_plots(mr, res_id, is_save_plots, is_show_plots)
        #     save_plots_abm(mr, res_id, is_save_plots, is_show_plots)
        # else:
        #     save_plots(mr, res_id, is_save_plots, is_show_plots)
    if is_show_plots:
        show_poc_plots(mr, res_id)
    if is_save_report:
        save_report(mr, res_id)
    return mr


def save_plots(mr, res_id, is_save_plots, is_show_plots):
    multiple_categories_to_plot = ['E', 'A', 'I', 'R', 'H', 'C', 'D', 'O', 'Q', 'U']  # categories to plot
    single_category_to_plot = 'C'  # categories to plot in final 3 plots #TODO: make selectable

    # plot graphs
    sol = mr.get('standard_sol')
    percentiles = mr.get('percentiles')
    p = mr.get('params')
    fig_multi_lines = go.Figure(figure_generator(sol, p, multiple_categories_to_plot))  # plot with lots of lines
    fig_age_structure = go.Figure(age_structure_plot(sol, p, single_category_to_plot))
    fig_bar_chart = go.Figure(stacked_bar_plot(sol, p, single_category_to_plot))  # bar chart (age structure)
    fig_uncertainty = go.Figure(uncertainty_plot(sol, p, single_category_to_plot, percentiles))  # uncertainty

    if is_show_plots:
        fig_multi_lines.show()
        fig_age_structure.show()
        fig_bar_chart.show()
        fig_uncertainty.show()

    if is_save_plots:
        fig_multi_lines.write_image(pu.fig_path(f"Disease_progress_{res_id}.png"))
        fig_age_structure.write_image(pu.fig_path(f"Age_structure_{res_id}.png"))
        fig_bar_chart.write_image(pu.fig_path(f"Age_structure_(bar_chart)_{res_id}.png"))
        fig_uncertainty.write_image(pu.fig_path(f"Uncertainty_{res_id}.png"))

#==============< Ruchita changes >==================================
def show_poc_plots(mr, res_id):
    mr, profile_df, params, report = get_model_result('Moria', profile)
    logging.info(f"Plotting ")

    columns_to_plot = ['Infected (symptomatic)', 'Hospitalised', 'Critical', 'Deaths']
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05,
                        subplot_titles=columns_to_plot)

    for i, col in enumerate(columns_to_plot):
        row_idx = int(i % 2 + 1)
        col_idx = int(i / 2 + 1)
        p1, p2 = plot_iqr(report, col)
        fig.add_trace(p1, row=row_idx, col=col_idx)
        fig.add_trace(p2, row=row_idx, col=col_idx)
        fig.update_yaxes(title_text=col, row=row_idx, col=col_idx)

    x_title = 'Time, days'
    fig.update_xaxes(title_text=profile, row=2, col=1)
    fig.update_xaxes(title_text=profile, row=2, col=2)

    fig.update_traces(mode='lines')
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
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


color_scheme_main = ['rgba(0, 100, 200, 0.2)', 'rgba(255, 255, 255,0)', 'rgb(0, 100, 200)']
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



def save_report(mr, res_id):
    logging.info("Saving report")
    df = mr.get('report')
    df.to_csv(pu.reports_path(f"all_R0_{res_id}.csv"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='AI4Good model runner')
    parser.add_argument('--model', type=str, choices=get_models().keys(), help='Model to run',
                        default=CompartmentalModel.ID)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument('--profile', type=str, help='Model profile to run, by default first one will be run')
    profile_group.add_argument('--run_all_profiles', action='store_true', help='Run all profiles in the model', default=False)

    camp_group = parser.add_mutually_exclusive_group()
    camp_group.add_argument('--camp', type=str, help='Camp to run model for', default='Moria')
    camp_group.add_argument('--run_all_camps', action='store_true', help='Run all camps', default=False)

    parser.add_argument('--do_not_load_from_model_result_cache', dest='load_from_cache', action='store_false',
                        help='Do not load from cache, re-compute everything', default=True)
    parser.add_argument('--do_not_save_to_model_result_cache', dest='save_to_cache', action='store_false',
                        help='Do save results to cache', default=True)
    parser.add_argument('--save_plots', dest='save_plots', action='store_true', help='Save plots', default=False)
    parser.add_argument('--show_plots', dest='show_plots', action='store_true', help='Show plots', default=False)
    parser.add_argument('--save_report', dest='save_report', action='store_true', help='Save model report', default=False)
    parser.add_argument('--profile_overrides', type=str, help='Model specific profile overrides as JSON', default=None)
    args = parser.parse_args()

    model = args.model
    assert model in facade.ps.get_models()
    if args.run_all_profiles:
        for profile in facade.ps.get_profiles(model):
            if args.run_all_camps:
                for camp in facade.ps.get_camps():
                    run_model(model, profile, camp, args.load_from_cache, args.save_to_cache, args.save_plots,
                              args.show_plots, args.save_report, args.profile_overrides)
            else:
                run_model(model, profile, args.camp, args.load_from_cache, args.save_to_cache, args.save_plots,
                          args.show_plots, args.save_report, args.profile_overrides)
    else:
        if args.profile is None:
            profile = facade.ps.get_profiles(model)[0]
        else:
            profile = args.profile

        if args.run_all_camps:
            for camp in facade.ps.get_camps():
                run_model(model, profile, camp, args.load_from_cache, args.save_to_cache, args.save_plots,
                          args.show_plots, args.save_report, args.profile_overrides)
        else:
            run_model(model, profile, args.camp, args.load_from_cache, args.save_to_cache,
                      args.save_plots, args.show_plots, args.save_report, args.profile_overrides)

    logging.info('Model Runner finished normally')
