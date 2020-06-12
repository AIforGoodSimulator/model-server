import argparse
import logging
import plotly.graph_objects as go
from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.models.model_registry import get_models
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.runner.facade import Facade
from ai4good.models.cm.functions import generate_csv
from ai4good.models.cm.plotter import figure_generator, age_structure_plot, stacked_bar_plot, uncertainty_plot
import ai4good.utils.path_utils as pu

facade = Facade.simple()


@typechecked
def run_model(_model: str, _profile: str, camp: str, load_from_cache: bool, save_to_cache: bool,
              is_save_plots: bool, is_show_plots: bool, is_save_report: bool):
    logging.info('Running %s model with %s profile', _model, _profile)
    _mdl: Model = get_models()[_model](facade.ps)
    res_id = _mdl.result_id(camp, _profile)
    if load_from_cache and facade.rs.exists(_mdl.id(), res_id):
        logging.info("Loading from model result cache")
        mr: ModelResult = facade.rs.load(_mdl.id(), res_id)
    else:
        logging.info("Running model for camp %s", camp)
        mr: ModelResult = _mdl.run(camp, _profile)
        if save_to_cache:
            logging.info("Saving model result to cache")
            facade.rs.store(_mdl.id(), res_id, mr)
    if is_save_plots or is_show_plots:
        save_plots(mr, res_id, is_save_plots, is_show_plots)
    if is_save_report:
        save_report(mr, res_id)


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


def save_report(mr, res_id):
    logging.info("Saving report")
    sols_raw = mr.get('sols_raw')
    p = mr.get('params')
    df = generate_csv(sols_raw, p,  input_type='raw')
    df.to_csv(pu.reports_path(f"all_R0_{res_id}.csv"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='AI4Good model runner')
    parser.add_argument('--model', type=str, choices=get_models().keys(), help='Model to run',
                        default=CompartmentalModel.ID)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument('--profile', type=str, help='Model profile to run, by default first one will be run')
    profile_group.add_argument('--run_all_profiles', action='store_true', help='Run all profiles in the model', default=False)
    parser.add_argument('--camp', type=str, help='Camp to run model for', default='Moria')
    parser.add_argument('--do_not_load_from_model_result_cache', dest='load_from_cache', action='store_false',
                        help='Do not load from cache, re-compute everything', default=True)
    parser.add_argument('--do_not_save_to_model_result_cache', dest='save_to_cache', action='store_false',
                        help='Do save results to cache', default=True)
    parser.add_argument('--save_plots', dest='save_plots', action='store_true', help='Save plots', default=False)
    parser.add_argument('--show_plots', dest='show_plots', action='store_true', help='Show plots', default=False)
    parser.add_argument('--save_report', dest='save_report', action='store_true', help='Save model report', default=False)
    args = parser.parse_args()

    model = args.model
    assert model in facade.ps.get_models()
    if args.run_all_profiles:
        for profile in facade.ps.get_profiles(model):
            run_model(model, profile, args.camp, args.load_from_cache, args.save_to_cache, args.save_plots,
                      args.show_plots, args.save_report)
    else:
        if args.profile is None:
            profile = facade.ps.get_profiles(model)[0]
        else:
            profile = args.profile
        run_model(model, profile, args.camp, args.load_from_cache, args.save_to_cache, args.save_plots,
                  args.show_plots, args.save_report)

    logging.info('Model Runner finished normally')