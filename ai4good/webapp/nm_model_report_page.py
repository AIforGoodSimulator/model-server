import logging
import textwrap
import dash_html_components as html
import dash_core_components as dcc

from ai4good.models.nm.nm_model import NetworkModel
from ai4good.models.nm.parameters.initialise_parameters import Parameters
from ai4good.webapp.apps import facade, model_runner, cache, local_cache, cache_timeout


@cache.memoize(timeout=cache_timeout)
def layout(camp, profile, cmp_profiles):
    _, profile_df, params = get_model_result(camp, profile)

    return html.Div(
        [
            dcc.Markdown(disclaimer(camp), style={'margin': 30}),
            html.H1(f'AI for Good Simulator Model Report for {camp} Camp {profile} profile', style={
                    'margin': 30}),
            dcc.Markdown(glossary(), style={'margin': 30}),
            dcc.Markdown(overview1(camp, params), style={'margin': 30}),

        ], style={'margin': 50}
    )


def disclaimer(camp):
    return textwrap.dedent(f'''
    ##### Disclaimer: This report is a draft report from AI for Good Simulator on the COVID-19 situation
    in {camp} camp. The insights are preliminary and they are subject to future model
    fixes and improvements on parameter values
    ''').replace('\n', ' ')


def glossary():
    return textwrap.dedent(f'''
    ## 0. Glossary 
    * **hospitalisation/critical care demand person days**: The residents go into the hospitalisation/critical stage signify they require care but they may not receive the care depending on the capacity so this aggregates the hospitalisation/critical demand by person times number of days. If a unit cost of care is known, the total medical expense can be calculated.
    * **IQR (interquartile range)**: The output of modelling are represented as Interquartile range representing a confidence interval of 25%-75%.
    ''')


def overview1(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 1. Overview
    This report provides simulation-based estimates for COVID-19 epidemic scenarios for the {camp} camp. 
    TODO: Add relevant description
    ''')


@local_cache.memoize(timeout=cache_timeout)
def get_model_result(camp: str, profile: str):
    logging.info("Reading data for: " + camp + ", " + profile)
    mr = model_runner.get_result(NetworkModel.ID, profile, camp)
    assert mr is not None
    profile_df = facade.ps.get_params(
        NetworkModel.ID, profile).drop(columns=['Profile'])
    params = Parameters()
    #report = mr.get('report_base_model')
    return mr, profile_df, params
