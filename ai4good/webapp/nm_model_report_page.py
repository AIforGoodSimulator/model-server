import logging
import textwrap
import dash_html_components as html
import dash_core_components as dcc

from ai4good.models.nm.nm_model import NetworkModel
from ai4good.models.nm.initialise_parameters import Parameters
from ai4good.webapp.apps import facade, model_runner, cache, local_cache, cache_timeout
from ai4good.webapp.metadata_report import GenerateMetadataDict, GenerateMetadataHTML


@cache.memoize(timeout=cache_timeout)
def layout(camp, profile, cmp_profiles):

    _, profile_df, params = get_model_result(camp, profile)

    return html.Div(
        [
            GenerateMetadataHTML(GenerateMetadataDict(NetworkModel.ID, camp, profile, model_runner)),
            dcc.Markdown(disclaimer(camp), style={'margin': 30}),
            html.H1(f'AI for Good Simulator Model Report for {camp} Camp {profile} profile', style={
                    'margin': 30}),
            dcc.Markdown(overview(camp, params), style={'margin': 30}),
            dcc.Markdown(compartments(), style={'margin': 30}),
            html.Img(src='/static/nm_compartments_small.png'),
            dcc.Markdown(logic_and_assumptions(camp, params), style={'margin': 30}),
            dcc.Markdown(profiles_summary(), style={'margin': 30}),
            dcc.Markdown(preliminary_results(), style={'margin': 30}),
            html.Img(src='/static/nm_results_infections.png'),
            html.Img(src='/static/nm_results_basic.png')
        ], style={'margin': 50}
    )


def disclaimer(camp):
    return textwrap.dedent(f'''
    ##### Disclaimer: This report is a draft report from AI for Good Simulator on the COVID-19 situation
    in {camp} camp. The insights are preliminary and they are subject to future model
    fixes and improvements on parameter values
    ''').replace('\n', ' ')


def overview(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ### 1. Overview
    This report provides simulation-based estimates for COVID-19 epidemic scenarios for the {camp} camp. 

    There are an estimated {str(params.total_population)} people currently living in the camp. 
    
    This model combines social network dynamics with the well-known SEIRS compartment model. This adds a
    stochasticity component based on social interactions to the modeling of an epidemic, while also allowing
    us to customize the parameters to individual-level specificity. We build upon the Python library developed
    by Ryan S. McGee called [SEIRS+](https://github.com/ryansmcgee/seirsplus). We built a series of networks
    that resemble the Moria camp according to age distribution, geographic arrangement of the camp, households,
    and ethnicities.
    ''')


def compartments():
    return textwrap.dedent(f'''
    ### 2. Compartments        
         - σ: rate of progression to infectiousness (inverse of latent period)
         - λ: rate of progression to (a)symptomatic state (inverse of pre-symptomatic period)
         - a: probability of an infected individual remaining asymptomatic
         - h: probability of a symptomatic individual being hospitalized
         - η: rate of progression to hospitalized state (inverse of onset-to-admission period)
         - γ: rate of recovery for non-hospitalized symptomatic individuals (inverse of symptomatic infectious period)
         - γA: rate of recovery for asymptomatic individuals (inverse of asymptomatic infectious period)
         - γH: rate of recovery hospitalized symptomatic individuals (inverse of hospitalized infectious period)
         - f: probability of death for hospitalized individuals (case fatality rate)
         - μH: rate of death for hospitalized individuals (inverse of admission-to-death period)
         - ξ: rate of re-susceptibility (inverse of temporary immunity period; 0 if permanent immunity)
        
    *Image and parameters taken from https://github.com/ryansmcgee/seirsplus/wiki/Extended-SEIRS-Model-Description*
    ''')


def logic_and_assumptions(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ### 3. Logic and Assumptions
     - Each individual is represented by a node, and all of their corresponding social interactions are represented
     by edges.
     - Individuals are assigned to households and zones randomly. These households and zones mimic the geographical
     distribution of Moria.
     - Edge weights vary from edge to edge proportional to the amount of time that two individuals interact with
     one another.
     - Nodes that do not have an edge connecting one another may still interact because of the probability of
     global contact.
     - We randomly choose a number of people from each household to go to the food queue. There, they come in contact
     with the five people in front of them, and the five people behind them.
     - Individuals only interact out of leisure with other individuals from their immediate surroundings within the
     same zone who share their ethnicity.
    ''')


def profiles_summary():
    return f''' 
    ### 4. Profiles
    #### Baseline
    - 1 food queue
    #### Baseline with interventions
    - 1 food queue
    - Hygiene (transmission rate reduced by 30%) + quarantine (global contact probability reduced to 25% of baseline
    value)
    - Duration of quarantine: 30-60
    - Duration of hygiene: 30-90
    #### Multiple food queues
    - 4 and 8 food queues
    #### Multiple food queues with interventions
    - 4 and 8 food queues
    - Hygiene (transmission rate reduced by 30%) + quarantine (global contact probability reduced to 25% of baseline
    value)
    - Duration of quarantine: 30-60
    - Duration of hygiene: 30-90
    '''


def preliminary_results():
    return f''' 
    ### 5. Results
    In the following images, we see that with an R0 of 2.5, a global contact probability of 0.8, and no interventions,
    we end up with a total number of infections of 26% between days 30-40, with approximately
    1% of fatalities at the end of the simulation.
    '''


def summary_baseline(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 4. Summary of Profiles
     - Baseline
        - 1 food queue
    ''')


def summary_binterventions(camp: str, params: Parameters):
    return textwrap.dedent(f'''
     - Baseline with interventions
        - 1 food queue
        - Hygiene (transmission rate reduced by 30%) + quarantine (global contact probability reduced to 25% of baseline
        value)
        - Duration of quarantine: 30-60
        - Duration of hygiene: 30-90
    ''')


def summary_foodqueues(camp: str, params: Parameters):
    return textwrap.dedent(f'''
     - Multiple food queues
        - 4 and 8 food queues
    ''')


def summary_fqinterventions(camp: str, params: Parameters):
    return textwrap.dedent(f'''
     - Multiple food queues with interventions
        - 4 and 8 food queues
        - Hygiene (transmission rate reduced by 30%) + quarantine (global contact probability reduced to 25% of baseline
        value)
        - Duration of quarantine: 30-60
        - Duration of hygiene: 30-90
    ''')


@local_cache.memoize(timeout=cache_timeout)
def get_model_result(camp: str, profile: str):
    logging.info("Reading data for: " + camp + ", " + profile)
    mr = model_runner.get_result(NetworkModel.ID, profile, camp)
    assert mr is not None
    profile_df = facade.ps.get_params(
        NetworkModel.ID, profile)
    params = Parameters(facade.ps, camp, profile_df, {})
    return mr, profile_df, params
