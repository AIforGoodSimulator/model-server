import logging
import textwrap
import dash_html_components as html
import dash_core_components as dcc

from ai4good.models.nm.nm_model import NetworkModel
from ai4good.models.nm.initialise_parameters import Parameters
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
            dcc.Markdown(overview(camp, params), style={'margin': 30}),

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
    * **hospitalisation/critical care demand person days**: The residents go into the hospitalisation/critical
    stage signify they require care but they may not receive the care depending on the capacity so this aggregates
    the hospitalisation/critical demand by person times number of days. If a unit cost of care is known, the total
    medical expense can be calculated.
    * **IQR (interquartile range)**: The output of modelling are represented as Interquartile range representing
    a confidence interval of 25%-75%.
    ''')


def overview(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 1. Overview
    This report provides simulation-based estimates for COVID-19 epidemic scenarios for the {camp} camp. 

    There are an estimated {str(params.total_population)} people currently living in the camp.

    This model combines social network dynamics with the well-known SEIRS compartment model. This adds a
    stochasticity component based on social interactions to the modeling of an epidemic, while also allowing
    us to customize the parameters to individual-level specificity. We build upon the Python library developed
    by Ryan S. McGee called (SEIRS+)[https://github.com/ryansmcgee/seirsplus]. We built a series of networks
    that resemble the Moria camp according to age distribution, geographic arrangement of the camp, households,
    and ethnicities.

    !()[path_to_image]
    where:
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
    
    *Image and parameters taken from McGee's repository. Link provided above.*
    ''')


def description(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 2. Description
    It runs a social network with edge weights corresponding to the transmissibility of the disease and nodes
    corresponding to an individual. On top, the transitions between states of an individual are handled much
    like in the compartment model, with added compartments regarding symptoms, critical care, isolation, and
    different quarantine procedures.
    ''')


def logic_and_assumptions(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 2. Logic and Assumptions
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


def summary_baseline(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 3. Summary of Profiles
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
