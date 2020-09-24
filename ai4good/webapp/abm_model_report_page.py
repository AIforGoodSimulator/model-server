from functools import reduce
import logging
import textwrap

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from ai4good.models.abm.abm_model import ABM
from ai4good.models.abm.initialise_parameters import Parameters
from ai4good.webapp.apps import dash_app, facade, model_runner, cache, local_cache, cache_timeout
from ai4good.webapp.abm_model_report_utils import *


@cache.memoize(timeout=cache_timeout)
def layout(camp, profile, cmp_profiles):

    _, profile_df, params, _ = get_model_result(camp, profile)

    return html.Div(
        [
            dcc.Markdown(disclaimer(camp), style={'margin': 30}),
            html.H1(f'AI for Good Simulator Model Report for {camp} Camp {profile} profile', style={'margin': 30}),
            #dcc.Markdown(glossary(), style={'margin': 30}),
            dcc.Markdown(overview1(camp, params), style={'margin': 30}),
            html.Img(src='/static/abm_model.png'),
            dcc.Markdown(overview1(camp, params), style={'margin': 30}),            
            dcc.Markdown(overview_population(camp,params), style={'margin': 30}),
            html.Img(src='/static/abm_camplayout.png'),             
            #dcc.Loading(html.Div([], id='main_section_part1', style={'margin': 30})),
            #dcc.Loading(html.Div([], id='main_section_part2', style={'margin': 30})),
            #base_profile_chart_section(),
            #dcc.Loading(html.Div([], id='cmp_section', style={'margin': 30})),
            #html.Div(camp, id='_camp_param', style={'display': 'none'}),
            #html.Div(profile, id='_profile_param', style={'display': 'none'}),
            #html.Div('¬'.join(cmp_profiles), id='_cmp_profiles', style={'display': 'none'})
            dcc.Markdown(overview_interventions(camp,params), style={'margin': 30}),
            html.Img(src='/static/abm_restable.png'),
            dcc.Markdown(overview_results(camp,params), style={'margin': 30}),
            html.Img(src='/static/abm_fm.png'),
            html.Img(src='/static/abm_quarantine.png'),
            html.Img(src='/static/abm_sectoring.png'),
            html.Img(src='/static/abm_allinterv.png'),
        ], style={'margin': 50}
    )


def disclaimer(camp):
    return textwrap.dedent(f'''
    ##### Disclaimer: This report is a draft report from AI for Good Simulator on the COVID-19 situation
    in {camp} camp. The insights are preliminary and they are subject to future model
    fixes and improvements on parameter values
    ''').replace('\n', ' ')


#def glossary():
#    return textwrap.dedent(f'''
#    ## 0. Glossary 
 #   * **hospitalisation/critical care demand person days**: The residents go into the hospitalisation/critical stage signify they require care but they may not receive the care depending on the capacity so this aggregates the hospitalisation/critical demand by person times number of days. If a unit cost of care is known, the total medical expense can be calculated.
#    * **IQR (interquartile range)**: The output of modelling are represented as Interquartile range representing a confidence interval of 25%-75%.
#    ''')


def overview1(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 1. Overview
    This report provides simulation-based estimates for COVID-19 epidemic scenarios for the {camp} camp. 
    There are an estimated {int(params.population)} people currently living in the camp. 
    The model we use is a deterministic, age-specific compartment model. The agent-based model describe the evolution of the epidemic given the camp setting and evaluate potential interventions to combat the spread of COVID-19; the parameters that control COVID-19 transmission rates and disease progression are estimated from the literature. 
    The model tracks individuals as they undertake daily activities in a simulated camp; COVID-19 can be transmitted when infected and susceptible individuals interact. 
    If an individual becomes infected, the infection progresses through a series of disease states; age and pre-existing conditions are accounted for in the probability of moving from one stage to the next.
    ''')
  
def overview_interventions(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    
    COVID-19 outbreaks are modelled without interventions and in the presence of four interventions feasible for displacement camps: 
    i) sectoring, 
    ii) transmission reduction (i.e. face masks),
    iii) remove-and-isolate, 
    iv) lockdown. 
    In sectoring, the central food line is eliminated and the camp is divided into sectors. Each sector has its own food line, and each individual uses the food line in the sector in which it lives. Thus, time spent in the food line is reduced and transmission in the food line becomes local rather than global. 
    Transmission reduction could be any policy or behaviour that reduces the probability of transmission when individuals interact (e.g., the use of face masks, frequent hand washing, maintaining safe distances from others). Frequent hand-washing and maintaining safe distances from others are likely to be impossible in Moria, but residents have been provided with face masks.
    In remove-and-isolate, households with symptomatic individuals are moved to an isolation facility to prevent onward transmission of the infection. Finally, in lockdown, individuals are constrained to remain within some distance of their homes, except when visiting shared toilets or food lines.
    A small proportion of the population violates the lockdown rule. For each intervention or combination of interventions, we conducted simulations in which we introduced an infected individual into the population, and we recorded the proportion of times that an epidemic occurred (i.e., 20 or more people became infected). 
    ''')


def overview_population(camp: str, params: Parameters):
    # GS: in this section SUMMARY OF POPULATION AND HOUSEHOLD STRUCTURE, i.e. # in isoboxes/tents, #ethnical backgrounds. CAN WE PUT Fig 3 pag 14 from the paper?
    #df = params.population_frame.copy()
    #df['Population structure'] = df['Population_structure'].map('{:,.1f}%'.format)
    #df['Number of residents'] = df['Population_structure'] * params.population / 100.0
    #df['Number of residents'] = df['Number of residents'].astype(int)
    #df = df[['Age', 'Population structure', 'Number of residents']]
    return textwrap.dedent(f'''
    The ABM model accounts for the following characteristic of the population and the camp:
       *Population specific parameters: population size,age, sex, condition (healty or with pre-existing condition), disease state.
       * Camp specific parameters: each individual is a member of a household that occupies either an isobox or a tent each characterized by a mean occupancy in the camp.The exact occupancy of each isobox or tent is drawn from a Poisson distribution, and
         individuals are assigned to isoboxes or tents randomly without regard to sex or age. The camp covers a 1 x 1 (e.g., km) square (figure 3). Isoboxes are assigned to random
         locations in a central square that covers one half of the area of the camp. Tents are assigned to random locations in the camp outside of the central square. There are 144 toilets evenly
         distributed throughout the camp. Toilets are placed at the centres of the squares that form a 12 x 12 grid covering the camp. The camp has one food line. The position of the food line is
         not explicitly modelled.  
         In Moria, the homes of people with the same ethnic or national background are spatially clustered, and people interact more frequently with others from the same background as themselves                
    ''')
    
        
def overview_results(camp: str, params: Parameters):     
     return textwrap.dedent(f'''
     Interventions are applied by changing the probability of interactions in different daily activities and by dividing the camp in sectors. It is possible to test multiple combinations of interventions in different interactions scenario. Example output is presented in the pictures below, where the number of infected people (in any disease state) is presented over a 200 day period.
     ''')  


