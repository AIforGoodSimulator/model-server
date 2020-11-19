import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from ai4good.webapp.apps import dash_app, facade, model_runner, _redis
import ai4good.webapp.run_model_page as run_model_page
import ai4good.webapp.common_elements as common_elements
from ai4good.webapp.model_runner import InputParameterCache
import ai4good.utils.path_utils as pu

not_sure_yes_no = -1
age_off_site_range = [60, 100]
radio_option_3 = [{'label':'Yes', 'value':1}, {'label':'No', 'value':0}, {'label':'Not Sure', 'value':not_sure_yes_no}]

layout = html.Div(
    [
        common_elements.nav_bar(),
 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/input_step3.png', title='Step 3 of 4', style={'width':'50%'}, className="step_counter")), 
                            html.P('Provide your best estimate if data is not available',className='card-text'),
                            html.H5('Health Interventions', className='card-text'),
                            html.Header('Available ICU Beds', className='card-text'),
                            html.Header('How many ICU beds are currently available to residents?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='available-ICU-beds', placeholder='', type='number', min=0, max=100, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Increased ICU Capacity Available', className='card-text'),
                            html.Header('How many additional ICU beds could be added to the existing capacity?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='increased-ICU-beds', placeholder='', type='number', min=0, max=100, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Remove High-risk Residents', className='card-text'),
                            html.Header('Is it possible to move residents at greater risk to COVID-19 off-site?', className='card-text', style={'color':'darkgray'}), 
                            dbc.RadioItems(
                                options=radio_option_3, id='remove-high-risk-off-site', inline=True, style={'margin-bottom':'25px'}),
                            html.Header('What is the age range of people that are moved off-site at the settlement?', className='card-text', style={'color':'darkgray'}), 
                            html.Div([
                                html.B(''), 
                                dbc.Label('', id='age-min-moved-off-site'), 
                                dcc.RangeSlider(id='age-range-moved-off-site', min=0, max=100, step=1, value=age_off_site_range, updatemode='drag', allowCross=False), 
                                dbc.Label('', id='age-max-moved-off-site')], 
                                style={'display':'grid', 'grid-template-columns':'6% 4% 80% 10%', 'margin-bottom':'25px'}),
                            html.Header([
                                html.B('Residents with Comorbidity '), 
                                html.Abbr('\u2139', title='Comorbidity - the simultaneous presence of two or more diseases or medical conditions in a patient'), 
                            ], className='card-text'), 
                            html.Header('What is the total number of people with known comorbidity of COVID-19?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='number-known-comorbidity', placeholder='Optional', type='number', min=0, max=100000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Isolation Capacity and Policy', className='card-text'),
                            html.Header('What is the total capacity of the isolation or quarantine centre at the settlement? If there is none, put 0. ', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='isolation-centre-capacity', placeholder='', type='number', min=0, max=100000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('How many days do residents need to spend in isolation or quarantine centre if they are suspected or tested positive', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='days-quarantine-tested-positive', placeholder='Optional', type='number', min=0, max=30, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Community Shielding', className='card-text'),
                            html.Header('Has community shielding been implemented already?', className='card-text', style={'color':'darkgray'}), 
                            dbc.RadioItems(
                                options=radio_option_3, id='community-shielding', inline=True, style={'margin-bottom':'25px'}),
                            html.Header('COVID-19 Community Surveillance Program', className='card-text'),
                            html.Header('Are the health or other actors in the camp actively surveilling for COVID-19 cases within the camp?', className='card-text', style={'color':'darkgray'}), 
                            dbc.RadioItems(
                                options=radio_option_3, id='community-surveillance-program', inline=True, style={'margin-bottom':'25px'}),
                            dbc.CardFooter(dbc.Button('Next', id='page-3-button', color='secondary', href='/sim/input_page_4', style={'float':'right'})),
                            html.Div(id='input-page-3-alert')
                            ], body=True), 
                        html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)

@dash_app.callback(
    [Output('age-min-moved-off-site', 'children'), Output('age-max-moved-off-site', 'children')], 
    [Input('age-range-moved-off-site', 'value')])
def update_age_range_input_moved_off_site(value):
    return value[0], value[1]

@dash_app.callback(
    [Output('available-ICU-beds', 'value'), Output('increased-ICU-beds', 'value'), 
     Output('remove-high-risk-off-site', 'value'), Output('age-range-moved-off-site', 'value'), 
     Output('number-known-comorbidity', 'value'), Output('isolation-centre-capacity', 'value'), 
     Output('days-quarantine-tested-positive', 'value'), Output('community-shielding','value'), 
     Output('community-surveillance-program', 'value')], 
    [Input('page-3-button', 'n_clicks')], 
    [State('available-ICU-beds', 'value'), State('increased-ICU-beds', 'value'), 
     State('remove-high-risk-off-site', 'value'), 
     State('age-min-moved-off-site', 'children'), State('age-max-moved-off-site', 'children'), 
     State('number-known-comorbidity', 'value'), State('isolation-centre-capacity', 'value'), 
     State('days-quarantine-tested-positive', 'value'), State('community-shielding','value'), 
     State('community-surveillance-program', 'value')])
def update_input_parameter_page_3(
    n_clicks, available_ICU_beds_value, increased_ICU_beds_value, remove_high_risk_off_site_value, 
    age_min_moved_off_site_value, age_max_moved_off_site_value, number_known_comorbidity_value, 
    isolation_centre_capacity_value, days_quarantine_tested_positive_value, community_shielding_value, 
    community_surveillance_program_value):
    inputParameterCache = InputParameterCache(_redis)
    input_param = {
        'available-ICU-beds': available_ICU_beds_value, 
        'increased-ICU-beds': increased_ICU_beds_value, 
        'remove-high-risk-off-site': remove_high_risk_off_site_value, 
        'age-min-moved-off-site': age_min_moved_off_site_value, 
        'age-max-moved-off-site': age_max_moved_off_site_value, 
        'number-known-comorbidity': number_known_comorbidity_value, 
        'isolation-centre-capacity': isolation_centre_capacity_value, 
        'days-quarantine-tested-positive': days_quarantine_tested_positive_value, 
        'community-shielding': community_shielding_value, 
        'community-surveillance-program': community_surveillance_program_value, 
    }
    if n_clicks is None:
        value = inputParameterCache.cache_get(input_param.keys())  # get cached value
        for i,j in enumerate(value):
            if j is None:  # if first time loading, get default value
                if list(input_param.keys())[i] in {'age-min-moved-off-site'}:
                    value[i] = age_off_site_range[0]
                elif list(input_param.keys())[i] in {'age-max-moved-off-site'}:
                    value[i] = age_off_site_range[1]
                else:
                    value[i] = None
        value[3] = [value[3], value[4]]  # replace the age min/max moved off site with age range
        del value[4]  # remove the other age moved off site
        return value
    else:
        inputParameterCache.cache_set(input_param, 3)  # put all input parameters in input page 3 to cache
        raise PreventUpdate
