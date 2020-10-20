import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
from dash.dependencies import Input, Output, State
import dash_table

layout = html.Div(
    [
        run_model_page.nav_bar(),
 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/input_step3.png', title='Step 3 of 4', style={'width':'50%'})), 
                            html.P('Provide your best estimate if data is not available',className='card-text'),
                            html.H5('Health Interventions', className='card-text'),
                            html.Header('Available ICU Beds', className='card-text'),
                            html.Header('How many ICU beds are currently available to residents?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='available-ICU-beds', placeholder='Required', type='number', min=0, max=100, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Increased ICU Capacity Available', className='card-text'),
                            html.Header('How many additional ICU beds could be added to the existing capacity?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='increased-ICU-beds', placeholder='Required', type='number', min=0, max=100, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Remove High-risk Residents', className='card-text'),
                            html.Header('Is it possible to remove high-risk residents off-site at the settlement?', className='card-text', style={'color':'darkgray'}), 
                            dbc.RadioItems(
                                options=[
                                    {'label':'Yes', 'value':1}, 
                                    {'label':'No', 'value':0},
                                    {'label':'Not Sure', 'value':-1},
                                ], value=-1, id='remove-high-risk-off-site', inline=True, style={'margin-bottom':'25px'}),
                            html.Header('What is the age range of people that are moved off-site at the settlement?', className='card-text', style={'color':'darkgray'}), 
                            html.Div([
                                dbc.Label('10', id='age-min-moved-off-site'), 
                                dcc.RangeSlider(id='age-range-moved-off-site', min=0, max=100, step=5, value=[10, 50], updatemode='drag', allowCross=False), 
                                dbc.Label('50', id='age-max-moved-off-site')], 
                                style={'display':'grid', 'grid-template-columns':'10% 80% 10%', 'margin-bottom':'25px'}),
                            html.Header('Residents with Comorbidity', className='card-text'),
                            html.Header('What is the total number of people with known comorbidity?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='number-known-comobidity', placeholder='Optional', type='number', min=0, max=100000, step=100, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Isolation Capacity', className='card-text'),
                            html.Header('What is the total capacity of the isolation centre at the settlement?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='isolation-capacity', placeholder='Required', type='number', min=0, max=100000, step=100, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Community Shielding', className='card-text'),
                            html.Header('Is it possible to implement shielding or has it been implemented already?', className='card-text', style={'color':'darkgray'}), 
                            dbc.RadioItems(
                                options=[
                                    {'label':'Yes', 'value':1}, 
                                    {'label':'No', 'value':0},
                                ], value=1, id='community-shielding', inline=True, style={'margin-bottom':'25px'}),
                            html.Header('Community Measures', className='card-text'),
                            html.Header('Does someone conduct inspections and checks as well as implement corrective actions?', className='card-text', style={'color':'darkgray'}), 
                            dbc.RadioItems(
                                options=[
                                    {'label':'Yes', 'value':1}, 
                                    {'label':'No', 'value':0},
                                    {'label':'Not Sure', 'value':-1},
                                ], value=-1, id='conduct-inspections-implement-corrections', inline=True, style={'margin-bottom':'25px'}),
                            html.Header('Isolation Policy', className='card-text'),
                            html.Header('What is the total number of days of quarantine for an individual who has tested postive for an infectious disease?', className='card-text', style={'color':'darkgray'}), 
                            dbc.Input(id='days-quarantine-tested-positive', placeholder='Optional', type='number', min=0, max=30, step=1, bs_size='sm', style={'margin-bottom':'25px'}),
                            dbc.CardFooter(dbc.Button('Next', id='page-3-button', color='secondary', href='/sim/input_page_4')),
                            html.Div(id='input-page-3-alert')
                            ], body=True
                        ), width=6
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
