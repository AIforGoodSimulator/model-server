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
import os

text_disclaimer = 'Disclaimer: This tool is for informational and research purposes only and should not be considered as a medical predictor. The input parameters you have provided is a determining factor in the simulation results. '

path_country = '../../fs/params/cm_model/contact_matrices/'
country_raw = [f for f in os.listdir(path_country) if os.path.isfile(os.path.join(path_country, f))]
country_clean = sorted([f.split('.')[0] for f in country_raw])

layout = html.Div(
    [
        run_model_page.nav_bar(),
        
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H6(text_disclaimer, className='card-text'),
                            html.Div(id='input-page-1-disclaimer')
                            ], body=True
                        ), width=6
                    ), justify='center', style={'margin-top':'10px'}
                )
            ])
        ]), 

        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/placeholder286x180.png', title='Step 1 of 4', style={'width':'50%'})), 
                            html.P('Fill in the following fields to provide data in order to run the simulation',className='card-text'),
                            html.H5('Settlement information', className='card-text'),
                            html.Header('Name of Settlement', className='card-text'),
                            dbc.Input(id='name-settlement', placeholder='Required', bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Location', className='card-text'),
                            dbc.Input(id='location', placeholder='Required', bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Country', className='card-text'),
                            html.Small(dcc.Dropdown(
                                options=[{'label': k, 'value': k} for k in country_clean], 
                                id='countries-dropdown', placeholder='Required', style={'margin-bottom':'25px'})),
                            html.Header('Total Area (sq. km)', className='card-text'),
                            dbc.Input(id='total-area', placeholder='Optional', type='number', min=0, max=100000, step=100, bs_size='sm', style={'margin-bottom':'25px'}),
                            dbc.CardFooter(dbc.Button('Next', id='page-1-button', color='secondary', href='/sim/input_page_2')),
                            html.Div(id='input-page-1-alert')
                            ], body=True
                        ), width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)
