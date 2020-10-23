import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
import ai4good.utils.path_utils as pu
from dash.dependencies import Input, Output, State
import dash_table
import os
import datetime
import sklearn

layout = html.Div(
    [
        run_model_page.nav_bar(),
        
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('Model Validation', className='card-title'),
                            html.P('Descriptions',className='card-text'),
                            html.H5('Section 1', className='card-text'),
                            html.Header('Result 1', className='card-text'),
                            html.Header('Result 2', className='card-text'),
                            html.P(''), 
                            html.H5('Section 2', className='card-text'),
                            html.Header('Result 3', className='card-text'),
                            html.Header('Result 4', className='card-text'),
                            dbc.CardFooter(dbc.Button('Back', id='validate-model-button', color='secondary', href='/sim/run_model', style={'float':'right'})),
                            html.Div(id='validate-model-page-alert'), 
                            ], body=True), 
                        html.Br()], width=8
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)
