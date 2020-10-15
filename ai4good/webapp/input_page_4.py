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
                            html.Center(html.Img(src='/static/placeholder286x180.png', title='Step 4 of 4', style={'width':'50%'})), 
                            html.P('More insights might be obtained when the following parameters are provided',className='card-text'),
                            html.H5('Advanced parameters', className='card-text'),
                            html.Header('Advanced parameter 1', className='card-text'),
                            dbc.Input(id='name-settlement', placeholder='Required', bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Advanced parameter 2', className='card-text'),
                            dbc.Input(id='location', placeholder='Required', bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Advanced parameter 3', className='card-text'),
                            dbc.Input(id='population', placeholder='Required', type='number', min=0, max=100000, step=10, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Advanced parameter 4', className='card-text'),
                            dbc.Input(id='population', placeholder='Optional', type='number', min=0, max=100000, step=100, bs_size='sm', style={'margin-bottom':'25px'}),
                            dbc.CardFooter(dbc.Button('Next', id='page-4-button', color='dark', href='/sim/run_model')),
                            html.Div(id='input-page-1-alert')
                            ], body=True
                        ), width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)
