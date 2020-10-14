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
                            html.P('Fill in the fllowing fields to provide data in order to run the simulation',className='card-text'),
                            html.H5('Settlement information', className='card-text'),
                            html.P('Name of Settlement', className='card-text'),
                            dbc.Input(id='name-settlement', placeholder='Required'),
                            html.Br(),
                            html.P('Location', className='card-text'),
                            dbc.Input(id='location', placeholder='Required'),
                            html.P('Total Population', className='card-text'),
                            dbc.Input(id='population', placeholder='Required', type='number', min=0, max=100000, step=10),
                            html.P('Total Area (sq. km)', className='card-text'),
                            dbc.Input(id='population', placeholder='Optional', type='number', min=0, max=100000, step=100),
                            dbc.Button('Next', id='page--button', color='success', href='/sim/input_page_2', block=True),
                            html.Br(),
                            html.Div(id='input-page-1-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ])
    ]
)
