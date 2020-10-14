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
                            html.P('Enter the percentage of the total population that each range represents',className='card-text'),
                            html.H5('Age Distribution (%)', className='card-text'),
                            html.P('0 - 5 years', className='card-text'),
                            dbc.Input(id='Age-percentage-00-05', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('6 - 9 years', className='card-text'),
                            dbc.Input(id='Age-percentage-06-09', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('10 - 19 years', className='card-text'),
                            dbc.Input(id='Age-percentage-10-19', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('20 - 29 years', className='card-text'),
                            dbc.Input(id='Age-percentage-20-29', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('30 - 39 years', className='card-text'),
                            dbc.Input(id='Age-percentage-30-39', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('40 - 49 years', className='card-text'),
                            dbc.Input(id='Age-percentage-40-49', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('50 - 59 years', className='card-text'),
                            dbc.Input(id='Age-percentage-50-59', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('60 - 36 years', className='card-text'),
                            dbc.Input(id='Age-percentage-60-69', placeholder='Required', type='number', min=0, max=100, step=1),
                            html.P('70 years or older', className='card-text'),
                            dbc.Input(id='Age-percentage-70+', placeholder='Required', type='number', min=0, max=100, step=1),                            
                            html.Br(),
                            dbc.CardFooter(dbc.Button('Next', id='page-2-button', color='dark', href='/sim/input_page_3')),
                            html.Br(),
                            html.Div(id='input-page-2-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ])
    ]
)
