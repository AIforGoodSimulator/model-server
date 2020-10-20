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
                            html.H4('Login', className='card-title'),
                            html.P('Welcome to AI for Good Simulator for COVID-19', className='card-text'),
                            dbc.Input(id='login-email', placeholder='Email', type='email', style={'margin-bottom':'15px'}),
                            dbc.Input(id='login-password', placeholder='Password', type='password', style={'margin-bottom':'25px'}),
                            dbc.CardFooter(
                                html.Div([
                                    dbc.Button('Login', id='login-button', color='dark', href='/sim/input_page_1'), 
                                    html.P(''), 
                                    dbc.Nav([
                                        dbc.NavLink('Forgot password?', id='forgot-password', href='#')
                                    ]), 
                                    dbc.Tooltip('Please contact AI for Good if your password is lost', target='forgot-password')
                                ], style={'display':'grid', 'grid-template-columns':'30% 20% 50%'}), 
                            ), 
                            html.Div(id='login-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ], style={'padding':'100px'})
    ]
)
