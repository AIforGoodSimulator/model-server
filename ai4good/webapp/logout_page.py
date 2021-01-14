import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import ai4good.webapp.common_elements as common_elements

thankyou_msg = 'Thank you for using AI for Good Simulator for COVID-19'
logout_msg = 'Are you sure to logout?'
progress_msg = 'You may log out and come back for the simulation results later'

layout = html.Div(
    [
        common_elements.nav_bar('landing'),
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('Logout', className='card-title'),
                            html.P(thankyou_msg, className='card-text'),
                            html.P(logout_msg,className='card-text'),
                            html.Form([
                                dbc.CardFooter(
                                    html.Div([
                                        dbc.Nav([
                                            dbc.NavLink('Simulation in progress?', id='logout-sim-progress', href='#')
                                        ]), 
                                        dbc.Tooltip(progress_msg, target='logout-sim-progress'),
                                        html.P(''), 
                                        html.Button('Logout', id='logout-submit-button', type='submit', disabled=False, className='mr-1', style={'float':'right'}), 
                                    ], style={'display':'grid', 'grid-template-columns':'50% 20% 30%'}), 
                                ), 
                            ], action='/logout/'), 
                            html.Div(id='logout-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ], style={'padding':'100px'})
    ]
)
