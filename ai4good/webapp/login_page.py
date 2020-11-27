import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from ai4good.webapp.apps import dash_auth_app
import ai4good.webapp.common_elements as common_elements

login_msg = 'Please provide valid email and password to continue'
forgot_password_msg = 'Please contact AI for Good if your password is lost'

layout = html.Div(
    [
        common_elements.nav_bar('register'),
        dcc.Location(id='login-url', refresh=False), 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('Login', className='card-title'),
                            html.P('Welcome to AI for Good Simulator for COVID-19', className='card-text'),
                            html.Form([
                                dbc.Input(id='login-email', placeholder='Email', type='email', name='email', autoComplete=True, autoFocus=True, style={'margin-bottom':'15px'}),
                                dbc.Input(id='login-password', placeholder='Password', type='password', name='password', style={'margin-bottom':'15px'}),
                                dbc.Label('',id='login-continue-warning', color='danger', style={'margin-bottom':'15px'}), 
                                dbc.CardFooter(
                                    html.Div([
                                        dbc.Nav([
                                            dbc.NavLink('Forgot password?', id='login-forgot-password', href='#')
                                        ]), 
                                        dbc.Tooltip(forgot_password_msg, target='login-forgot-password'), 
                                        html.P(''), 
                                        #dbc.Button('Login', id='login-button', color='dark', href='/sim/landing', style={'float':'right'}), 
                                        html.Button('Login', id='login-submit-button', type='submit', disabled=True, className='mr-1', style={'float':'right'}), 
                                    ], style={'display':'grid', 'grid-template-columns':'50% 20% 30%'}), 
                                ), 
                            ], action='/login/', method='post'), 
                            html.Div(id='login-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ], style={'padding':'100px'})
    ]
)

@dash_auth_app.callback(
    [Output('login-submit-button', 'disabled'), Output('login-submit-button', 'className'), 
     Output('login-submit-button', 'title'), Output('login-continue-warning', 'children')], 
    [Input('login-email','value'), Input('login-password','value')], 
    [State('login-url', 'href')])
def update_login_page(login_email_value, login_password_value, login_href_value):
    query_dict = parse_qs(urlparse(login_href_value).query)
    error_msg = query_dict.get('error')
    if (not login_email_value) | (not login_password_value):
        return [True, 'mr-1', login_msg, error_msg]
    else:
        return [False, 'btn-dark', '', error_msg]
