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

register_info_msg = 'Please provide valid email and password to register'
register_already_msg = 'Please login to use AI for Good if your have already registered'
register_password_msg = 'Confirm password has to be the same as password'

layout = html.Div(
    [
        common_elements.nav_bar('login'),
        dcc.Location(id='register-url', refresh=False), 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('Register', className='card-title'),
                            html.P('Register to use AI for Good Simulator for COVID-19', className='card-text'),
                            html.Form([
                                dbc.Input(id='register-email', placeholder='Email', type='email', name='email', autoComplete=True, autoFocus=True, style={'margin-bottom':'15px'}),
                                dbc.Input(id='register-password', placeholder='Password', type='password', name='password', minLength=4, style={'margin-bottom':'15px'}),
                                dbc.Input(id='register-password-confirm', placeholder='Confirm password', type='password', name='password_confirm', minLength=4, style={'margin-bottom':'15px'}),
                                dbc.Label('',id='register-continue-warning', color='danger', style={'margin-bottom':'15px'}), 
                                dbc.CardFooter(
                                    html.Div([
                                        dbc.Nav([
                                            dbc.NavLink('Already registered?', id='register-already', href='#')
                                        ]), 
                                        dbc.Tooltip(register_already_msg, target='register-already'), 
                                        html.P(''), 
                                        html.Button('Register', id='register-submit-button', type='submit', disabled=True, className='mr-1', style={'float':'right'}), 
                                    ], style={'display':'grid', 'grid-template-columns':'50% 20% 30%'}), 
                                ), 
                            ], action='/register/', method='post'), 
                            html.Div(id='register-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ], style={'padding':'100px'})
    ]
)

@dash_auth_app.callback(
    [Output('register-submit-button', 'disabled'), Output('register-submit-button', 'className'), 
     Output('register-submit-button', 'title'), Output('register-continue-warning', 'children')], 
    [Input('register-email','value'), Input('register-password','value'), 
     Input('register-password-confirm','value')], 
    [State('register-url', 'href')])
def update_register_page(register_email_value, register_password_value, register_password_confirm_value, register_href_value):
    error_msg = ''
    if (register_password_value != register_password_confirm_value):
        error_msg = register_password_msg
    else:
        query_dict = parse_qs(urlparse(register_href_value).query)
        query_list = (query_dict.get('error'))
        if query_list:
            query_msg = ''.join(query_list)
            if not register_email_value:
                error_msg = query_msg
            elif (register_email_value in query_msg):
                error_msg = query_msg
        
    if (not register_email_value) | (not register_password_value):
        return [True, 'mr-1', register_info_msg, error_msg]
    else:
        if not error_msg:
            return [False, 'btn-dark', '', '']
        else:
            return [True, 'mr-1', register_info_msg, error_msg]
