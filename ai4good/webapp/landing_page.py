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

text_introduction = 'The Simulator is a web tool for NGOs and local authorities to model COVID-19 outbreak inside refugee camps and prepare timely and proportionate response measures needed to flatten the curve and reduce the number of fatalities. This tool helps to predict the possble outbreak scenarios and their potential outcomes and help NGOs design an optimal intervention strategy. '

tooltip_partnership = 'This web tool is developed by AI for Good and Deutsche Bank with cloud infrastructure support by Microsoft'

layout = html.Div(
    [
        run_model_page.nav_bar(),
        
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('AI for Good Simulator', className='card-title', style={'margin-bottom':'25px'}),
                            dcc.Link(html.Center('In partnership with Deutsche Bank and Microsoft'), id='partnership-information', href='#', target='_blank'),
                            html.P(''),
                            dbc.Tooltip(tooltip_partnership, target='partnership-information'), 
                            html.P(html.Center(text_introduction), className='card-text', style={'margin-bottom':'35px'}),
                            dbc.CardFooter(
                                html.Div([
                                    html.Center(dbc.Button('Get Started', id='landing-button', color='secondary', href='/sim/login_page')), 
                                ]), 
                            ), 
                            html.Div(id='landing-alert')
                            ], body=True
                        ), width=6
                    ), justify='center'
                )
            ])
        ], style={'padding':'100px'})
    ]
)
