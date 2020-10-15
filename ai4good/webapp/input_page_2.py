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

age = ['0 - 5', '6 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70+']
id_age_perc =['Age-percentage-' + x.replace(' ','') for x in age]
dist = [10, 10, 10, 10, 10, 10, 10, 10, 20]

def generate_html_age_group(age):
    return html.Header([str(age) + ' years'], className='card-text')

def generate_input_distribution(id_age_perc, dist):
    return dbc.Input(id=id_age_perc, placeholder='Required', value=dist, type='number', min=0, max=100, step=1, bs_size='sm', style={'margin-bottom':'25px'})

layout = html.Div(
    [
        run_model_page.nav_bar(),
 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/placeholder286x180.png', title='Step 2 of 4', style={'width':'50%'})), 
                            html.P('Enter the percentage of the total population that each range represents (total percentage should be 100)',className='card-text'),
                            html.H5('Age Group Distribution (%)', className='card-text'),
                            generate_html_age_group(age[0]),
                            generate_input_distribution(id_age_perc[0], dist[0]), 
                            generate_html_age_group(age[1]),
                            generate_input_distribution(id_age_perc[1], dist[1]), 
                            generate_html_age_group(age[2]),
                            generate_input_distribution(id_age_perc[2], dist[2]), 
                            generate_html_age_group(age[3]),
                            generate_input_distribution(id_age_perc[3], dist[3]), 
                            generate_html_age_group(age[4]),
                            generate_input_distribution(id_age_perc[4], dist[4]), 
                            generate_html_age_group(age[5]),
                            generate_input_distribution(id_age_perc[5], dist[5]), 
                            generate_html_age_group(age[6]),
                            generate_input_distribution(id_age_perc[6], dist[6]), 
                            generate_html_age_group(age[7]),
                            generate_input_distribution(id_age_perc[7], dist[7]), 
                            generate_html_age_group(age[8]),
                            generate_input_distribution(id_age_perc[8], dist[8]), 
                            html.Br(),
                            dbc.CardFooter(dbc.Button('Next', id='page-2-button', color='dark', href='/sim/input_page_3')),
                            html.Br(),
                            html.Div(id='input-page-2-alert')
                            ], body=True
                        ), width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)
