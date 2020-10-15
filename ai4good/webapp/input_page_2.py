import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
from dash.dependencies import Input, Output, State, ALL
import dash_table

age = ['0 - 5', '6 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70+']
id_age_perc =['age-percentage-' + x.replace(' ','') for x in age]
dist = [10, 10, 10, 10, 10, 10, 10, 10, 20]

def generate_html_age_group(age, id_age_perc, dist):
    return html.Div([
               html.Header(''),
               html.Header([str(age)], className='card-text'), 
               dbc.Input(id={
                   'type':'age-dist-input',
                   'index':id_age_perc
               }, placeholder='Required', value=dist, type='number', min=0, max=100, step=1, bs_size='sm', style={'justify':'right', 'margin-bottom':'25px'}), 
           ], style={'display':'grid', 'grid-template-columns':'20% 60% 20%', 'margin-below':'25px'})

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
                            html.P('Enter the percentage of the total population that each range represents',className='card-text'),
                            html.H5('Age Group Distribution (%)', className='card-text'),
                            html.Div([
                                html.H5(''), 
                                html.H5('Age Group (years)'), 
                                html.H5('Distribution (%)')], 
                                style={'display':'grid', 'grid-template-columns':'10% 60% 30%'}),
                            generate_html_age_group(age[0], id_age_perc[0], dist[0]),
                            generate_html_age_group(age[1], id_age_perc[1], dist[1]),
                            generate_html_age_group(age[2], id_age_perc[2], dist[2]),
                            generate_html_age_group(age[3], id_age_perc[3], dist[3]),
                            generate_html_age_group(age[4], id_age_perc[4], dist[4]),
                            generate_html_age_group(age[5], id_age_perc[5], dist[5]),
                            generate_html_age_group(age[6], id_age_perc[6], dist[6]),
                            generate_html_age_group(age[7], id_age_perc[7], dist[7]),
                            generate_html_age_group(age[8], id_age_perc[8], dist[8]),
                            html.Div([
                                html.B(''), 
                                html.B('Total:'), 
                                html.B('100%', id='age_percentage_total')], 
                                style={'display':'grid', 'grid-template-columns':'20% 60% 20%', 'margin-below':'25px'}),
                            html.Div([
                                html.P(''), 
                                html.P('Must equal to 100%', className='card-text', style={'color':'darkgray'}), 
                                html.P('')], 
                                style={'display':'grid', 'grid-template-columns':'20% 60% 20%', 'margin-below':'25px'}),
                            dbc.CardFooter(dbc.Button('Next', id='page-2-button', color='dark', href='/sim/input_page_3')),
                            html.Div(id='input-page-2-alert')
                            ], body=True
                        ), width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)

@dash_app.callback(
    Output('age_percentage_total', 'children'), 
    [Input({'type':'age-dist-input', 'index':ALL}, 'value')])
def update_age_range_input_moved_off_site(values):
    return ([str(sum(values)) + '%'])
