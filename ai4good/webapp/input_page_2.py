import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import ai4good.utils.path_utils as pu

age = ['0 - 5', '6 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70+']
id_age_popu =['age-population-' + x.replace(' ','') for x in age]
age_perc = [10, 10, 10, 10, 10, 10, 10, 10, 20] # starting age group population percentage
total_popu = 20000 # starting total population

def int_perc_1dp(nom, dem):
    perc = nom/dem*100 if dem !=0 else 0
    perc_1dp = "{:.1f}".format(perc)
    return str(perc_1dp)+'%'

def generate_html_age_group(age, id_age_popu, age_perc, total_popu):
    age_group_popu = total_popu*age_perc/100
    return html.Div([
               html.Header(''),
               html.Header([str(age)], className='card-text'), 
               dbc.Input(id={
                   'type':'age-popu-input',
                   'index':id_age_popu
               }, placeholder='Required', type='number', value=age_group_popu, min=0, max=total_popu, step=1, bs_size='sm', style={'justify':'right', 'margin-bottom':'25px'}), 
               dcc.Slider(id={
                   'type':'age-popu-slider', 
                   'index':id_age_popu
               }, value=age_group_popu, min=0, max=total_popu, step=1, updatemode='drag'),
               dbc.Label(str(age_perc)+'%', id={
                   'type':'age-perc-label', 
                   'index':id_age_popu}), 
    ], id=id_age_popu, style={'display':'grid', 'grid-template-columns':'5% 20% 20% 45% 10%', 'margin-below':'25px'})

layout = html.Div(
    [
        run_model_page.nav_bar(),
 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/input_step2.png', title='Step 2 of 4', style={'width':'50%'})), 
                            html.P('Fill in the following about the age structure of the settlement',className='card-text'),
                            html.H5('Age Group Distribution', className='card-text'),
                            html.Header('Total Population', className='card-text'),
                            dbc.Input(id='total-population', placeholder='Required', value=20000, type='number', min=0, max=100000, step=10, n_submit=0, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Age Structure', className='card-text'),
                            html.Header('Enter the percentage or actual population that each range represents',className='card-text', style={'color':'darkgray'}),
                            html.Div([
                                html.Header(''),
                                html.Header('Age Group (years)'), 
                                html.Header('Population'), 
                                html.Header('Distribution'), 
                                html.Header('(%)')], 
                                style={'display':'grid', 'grid-template-columns':'5% 20% 35% 35% 5%'}),
                            generate_html_age_group(age[0], id_age_popu[0], age_perc[0], total_popu),
                            generate_html_age_group(age[1], id_age_popu[1], age_perc[1], total_popu),
                            generate_html_age_group(age[2], id_age_popu[2], age_perc[2], total_popu),
                            generate_html_age_group(age[3], id_age_popu[3], age_perc[3], total_popu),
                            generate_html_age_group(age[4], id_age_popu[4], age_perc[4], total_popu),
                            generate_html_age_group(age[5], id_age_popu[5], age_perc[5], total_popu),
                            generate_html_age_group(age[6], id_age_popu[6], age_perc[6], total_popu),
                            generate_html_age_group(age[7], id_age_popu[7], age_perc[7], total_popu),
                            generate_html_age_group(age[8], id_age_popu[8], age_perc[8], total_popu),
                            html.Div([
                                html.B(''), 
                                html.B('Group Total:'), 
                                html.B('', id='age_population_total'), 
                                html.B('Percentage Total:'), 
                                html.B('', id='age_percentage_total')], 
                                style={'display':'grid', 'grid-template-columns':'5% 25% 26% 32% 12%', 'margin-below':'25px'}),
                            html.Div(
                                html.P(html.Center('Must equal to total population or 100%', className='card-text', style={'color':'darkgray'}))),
                            dbc.CardFooter(dbc.Button('Next', id='page-2-button', color='secondary', disabled=False, href='/sim/input_page_3')),
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
    [Output({'type':'age-perc-label', 'index':MATCH}, 'children')], 
    [Input({'type':'age-popu-input', 'index':MATCH}, 'value')], 
    [State('total-population', 'value')])
def update_age_group_label(input_value, total_value):
    updated_perc_str = int_perc_1dp(input_value, total_value)
    return [updated_perc_str]

@dash_app.callback(
    [Output({'type':'age-popu-input', 'index':MATCH}, 'value')], 
    [Input({'type':'age-popu-slider', 'index':MATCH}, 'value')])
def update_age_group_input(input_value):
    return [input_value]

@dash_app.callback(
    [Output('age_population_total', 'children'), Output('age_percentage_total', 'children'), Output('page-2-button', 'disabled')], 
    [Input({'type':'age-popu-input', 'index':ALL}, 'value'), Input({'type':'age-popu-slider', 'index':ALL}, 'value')], 
    [State('total-population', 'value')])
def update_age_group_total(input_values, slider_values, total_value):
    sum_input = sum(input_values)
    sum_slider = sum(slider_values)
    sum_perc_str = int_perc_1dp(sum_input, total_value)
    if (sum_input==total_value):
        return str(sum_input), sum_perc_str, False
    else:
        return str(sum_input), sum_perc_str, True

@dash_app.callback(
    [Output({'type':'age-popu-input', 'index':ALL}, 'max'), Output({'type':'age-popu-slider', 'index':ALL}, 'max'), Output({'type':'age-popu-slider', 'index':ALL}, 'value')], 
    [Input('total-population', 'value')],
    [State({'type':'age-popu-input', 'index':ALL}, 'value'), State('total-population', 'min'), State('total-population', 'max')])
def update_age_popu_max(total_value, input_values, total_min, total_max):
    updated_maxs = [total_value]*len(input_values)
    if total_value is None:
        raise PreventUpdate
    elif total_value <= total_min:
        raise PreventUpdate
    elif total_value > total_max:
        raise PreventUpdate
    else:
        return updated_maxs, updated_maxs, input_values
