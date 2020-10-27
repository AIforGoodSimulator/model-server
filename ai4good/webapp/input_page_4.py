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

not_sure_effectiveness = 0

health_intervent = ['Social', 'Face', 'Handwashing', 'Testing', 'Lockdown']
health_intervent_full = ['Social Distancing', 'Face Covering / Mask Wearing', 'Handwashing / Soap / Handwashing Facilities', 'Testing for Infectious Diseases', 'Lockdown / Movement Restrictions']
health_intervent_option = ['0 Not sure', '1 Not Effective', '2 Somewhat Effective', '3 Effective', '4 Very Effective']
health_intervent_required = ['Required', 'Required', 'Required', 'Optional', 'Optional']
tab_id_health_intervent = ['tab-intervene-' + x.lower() for x in health_intervent]
radio_id_health_intervent = ['radio-intervene-' + x.lower() for x in health_intervent]

activity_gathering = ['Admin', 'Food', 'Health', 'Recreational', 'Religious']
activity_gathering_full = ['Administrative Activities such as Cash Aid and Immigration Support', 'Food Provision and Distribution Queue', 'Health Clinic Visit, Health Check and Consultation', 'Recreational Activities and Events such as Hanging Out and Playing Football', 'Religious Gatherings and Events']
activity_gathering_required = ['Optional', 'Optional', 'Optional', 'Optional', 'Optional']
activity_gaterhing_detail = ['No. of Places', 'No. of Individuals Participating', 'No. of Visits per Individual (average per week)']
tab_id_activity_gathering = ['tab-activity-' + x.lower() for x in activity_gathering]
id_activity_no_place = ['activity-no-place-' + x.lower() for x in activity_gathering]
id_activity_no_person = ['activity-no-person-' + x.lower() for x in activity_gathering]
id_activity_no_visit = ['activity-no-visit-' + x.lower() for x in activity_gathering]

def generate_health_intervent_children(id_index):
    children = [html.Div([
        html.P(''), 
        dbc.Label(health_intervent_full[id_index], color='secondary'), 
        dbc.Container(dbc.Label(health_intervent_required[id_index], color='secondary', size='sm')), 
        dbc.RadioItems(
            options = [{'label':y, 'value':} for x,y in enumerate(health_intervent_option)], value=not_sure_effectiveness, id=radio_id_health_intervent[id_index])
    ], style={'margin-left':'25px', 'margin-right':'25px', 'margin-bottom':'20px'})]
    return children

def generate_activity_gathering_children(id_index):
    children = [html.Div([
        html.P(''), 
        dbc.Label(activity_gathering_full[id_index], color='secondary'), 
        html.Header(activity_gaterhing_detail[0], className='card-text'), 
        dbc.Input(id=id_activity_no_place[id_index], placeholder=activity_gathering_required[id_index], type='number', min=0, max=1000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),        
        html.Header(activity_gaterhing_detail[1], className='card-text'), 
        dbc.Input(id=id_activity_no_person[id_index], placeholder=activity_gathering_required[id_index], type='number', min=0, max=100000, step=100, bs_size='sm', style={'margin-bottom':'25px'}),        
        html.Header(activity_gaterhing_detail[2], className='card-text'), 
        dbc.Input(id=id_activity_no_visit[id_index], placeholder=activity_gathering_required[id_index], type='number', min=0, max=100, step=1, bs_size='sm', style={'margin-bottom':'25px'}),        
    ], style={'margin-left':'25px', 'margin-right':'25px', 'margin-bottom':'20px'})]
    return children

layout = html.Div(
    [
        run_model_page.nav_bar(),
 
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/input_step4.png', title='Step 4 of 4', style={'width':'50%'}, className="step_counter")), 
                            html.P('Provide your best estimate if data is not available', className='card-text'),
                            html.H5('Health Interventions (Part II)', className='card-text'),
                            html.Header('Effectiveness of Interventions', className='card-text'),
                            html.Header('How effective are the following measures at the camp?', className='card-text', style={'color':'darkgray'}), 
                            html.Div([
                                dbc.Tabs([
                                    dbc.Tab(label=health_intervent[0], tab_id=tab_id_health_intervent[0], children=generate_health_intervent_children(0)), 
                                    dbc.Tab(label=health_intervent[1], tab_id=tab_id_health_intervent[1], children=generate_health_intervent_children(1)), 
                                    dbc.Tab(label=health_intervent[2], tab_id=tab_id_health_intervent[2], children=generate_health_intervent_children(2)), 
                                    dbc.Tab(label=health_intervent[3], tab_id=tab_id_health_intervent[3], children=generate_health_intervent_children(3)), 
                                    dbc.Tab(label=health_intervent[4], tab_id=tab_id_health_intervent[4], children=generate_health_intervent_children(4)), 
                                ], id='tabs-health-intervention', active_tab=tab_id_health_intervent[0]), 
                            ], style={'border':'1px lightgray solid'}),
                            html.P(''),
                            html.P(''),
                            html.H5('Human Interactions', className='card-text'),
                            html.Header('Activities and Gatherings', className='card-text'),
                            html.Header('What are scope of acitivities and gatherings at the camp?', className='card-text', style={'color':'darkgray'}), 
                            html.Div([
                                dbc.Tabs([
                                    dbc.Tab(label=activity_gathering[0], tab_id=tab_id_activity_gathering[0], children=generate_activity_gathering_children(0)), 
                                    dbc.Tab(label=activity_gathering[1], tab_id=tab_id_activity_gathering[1], children=generate_activity_gathering_children(1)), 
                                    dbc.Tab(label=activity_gathering[2], tab_id=tab_id_activity_gathering[2], children=generate_activity_gathering_children(2)), 
                                    dbc.Tab(label=activity_gathering[3], tab_id=tab_id_activity_gathering[3], children=generate_activity_gathering_children(3)), 
                                    dbc.Tab(label=activity_gathering[4], tab_id=tab_id_activity_gathering[4], children=generate_activity_gathering_children(4)), 
                                ], id='tabs-activity-gathering', active_tab=tab_id_activity_gathering[0]), 
                            ], style={'border':'1px lightgray solid'}),
                            html.P(''),
                            dbc.CardFooter(dbc.Button('Run Simulation', id='page-4-button', color='secondary', href='/sim/run_model', style={'float':'right'})),
                            html.Div(id='input-page-4-alert')
                            ], body=True), 
                        html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)

