<<<<<<< HEAD
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from ai4good.webapp.apps import dash_app, facade, model_runner, _redis
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import InputParameterCache
import ai4good.webapp.run_model_for_dashboard as run_model_for_dashboard
import ai4good.utils.path_utils as pu

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
            options = [{'label':y, 'value':x} for x,y in enumerate(health_intervent_option)], value=not_sure_effectiveness, id=radio_id_health_intervent[id_index])
    ], style={'margin-left':'25px', 'margin-right':'25px', 'margin-bottom':'20px'})]
    return children

def generate_activity_gathering_children(id_index):
    children = [html.Div([
        html.P(''), 
        dbc.Label(activity_gathering_full[id_index], color='secondary'), 
        html.Header(activity_gaterhing_detail[0], className='card-text'), 
        dbc.Input(id=id_activity_no_place[id_index], placeholder=activity_gathering_required[id_index], type='number', min=0, max=1000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),        
        html.Header(activity_gaterhing_detail[1], className='card-text'), 
        dbc.Input(id=id_activity_no_person[id_index], placeholder=activity_gathering_required[id_index], type='number', min=0, max=100000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),        
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
                            dbc.CardFooter(dbc.Button('Run Simulation', id='page-4-button', color='secondary', href='/sim/waiting', style={'float':'right'})),
                            html.Div(id='input-page-4-alert')
                            ], body=True), 
                        html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)

@dash_app.callback(
    [Output('radio-intervene-social', 'value'), Output('radio-intervene-face', 'value'), 
     Output('radio-intervene-handwashing', 'value'), Output('radio-intervene-testing', 'value'), 
     Output('radio-intervene-lockdown', 'value'), Output('activity-no-place-admin', 'value'), 
     Output('activity-no-place-food', 'value'), Output('activity-no-place-health', 'value'), 
     Output('activity-no-place-recreational', 'value'), Output('activity-no-place-religious', 'value'), 
     Output('activity-no-person-admin','value'), Output('activity-no-person-food','value'), 
     Output('activity-no-person-health','value'), Output('activity-no-person-recreational','value'), 
     Output('activity-no-person-religious','value'), Output('activity-no-visit-admin','value'), 
     Output('activity-no-visit-food','value'), Output('activity-no-visit-health','value'), 
     Output('activity-no-visit-recreational','value'), Output('activity-no-visit-religious','value')], 
    [Input('page-4-button', 'n_clicks')], 
    [State('radio-intervene-social', 'value'), State('radio-intervene-face', 'value'), 
     State('radio-intervene-handwashing', 'value'), State('radio-intervene-testing', 'value'), 
     State('radio-intervene-lockdown', 'value'), State('activity-no-place-admin', 'value'), 
     State('activity-no-place-food', 'value'), State('activity-no-place-health', 'value'), 
     State('activity-no-place-recreational', 'value'), State('activity-no-place-religious', 'value'), 
     State('activity-no-person-admin','value'), State('activity-no-person-food','value'), 
     State('activity-no-person-health','value'), State('activity-no-person-recreational','value'), 
     State('activity-no-person-religious','value'), State('activity-no-visit-admin','value'), 
     State('activity-no-visit-food','value'), State('activity-no-visit-health','value'), 
     State('activity-no-visit-recreational','value'), State('activity-no-visit-religious','value')])
def update_input_parameter_page_4(
    n_clicks, radio_intervene_social_value, radio_intervene_face_value, 
    radio_intervene_handwashing_value, radio_intervene_testing_value, radio_intervene_lockdown_value, 
    activity_no_place_admin_value, activity_no_place_food_value, activity_no_place_health_value, 
    activity_no_place_recreational_value, activity_no_place_religious_value, 
    activity_no_person_admin_value, activity_no_person_food_value, activity_no_person_health_value, 
    activity_no_person_recreational_value, activity_no_person_religious_value, 
    activity_no_visit_admin_value, activity_no_visit_food_value, activity_no_visit_health_value, 
    activity_no_visit_recreational_value, activity_no_visit_religious_value):
    inputParameterCache = InputParameterCache(_redis)
    input_param = {
        'radio-intervene-social': radio_intervene_social_value, 
        'radio-intervene-face': radio_intervene_face_value, 
        'radio-intervene-handwashing': radio_intervene_handwashing_value, 
        'radio-intervene-testing': radio_intervene_testing_value, 
        'radio-intervene-lockdown': radio_intervene_lockdown_value, 
        'activity-no-place-admin': activity_no_place_admin_value, 
        'activity-no-place-food': activity_no_place_food_value, 
        'activity-no-place-health': activity_no_place_health_value, 
        'activity-no-place-recreational': activity_no_place_recreational_value, 
        'activity-no-place-religious': activity_no_place_religious_value, 
        'activity-no-person-admin': activity_no_person_admin_value, 
        'activity-no-person-food': activity_no_person_food_value, 
        'activity-no-person-health': activity_no_person_health_value, 
        'activity-no-person-recreational': activity_no_person_recreational_value, 
        'activity-no-person-religious': activity_no_person_religious_value, 
        'activity-no-visit-admin': activity_no_visit_admin_value, 
        'activity-no-visit-food': activity_no_visit_food_value, 
        'activity-no-visit-health': activity_no_visit_health_value, 
        'activity-no-visit-recreational': activity_no_visit_recreational_value, 
        'activity-no-visit-religious': activity_no_visit_religious_value, 
    }
    if n_clicks is None:
        value = inputParameterCache.cache_get(input_param.keys())  # get cached value
        for i,j in enumerate(value):
            if j is None:  # if first time loading, get default value
                if list(input_param.keys())[i] in {'radio-intervene-social', 'radio-intervene-face', 'radio-intervene-handwashing', 'radio-intervene-testing', 'radio-intervene-lockdown'}:
                    value[i] = not_sure_effectiveness
                else:
                    value[i] = None
        return value
    else:
        inputParameterCache.cache_set(input_param, 4)  # put all input parameters in input page 4 to cache
        raise PreventUpdate

@dash_app.callback(Output('input-page-4-alert', 'children'),[Input('page-4-button', 'n_clicks')])
def model_Dashboard(n_clicks):
    if n_clicks != 0:
        run_model_for_dashboard.run_model_results_for_message("message_1")
        run_model_for_dashboard.run_model_results_for_message("message_5")