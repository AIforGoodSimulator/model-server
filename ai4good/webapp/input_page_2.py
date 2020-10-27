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
import ai4good.utils.path_utils as pu

age = ['0 - 5', '6 - 9', '10 - 19', '20 - 29', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70+']
id_age_popu =['age-population-' + x.replace(' ','') for x in age]
age_perc_start = [15, 15, 10, 10, 10, 10, 10, 10, 10] # starting age group population percentage
total_popu = 20000 # starting total population
err_group_total_not_equal_popu = 'Group total must equal to total population or 100%'

id_gender_perc = ['gender-perc-female', 'gender-perc-male']
id_ethnic_no_top = ['ethnic-no-1', 'ethnic-no-2', 'ethnic-no-3']
id_ethnic_no_mid = ['ethnic-no-4', 'ethnic-no-5', 'ethnic-no-6']
id_ethnic_no_dwn = ['ethnic-no-7', 'ethnic-no-8', 'ethnic-no-9']

accommodation_info = ['Type 1', 'Type 2', 'Type 3']
accommodation_info_full = ['Accommodation Type 1', 'Accommodation Type 2', 'Accommodation Type 3']
accommodation_info_required = ['Optional', 'Optional', 'Optional']
accommodation_info_detail = ['Area covered (m²)', 'No. of total camp residents in this type of accommodation', 'No. of existing units of accommodation']
tab_id_accommodation_info = ['tab-accommodation-info-' + x.replace(' ','').lower() for x in accommodation_info]
id_accommodation_area = ['accommodation-area-' + x.replace(' ','').lower() for x in accommodation_info]
id_accommodation_no_person = ['accommodation-no-person-' + x.replace(' ','').lower() for x in accommodation_info]
id_accommodation_no_unit = ['accommodation-no-unit-' + x.replace(' ','').lower() for x in accommodation_info]

def int_perc_1dp(nom, dem):
    perc = nom/dem*100 if dem !=0 else 0
    perc_1dp = "{:.1f}".format(perc)
    return str(perc_1dp)+'%'

def generate_html_age_group(age, id_age_popu, age_perc, total_popu):
    return html.Div([
               html.Header(''),
               html.Header([str(age)], className='card-text'), 
               dbc.Input(id={
                   'type':'age-popu-input',
                   'index':id_age_popu
               }, placeholder='Required', type='number', value=[], min=0, max=total_popu, step=1, bs_size='sm', style={'justify':'right', 'margin-bottom':'5px'}), 
               dcc.Slider(id={
                   'type':'age-popu-slider', 
                   'index':id_age_popu
               }, value=[], min=0, max=total_popu, step=1, updatemode='drag'),
               dbc.Label(str(age_perc)+'%', id={
                   'type':'age-perc-label', 
                   'index':id_age_popu}), 
    ], id=id_age_popu, style={'display':'grid', 'grid-template-columns':'5% 20% 20% 43% 12%', 'margin-bottom':'5px'})

def generate_three_column_input(id_three_column, max_no, margin_bottom='25px'):
    children = html.Div([
        dbc.Input(id=id_three_column[0], placeholder='Optional', type='number', min=0, max=max_no, step=1, n_submit=0, bs_size='sm'),
        html.Header(''),
        dbc.Input(id=id_three_column[1], placeholder='Optional', type='number', min=0, max=max_no, step=1, n_submit=0, bs_size='sm'),
        html.Header(''),
        dbc.Input(id=id_three_column[2], placeholder='Optional', type='number', min=0, max=max_no, step=1, n_submit=0, bs_size='sm'), 
    ], style={'display':'grid', 'grid-template-columns':'28% 8% 28% 8% 28%', 'margin-bottom':margin_bottom})
    return children

def generate_accommodation_info_children(id_index):
    children = [html.Div([
        html.P(''), 
        dbc.Label(accommodation_info_full[id_index], color='secondary'), 
        html.Header(accommodation_info_detail[0], className='card-text'), 
        dbc.Input(id=id_accommodation_area[id_index], placeholder=accommodation_info_required[id_index], type='number', min=0, max=1000000, step=100, bs_size='sm', style={'margin-bottom':'25px'}),        
        html.Header(accommodation_info_detail[1], className='card-text'), 
        dbc.Input(id=id_accommodation_no_person[id_index], placeholder=accommodation_info_required[id_index], type='number', min=0, max=100000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),        
        html.Header(accommodation_info_detail[2], className='card-text'), 
        dbc.Input(id=id_accommodation_no_unit[id_index], placeholder=accommodation_info_required[id_index], type='number', min=0, max=10000, step=1, bs_size='sm', style={'margin-bottom':'25px'}),        
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
                            html.Center(html.Img(src='/static/input_step2.png', title='Step 2 of 4', style={'width':'50%'}, className="step_counter")), 
                            html.P('Fill in the following about the age structure accommodation type of the settlement',className='card-text'),
                            html.H5('Population', className='card-text'),
                            html.Header('Total Population', className='card-text'),
                            html.Header('What is the total population in the camp, rounding off to the nearest 10?',className='card-text', style={'color':'darkgray'}),
                            dbc.Input(id='total-population', placeholder='Required', value=20000, type='number', min=0, max=100000, step=10, n_submit=0, bs_size='sm', style={'margin-bottom':'25px'}),
                            html.Header('Age Group Structure', className='card-text'),
                            html.Header('Enter the percentage or actual population that each age range represents',className='card-text', style={'color':'darkgray'}),
                            html.Div([
                                html.Header(''),
                                html.Header('Age Group (years)'), 
                                html.Header('Population'), 
                                html.Header('Distribution'), 
                                html.Header('(%)')], 
                                style={'display':'grid', 'grid-template-columns':'5% 20% 35% 33% 7%'}),
                            generate_html_age_group(age[0], id_age_popu[0], age_perc_start[0], total_popu),
                            generate_html_age_group(age[1], id_age_popu[1], age_perc_start[1], total_popu),
                            generate_html_age_group(age[2], id_age_popu[2], age_perc_start[2], total_popu),
                            generate_html_age_group(age[3], id_age_popu[3], age_perc_start[3], total_popu),
                            generate_html_age_group(age[4], id_age_popu[4], age_perc_start[4], total_popu),
                            generate_html_age_group(age[5], id_age_popu[5], age_perc_start[5], total_popu),
                            generate_html_age_group(age[6], id_age_popu[6], age_perc_start[6], total_popu),
                            generate_html_age_group(age[7], id_age_popu[7], age_perc_start[7], total_popu),
                            generate_html_age_group(age[8], id_age_popu[8], age_perc_start[8], total_popu),
                            html.Div([
                                html.B(''), 
                                html.B('Group Total:'), 
                                html.B('', id='age_population_total'), 
                                html.B('Percentage Total:'), 
                                html.B('', id='age_percentage_total')], className='card-text', 
                                style={'display':'grid', 'grid-template-columns':'5% 25% 26% 30% 14%'}),
                            html.Div([
                                html.B(''), 
                                dbc.Label('Must equal to total population or 100%', id='age-group-continue-warning', color='secondary'), 
                                dbc.Button('Default %', size='sm', color='secondary', id='age-default-perc', style={'float':'right'})], className='card-text', style={'display':'grid', 'grid-template-columns':'5% 77% 18%'}),
                            html.P(''),
                            html.Header('Male and Female Population', className='card-text'),
                            html.Div([
                                html.B('Female (%): '), 
                                html.B('Male (%):')], 
                                style={'display':'grid', 'grid-template-columns':'90% 10%', 'color':'darkgray'}), 
                            html.Div([
                                html.Label('', id='gender-perc-female'), 
                            dcc.Slider(id='slider-gender-perc', min=0, max=100, step=1, value=50, included=False, updatemode='drag', marks={50: {'label':'50'}}), 
                                html.Label('', id='gender-perc-male')], 
                                style={'display':'grid', 'grid-template-columns':'10% 80% 10%', 'margin-bottom':'25px'}),                            
                            html.Header('Population by Ethnicity', className='card-text'),
                            html.Header('Enter the population represented by each ethnic group',className='card-text', style={'color':'darkgray'}),
                            generate_three_column_input(id_ethnic_no_top, 10000,'10px'), 
                            generate_three_column_input(id_ethnic_no_mid, 10000,'10px'), 
                            generate_three_column_input(id_ethnic_no_dwn, 10000), 
                            html.P(''),
                            html.H5('Accommodation Information', className='card-text'),
                            html.Header('Population Accommodation', className='card-text'),
                            html.Header('Accommodation details provide population density estimate and suggest mobility pattern', className='card-text', style={'color':'darkgray'}), 
                            html.Div([
                                dbc.Tabs([
                                    dbc.Tab(label=accommodation_info[0], tab_id=tab_id_accommodation_info[0], children=generate_accommodation_info_children(0)), 
                                    dbc.Tab(label=accommodation_info[1], tab_id=tab_id_accommodation_info[1], children=generate_accommodation_info_children(1)), 
                                    dbc.Tab(label=accommodation_info[2], tab_id=tab_id_accommodation_info[2], children=generate_accommodation_info_children(2)), 
                                ], id='tabs-accommodation-info', active_tab=tab_id_accommodation_info[0]), 
                            ], style={'border':'1px lightgray solid'}),
                            html.P(''),
                            dbc.CardFooter(dbc.Button('Next', id='page-2-button', color='secondary', disabled=False, href='/sim/input_page_3', style={'float':'right'})), 
                            dbc.Label('',id='page-2-continue-warning', color='danger', style={'text-align':'right'}), 
                            html.Div(id='input-page-2-alert')
                            ], body=True), 
                        html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)

@dash_app.callback(
    [Output('gender-perc-female','children'), Output('gender-perc-male','children')], 
    [Input('slider-gender-perc','value')])
def update_gender_perc_label(slider_value):
    perc_female = slider_value
    perc_male = 100 - slider_value
    return str(perc_female), str(perc_male)

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
    [Output('age_population_total', 'children'), Output('age_percentage_total', 'children'), Output('page-2-button', 'disabled'), Output('page-2-continue-warning', 'children'), Output('age-group-continue-warning', 'color')], 
    [Input({'type':'age-popu-input', 'index':ALL}, 'value'), Input({'type':'age-popu-slider', 'index':ALL}, 'value')], 
    [State('total-population', 'value')])
def update_age_group_total(input_values, slider_values, total_value):
    sum_input = sum(input_values)
    sum_slider = sum(slider_values)
    sum_perc_str = int_perc_1dp(sum_input, total_value)
    if (sum_input==total_value):
        return str(sum_input), sum_perc_str, False, '', 'secondary'
    else:
        return str(sum_input), sum_perc_str, True, err_group_total_not_equal_popu, 'danger'

@dash_app.callback(
    [Output({'type':'age-popu-input', 'index':ALL}, 'max'), Output({'type':'age-popu-slider', 'index':ALL}, 'max'), Output({'type':'age-popu-slider', 'index':ALL}, 'value'), Output('slider-gender-perc','value')], 
    [Input('total-population', 'value'), Input('age-default-perc','n_clicks')],
    [State({'type':'age-popu-input', 'index':ALL}, 'value'), State('total-population', 'min'), State('total-population', 'max'), State('gender-perc-female', 'children'), State('gender-perc-male','children'), State('page-2-button', 'n_clicks')])
def update_age_popu_max(total_value, default_n_clicks, input_values, total_min, total_max, gender_perc_female, gender_perc_male, next_n_clicks):
    inputParameterCache = InputParameterCache(_redis)
    context = dash.callback_context
    if total_value is None:
        raise PreventUpdate
    elif total_value <= total_min:
        raise PreventUpdate
    elif total_value > total_max:
        raise PreventUpdate
    else:
        updated_maxs = [total_value]*len(input_values)
        gender_values = [gender_perc_female, gender_perc_male]
        gender_reset_values = [50, 50]
        slider_reset_values = [round(total_value*x/100) for x in age_perc_start]
        if sum(slider_reset_values) != total_value:  # reset last age group population due to rounding off
            slider_reset_values[-1] = total_value - sum(slider_reset_values) + slider_reset_values[-1]
        if not context.triggered:
            slider_values = slider_reset_values
        else:
            id_trig = context.triggered[0]['prop_id'].split('.')[0]
            if id_trig == 'total-population':
                if next_n_clicks is None:  # age group population and gender percentage update from cache
                    input_param_age_group = {
                        id_age_popu[0]: 0, 
                        id_age_popu[1]: 0, 
                        id_age_popu[2]: 0, 
                        id_age_popu[3]: 0, 
                        id_age_popu[4]: 0, 
                        id_age_popu[5]: 0, 
                        id_age_popu[6]: 0, 
                        id_age_popu[7]: 0, 
                        id_age_popu[8]: 0, 
                    }
                    input_param_gender = {
                        'gender-perc-female': 0, 
                        'gender-perc-male': 0, 
                    }
                    slider_values = inputParameterCache.cache_get(input_param_age_group.keys())
                    gender_values = inputParameterCache.cache_get(input_param_gender.keys())
                    if None in slider_values:
                        slider_values = slider_reset_values
                    if None in gender_values:
                        gender_values = gender_reset_values
                else:
                    slider_values = slider_reset_values
            elif id_trig == 'age-default-perc':
                slider_values = slider_reset_values
            else: # should not happen
                raise PreventUpdate
            return updated_maxs, updated_maxs, slider_values, gender_values[0]

@dash_app.callback(
    [Output('total-population', 'value'), 
     Output('ethnic-no-1', 'value'), Output('ethnic-no-2', 'value'), Output('ethnic-no-3', 'value'), 
     Output('ethnic-no-4', 'value'), Output('ethnic-no-5', 'value'), Output('ethnic-no-6', 'value'), 
     Output('ethnic-no-7', 'value'), Output('ethnic-no-8', 'value'), Output('ethnic-no-9', 'value'), 
     Output('accommodation-area-type1', 'value'), Output('accommodation-area-type2', 'value'), Output('accommodation-area-type3', 'value'), 
     Output('accommodation-no-person-type1', 'value'), Output('accommodation-no-person-type2', 'value'), Output('accommodation-no-person-type3', 'value'), 
     Output('accommodation-no-unit-type1', 'value'), Output('accommodation-no-unit-type2', 'value'), Output('accommodation-no-unit-type3', 'value')], 
    [Input('page-2-button', 'n_clicks')], 
    [State('total-population', 'value'), State({'type':'age-popu-input','index':ALL}, 'value'), 
     State('gender-perc-female', 'children'), State('gender-perc-male', 'children'), 
     State('ethnic-no-1', 'value'), State('ethnic-no-2', 'value'), State('ethnic-no-3', 'value'), 
     State('ethnic-no-4', 'value'), State('ethnic-no-5', 'value'), State('ethnic-no-6', 'value'), 
     State('ethnic-no-7', 'value'), State('ethnic-no-8', 'value'), State('ethnic-no-9', 'value'), 
     State('accommodation-area-type1', 'value'), State('accommodation-area-type2', 'value'), State('accommodation-area-type3', 'value'), 
     State('accommodation-no-person-type1', 'value'), State('accommodation-no-person-type2', 'value'), State('accommodation-no-person-type3', 'value'), 
     State('accommodation-no-unit-type1', 'value'), State('accommodation-no-unit-type2', 'value'), State('accommodation-no-unit-type3', 'value')])
def update_input_parameter_page_2(
    n_clicks, total_population_value, age_popu_input_values, 
    gender_perc_female_value, gender_perc_male_value, 
    ethnic_no_1_value, ethnic_no_2_value, ethnic_no_3_value, 
    ethnic_no_4_value, ethnic_no_5_value, ethnic_no_6_value, 
    ethnic_no_7_value, ethnic_no_8_value, ethnic_no_9_value, 
    accommodation_area_type1_value, accommodation_area_type2_value, accommodation_area_type3_value, 
    accommodation_no_person_type1_value, accommodation_no_person_type2_value, accommodation_no_person_type3_value, 
    accommodation_no_unit_type1_value, accommodation_no_unit_type2_value, accommodation_no_unit_type3_value
):
    inputParameterCache = InputParameterCache(_redis)
    input_param = {
        'total-population': total_population_value, 
        id_age_popu[0]: age_popu_input_values[0], 
        id_age_popu[1]: age_popu_input_values[1], 
        id_age_popu[2]: age_popu_input_values[2], 
        id_age_popu[3]: age_popu_input_values[3], 
        id_age_popu[4]: age_popu_input_values[4], 
        id_age_popu[5]: age_popu_input_values[5], 
        id_age_popu[6]: age_popu_input_values[6], 
        id_age_popu[7]: age_popu_input_values[7], 
        id_age_popu[8]: age_popu_input_values[8], 
        'gender-perc-female': gender_perc_female_value, 
        'gender-perc-male': gender_perc_male_value, 
        'ethnic-no-1': ethnic_no_1_value, 
        'ethnic-no-2': ethnic_no_2_value, 
        'ethnic-no-3': ethnic_no_3_value, 
        'ethnic-no-4': ethnic_no_4_value, 
        'ethnic-no-5': ethnic_no_5_value, 
        'ethnic-no-6': ethnic_no_6_value, 
        'ethnic-no-7': ethnic_no_7_value, 
        'ethnic-no-8': ethnic_no_8_value, 
        'ethnic-no-9': ethnic_no_9_value, 
        'accommodation-area-type1': accommodation_area_type1_value, 
        'accommodation-area-type2': accommodation_area_type2_value, 
        'accommodation-area-type3': accommodation_area_type3_value, 
        'accommodation-no-person-type1': accommodation_no_person_type1_value, 
        'accommodation-no-person-type2': accommodation_no_person_type2_value, 
        'accommodation-no-person-type3': accommodation_no_person_type3_value, 
        'accommodation-no-unit-type1': accommodation_no_unit_type1_value, 
        'accommodation-no-unit-type2': accommodation_no_unit_type2_value, 
        'accommodation-no-unit-type3': accommodation_no_unit_type3_value, 
    }
    if n_clicks is None:
        value = inputParameterCache.cache_get(input_param.keys())  # get cached value
        for i,j in enumerate(value):
            if j is None:  # if first time loading, get default value
                if 'total-population' == list(input_param.keys())[i]:
                    value[i] = total_popu
                else:
                    value[i] = None
        del value[1:12]  # remove age group population and gender percentage cache
        return value
    else:
        inputParameterCache.cache_set(input_param, 2)  # put all input parameters in input page 2 to cache
        raise PreventUpdate
