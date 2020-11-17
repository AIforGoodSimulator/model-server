import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import os
import json
import ai4good.utils.path_utils as pu
from ai4good.utils.logger_util import get_logger
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.models.validate.model_validation_metrics import model_validation_metrics
from ai4good.models.validate.model_validation_plots import model_validation_plots
from ai4good.models.validate.initialise_parameters import Parameters
import ai4good.webapp.common_elements as common_elements

logger = get_logger(__name__)

# get output paths
population = 18700

output_filetype = 'csv'
base_sample = '../../ai4good'
base_output = base_sample
path_sample = pu._path(f'{base_sample}/models/validate/output_sample/', '')
path_output = path_sample
sample_output = [f for f in os.listdir(path_sample) if os.path.isfile(os.path.join(path_sample, f))]
sample_output_clean = sorted([f.split('.')[0] for f in sample_output])

# create tooltip
metric_tooltip = 'Key: \n' + \
    ' MAPE = Mean absolute percentage error \n' + \
    ' RMSE = Root mean square error \n' + \
    ' MSE  = Mean square error \n' + \
    ' MeanAE = Mean average error \n' + \
    ' MedianAE = Median average error \n' + \
    ' R2_Score = Coefficient of determination (R squared) \n' + \
    ' MSLE = Mean square logarithmic error'
plot_tooltip = 'To isolate two traces, double click on one in the legend \n' + \
    'then single click on the second one to show'

# initialise metric filter
age_categories = []
age_categories_dropdown = []
case_cols_dropdown = []
case_cols = []

# output for validation results
layout = html.Div(
    [
        common_elements.nav_bar(),
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('Model Validation', className='card-title'), 
                            html.P('Model validation comparison to baseline', className='card-text'), 
                            html.H5('Validation display', className='card-text'), 
                            html.Header('Choose baseline and model outputs', className='card-text'), 
                            html.Small(dcc.Dropdown(options=[{'label': k, 'value': k} for k in sample_output_clean], id='validate-baseline-output-dropdown', placeholder='Baseline output', style={'width':'60%'})), 
                            html.Small(dcc.Dropdown(options=[{'label': k, 'value': k} for k in sample_output_clean], id='validate-model-output-dropdown', placeholder='Model output', style={'width':'60%'})), 
                            html.Div(html.Small(id='validate-model-status-text', style={'margin-bottom':'20px'})), 
                            html.Header('Choose age group and case', className='card-text'), 
                            html.Small(dcc.Dropdown(options=[], id='validate-age-dropdown', placeholder='Age group (years)', style={'width':'60%'})), 
                            html.Small(dcc.Dropdown(options=[], id='validate-case-dropdown', placeholder='Case group', style={'width':'60%', 'margin-bottom':'25px'})), 
                            dcc.Loading(children=[
                                html.H5([
                                    html.B('Validation metrics '), 
                                    html.Abbr('\u2139', title=metric_tooltip), 
                                ], className='card-text'), 
                                html.Div(html.Small(id='validate-metric-table-div', style={'text-align':'right'})), 
                                html.P(''), 
                                html.H5([
                                    html.B('Validation plots '), 
                                    html.Abbr('\u2139', title=plot_tooltip), 
                                ], className='card-text'), 
                                html.Div(id='validate-metric-graph-div'), 
                                html.Div('', id='validate-data-storage', style={'display': 'none'})  # hidden data storage
                            ], id='validate-loading-storage', type='circle'), 
                            dbc.CardFooter(dbc.Button('Back', id='validate-model-button', color='secondary', href='/sim/run_model', style={'float':'right'})),
                            html.Div(id='validate-model-page-alert'), 
                            ], body=True), 
                        html.Br()], width=12
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)

def get_validation_param(model:str, param:Parameters):
    # get validation parameters
    age_categories = param.age_categories
    case_cols = param.case_cols(model)
    return age_categories, case_cols

def generate_validation_data(model:str, baseline_output:str, model_output:str):
    # get output paths
    baseline_output_filename = baseline_output + '.' + output_filetype
    model_output_filename = model_output + '.' + output_filetype
    baseline_output = os.path.join(path_output, baseline_output_filename)
    model_output = os.path.join(path_output, model_output_filename)
    validate_output_filepath = {   ## dictionary of model validation output files
        'baseline_output': baseline_output, 
        'model_output': model_output
    }
    # log validation input arguments
    logger.info("Validation input arguments: model: %s, baseline_output: %s, model_output: %s", str(model), str(validate_output_filepath['baseline_output']), str(validate_output_filepath['model_output']))
    # get validation data
    df_baseline = pd.read_csv(validate_output_filepath['baseline_output'])
    df_model = pd.read_csv(validate_output_filepath['model_output'])
    return df_baseline, df_model

@dash_app.callback(
    [Output('validate-model-status-text', 'children'), Output('validate-data-storage', 'Children'), 
        Output('validate-age-dropdown','options'), Output('validate-case-dropdown','options')], 
    [Input('validate-baseline-output-dropdown', 'value'), Input('validate-model-output-dropdown', 'value')])
def update_validation_data(baseline_output_value, model_output_value):
    dataset = None
    age_dropdown_options = []
    case_dropdown_options = []
    if (baseline_output_value is None) & (model_output_value is None):
        status_str = 'Select output data'
    elif (baseline_output_value is None):
        status_str = 'Select baseline output data'
    elif (model_output_value is None):
        status_str = 'Select model output data'
    else:
        # check baseline and model output compatibility
        baseline_output_model = baseline_output_value.split('_')[0].strip().upper()
        model_output_model = model_output_value.split('_')[0].strip().upper()
        # update status
        if (baseline_output_model != model_output_model):
            status_str = 'Baseline output does not match with model output'
        else:
            # generate validation metrics
            model = baseline_output_model
            param = Parameters()
            [age_categories, case_cols] = get_validation_param(model, param)
            [df_baseline, df_model] = generate_validation_data(model, baseline_output_value, model_output_value)
            df_model_metrics = model_validation_metrics(population, model, age_categories, case_cols, df_baseline, df_model)
            # serialise output as JSON
            dataset = {
                'df_model_metrics': df_model_metrics.to_json(orient='split'), 
                'df_baseline': df_baseline.to_json(orient='split'), 
                'df_model': df_model.to_json(orient='split'), 
                'age_categories': age_categories, 
                'case_cols': case_cols, 
                'model': model}
            # update metric dropdown options
            if model.upper() in ["CM", "ABM"]:
                age_categories_dropdown = age_categories[:]
                age_categories_dropdown.append('All')
            elif model.upper() == "NM":
                age_categories_dropdown = ['All']
            case_cols_dropdown = case_cols[:]
            case_cols_dropdown.append('All')
            age_dropdown_options = [{'label': k, 'value': k} for k in age_categories_dropdown]
            case_dropdown_options = [{'label': k, 'value': k} for k in case_cols_dropdown]
            status_str = 'Model validation data available for ' + baseline_output_model
    return [status_str, json.dumps(dataset), age_dropdown_options, case_dropdown_options]

@dash_app.callback(
    [Output('validate-metric-table-div', 'children'), Output('validate-metric-graph-div', 'children')], 
    [Input('validate-age-dropdown', 'value'), Input('validate-case-dropdown', 'value'), Input('validate-data-storage', 'Children')], 
    [State('validate-age-dropdown', 'options'), State('validate-case-dropdown', 'options')])
def update_validation_display(age_value, case_value, data_storage_children, age_dropdown_options, case_dropdown_options):
    if (not age_dropdown_options) | (not case_dropdown_options):
        return [None, None]  ## do not return table or plots before initialisations
    else:
        # load json data
        dataset = json.loads(data_storage_children)
        df_model_metrics = pd.read_json(dataset['df_model_metrics'], orient='split')
        df_baseline = pd.read_json(dataset['df_baseline'], orient='split')
        df_model = pd.read_json(dataset['df_model'], orient='split')
        age_categories = dataset['age_categories']
        case_cols = dataset['case_cols']
        model = dataset['model']
        
        # update validation metric table    
        if (age_value is None) & ((case_value is None) | (case_value == 'All')):
            query_age = f"age == '{age_categories[0]}'"
            query_case = ''
        elif (age_value is None) | (age_value == 'All'):
            query_age = ''
        else:
            query_age = f"age == '{age_value}'"

        if (case_value is None) | (case_value == 'All'):
            query_case = ''
        else:
            query_case = f"case == '{case_value}'"

        query = ' & '.join(s for s in [query_age, query_case] if s)
        
        if query.strip():
            db_model_metric_table = df_model_metrics.round(2).query(query)
        else:
            db_model_metric_table = df_model_metrics.round(2)
        metric_table = dbc.Table.from_dataframe(db_model_metric_table, 
                                                id='validation-metric-table', striped=True, bordered=True, hover=True)

        # update validation plots
        if (age_value == 'All'):
            age_cate_plot = age_categories
        elif (age_value is None):
            age_cate_plot = age_categories
        else:
            age_cate_plot = [age_value]

        if (case_value == 'All'):
            case_col_plot = case_cols
        elif (case_value is None):
            case_col_plot = [case_cols[0]]
        else:
            case_col_plot = [case_value]

        graph_divs = model_validation_plots(population, model, age_cate_plot, case_col_plot, df_baseline, df_model)
        return [metric_table, graph_divs]
