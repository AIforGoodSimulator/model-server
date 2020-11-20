import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
from ai4good.webapp.apps import model_runner
from ai4good.webapp.model_runner import ModelScheduleRunResult
import ai4good.utils.path_utils as pu
from dash.dependencies import Input, Output, State
import ai4good.webapp.common_elements as common_elements
import dash_table
import os
from datetime import datetime
import time

# some default stuff
base = '../../fs'
initial_status = "Simulation Running ..."
finished_status = "Simulation Completed"
time_format_raw = "%Y-%m-%d %H:%M:%S.%f"
time_format = "%Y-%m-%d, %H:%M:%S"
initial_time = time.localtime()
initial_time = time.strftime(time_format, initial_time)

# model Parameter
model = 'compartmental-model'
profile = 'baseline'
camp = 'Moria'

# def run_model_results_for_message_1():
#     for prof in profile:
#         res = model_runner.run_model(model, profile, camp)
#         if res == ModelScheduleRunResult.SCHEDULED:
#             print("Model run scheduled")
#         elif res == ModelScheduleRunResult.CAPACITY:
#             print("Can not run model now, over capacity, try again later")
#         elif res == ModelScheduleRunResult.ALREADY_RUNNING:
#             print("Already running")
#         else:
#             raise RuntimeError("Unsupported result type: "+str(res))
#     return None

# # Runs the model so that there is something to check for in cache
# run_model_results_for_message_1()



layout = html.Div([
        dcc.Interval(id='waiting-interval', interval=10 * 1000, n_intervals=0),
        common_elements.nav_bar(),
        html.Br(),
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Center(html.Img(src='/static/input_step5.png', title='Step completed', style={'width':'50%'}, className="step_counter")), 
                            html.Header('Please wait for the simulation to complete.', className='card-text'),
                            html.Div([
                                html.P('', id="waiting-status", style={'margin':'0px'}),
                                html.P('', id="waiting-start-time", style={'margin':'0px'}),
                                html.P('', id="waiting-end-time", style={'margin':'0px'}),
                                html.P(("Last Updated: ", initial_time), id="waiting-update", className="status_Time", style={'font-size':'0.8em'}),
                            ], className="results_Controls"), 
                            dbc.CardFooter(dbc.Button('View Report', id='model_results_button', className="mr-1 holo", style={'float':'right'})),
                            html.Div(id='waiting-page-alert')
                            ], body=True), 
                         html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)


# Check every 10 seconds to check if there is a report ready
@dash_app.callback(
    [Output('waiting-update', 'children'), Output('waiting-status', 'children'), 
     Output('waiting-start-time', 'children'), Output('waiting-end-time', 'children')], 
    [Input('waiting-interval', 'n_intervals')])
def check_model_status(n):
    mr = model_runner.get_result(model, profile, camp)
    assert mr is not None
    history_df = model_runner.history_df()
    history_mr_start = history_df[
        (history_df.Key.isin([str(f"('{model}', '{profile}', '{camp}')")])) &
        (history_df.Status.isin(['ModelRunResult.RUNNING']))]
    start_time_raw = datetime.strptime(str(history_mr_start.Time.iloc[0]), time_format_raw)
    start_time = datetime.strftime(start_time_raw, time_format)
    last_updated_time = time.strftime(time_format, time.localtime())
    if (model_runner.results_exist(model, profile, camp)):
        history_mr_end = history_df[
            (history_df.Key.isin([str(f"('{model}', '{profile}', '{camp}')")])) &
            (history_df.Status.isin(['ModelRunResult.SUCCESS']))]
        end_time_raw = datetime.strptime(str(history_mr_end.Time.iloc[0]), time_format_raw)
        end_time = datetime.strftime(end_time_raw, time_format)
        return ("Last Updated: " + str(last_updated_time)), ("Status: ", finished_status), ("Start time: " + str(start_time)), ("End time: " + str(end_time))
    else:
        return ("Last Updated: " + str(last_updated_time)), ("Status: ", initial_status), ("Start time: " + str(start_time)), ("End time: ", '')
