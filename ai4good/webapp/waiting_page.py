import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
from ai4good.webapp.apps import model_runner
from ai4good.webapp.apps import dash_app, facade, model_runner, _redis
from ai4good.webapp.model_runner import InputParameterCache, ModelsRunningNow
from ai4good.webapp.model_runner import ModelScheduleRunResult
import ai4good.utils.path_utils as pu
from dash.dependencies import Input, Output, State
import dash_table
import os
import time
import ai4good.webapp.run_model_for_dashboard as run_model_for_dashboard
# # some default stuff
# base = '../../fs'
initial_status = "Simulation Running ..."
initial_time = time.localtime()
initial_time = time.strftime("%m/%d/%Y, %H:%M:%S", initial_time)
#
# # model Parameter
# model = 'compartmental-model'
# profile = 'baseline'
# camp = 'Moria'


layout = html.Div([
        dcc.Interval(id='interval1', interval=10 * 1000, n_intervals=0),
        run_model_page.nav_bar(),
        html.Br(),
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Header('Please wait for the simulation to complete.', className='card-text'),
                            html.Div([
                                html.P(("Status: ", initial_status),id="status_String",style={'margin':'0px'}),
                                html.P(("Last Updated: ", initial_time),id="update_String", className="status_Time", style={'font-size':'0.8em'}),
                                dbc.Button("View Report", id="model_results_button", className="mr-1 holo"),
                            ], className="results_Controls")
                            ], body=True
                        ),html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)


# Check every 10 seconds to check if there is a report ready
# Or make the previous run model blocking and update the message once the operations there are done
@dash_app.callback([Output('update_String', 'children'),Output('status_String', 'children')],
    [Input('interval1', 'n_intervals')])
def check_Model(n):
    if (model_runner.results_exist(model, profile)):
        return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), ("Status: Finished")
    else:
        print("No results yet " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())))
        return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), ("Status: ", initial_status)
