import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.model_runner import ModelsRunningNow, ModelScheduleRunResult, _sid, ModelRunner
from dash.dependencies import Input, Output, State
import time
import ai4good.webapp.common_elements as common_elements
from ai4good.webapp.run_model_for_dashboard import run_model_results_for_messages, check_model_results_for_messages, collate_model_results_for_user, check_model_results_for_messages_unrun
from ai4good.webapp.apps import dash_app, facade, _redis, dask_client, model_runner
from ai4good.utils.logger_util import get_logger
import json

logger = get_logger(__name__)
initial_status = "Simulation Running ..."
initial_time = time.localtime()
initial_time = time.strftime("%m/%d/%Y, %H:%M:%S", initial_time)


layout = html.Div([
        # dcc.Interval(id='starting', interval=1000, n_intervals=0, max_intervals=1),
        dcc.Interval(id='interval1', interval=10 * 1000, n_intervals=0),
        common_elements.nav_bar(),
        html.Br(),
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('COVID-19 Simulator', className='card-title'),
                            html.Header('Please wait for the simulation to complete.', className='card-text'),
                            html.Div([
                                html.P(("Status: ", initial_status), id="status_String",style={'margin':'0px'}),
                                html.P(("Last Updated: ", initial_time), id="update_String", className="status_Time",
                                       style={'font-size':'0.8em'}),
                                dbc.Button("Rerun the models", id="model_rerun_button", className="mr-1 holo",
                                           disabled=True),
                                dbc.Button("View Results", id="model_dashboard_button_ui", className="mr-1 holo", disabled=True),
                            ], className="results_Controls")
                            ], body=True
                        ),html.Br()], width=6
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ]),
        dcc.Store(id='memory'),
    ]
)


# we need to have a way to prevent a user submitting the runs multiple times
# (some batch model running now might help)
@dash_app.callback([Output('memory', 'data')],
                   [Input("model_rerun_button", "n_clicks")])
def re_run_model_results(run_n):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ["placeholder"]
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'model_rerun_button':
            rerun_config = check_model_results_for_messages_unrun(model_runner,["message_1", "message_5"])
            model_runner.batch_run_model(rerun_config)
            return ["placeholder"]
        else:
            return ["placeholder"]
    # output a queue to resubmit later if that happens but with the increased capacity
    # on the model runner this might never happen
    # check if they are all running
    # for result in res:
    #     if res == ModelScheduleRunResult.CAPACITY:
    #         message = "some models are not running due to capacity reason"
    #         return [message]
    # message = "all models are running now"
    # return [message]


# Check every 10 seconds to check if there is a report ready
@dash_app.callback([Output('update_String', 'children'), Output('status_String', 'children'),
                    Output("model_dashboard_button_ui", "disabled"), Output("model_dashboard_button_ui", "href"),
                    Output("model_rerun_button", "disabled"), ],
    [Input('interval1', 'n_intervals')])
def check_model(n):
    results_ready = check_model_results_for_messages(model_runner, ["message_1", "message_5"])
    if results_ready:
        user_input = json.loads(model_runner.user_input)
        camp = str(user_input['name-camp'])
        total_population = int(user_input["total-population"])
        collate_model_results_for_user(model_runner, ["message_1", "message_5"], camp, total_population)
        return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), \
               "Status: Finished", False, f'/sim/dashboard?camp={camp}', True
    else:
        print("No results yet " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())))
        return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), \
               ("Status: ", initial_status), True, "", False

