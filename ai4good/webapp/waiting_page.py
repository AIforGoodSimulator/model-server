import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelsRunningNow, ModelScheduleRunResult, _sid, ModelRunner
from dash.dependencies import Input, Output, State
import time
from ai4good.webapp.run_model_for_dashboard import run_model_results_for_messages
from ai4good.webapp.apps import dash_app, facade, _redis, dask_client
from ai4good.utils.logger_util import get_logger

logger = get_logger(__name__)
initial_status = "Simulation Running ..."
initial_time = time.localtime()
initial_time = time.strftime("%m/%d/%Y, %H:%M:%S", initial_time)


layout = html.Div([
        dcc.Interval(id='interval2', interval=1000, n_intervals=0, max_intervals=1),
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
        ]),
        dcc.Store(id='memory'),
    ]
)


@dash_app.callback([Output('memory', 'data')],[Input('interval2', 'n_intervals')])
def run_model_results(n):
    running_log = ModelsRunningNow(_redis)
    running_log.clear_run()
    model_runner = ModelRunner(facade, _redis, dask_client, _sid)
    queue = run_model_results_for_messages(model_runner,["message_1", "message_5"])
    return [queue]


@dash_app.callback([Output('update_String', 'children'),Output('status_String', 'children')],
    [Input('memory', 'data'),Input('interval1', 'n_intervals')])
def empty_queue(queue,n):
    model_runner = ModelRunner(facade, _redis, dask_client, _sid)
    if len(queue) > 0:
        queue_string = queue.pop(0)
        splited_string = queue_string.split('|')
        model = splited_string[0]
        profile = splited_string[1]
        res = model_runner.run_model(model, profile)
        if res == ModelScheduleRunResult.SCHEDULED:
            logger.info("Model run scheduled")
        elif res == ModelScheduleRunResult.CAPACITY:
            # put the pair back to the queue
            queue.append(queue_string)
            logger.info("Can not run model now, added to queue")
        elif res == ModelScheduleRunResult.ALREADY_RUNNING:
            logger.info("Already running")
        else:
            raise RuntimeError("Unsupported result type: " + str(res))
        return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), ("Status: ", initial_status)
    else:
        ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), ("Status: Finished")

# Check every 10 seconds to check if there is a report ready
# Or make the previous run model blocking and update the message once the operations there are done
# @dash_app.callback([Output('update_String', 'children'),Output('status_String', 'children')],
#     [Input('interval1', 'n_intervals')])
# def check_Model(n):
#
#     if (model_runner.results_exist(model, profile)):
#         return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), ("Status: Finished")
#     else:
#         print("No results yet " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())))
#         return ("Last Updated: " + str(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))), ("Status: ", initial_status)
