import pprint
import dash
import dash_core_components as dcc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
from ai4good.webapp.model_runner import ModelScheduleRunResult
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_table


def camp_selector():
    return dbc.Row([
        dbc.Col(
            html.Div([
                html.Label('Camp', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='camp-dropdown',
                    options=[{'label': c, 'value': c} for c in facade.ps.get_camps()]
                ),
            ]),
            width=3,
        ),
        dbc.Col(html.Div([dbc.Card('', id='camp-info', body=True)], style={'height': '100%'}),
            width=6,
        ),
    ], style={'margin': 10})


def model_selector():
    return dbc.Row([
        dbc.Col(
            html.Div([
                html.Label('Model', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': m, 'value': m} for m in facade.ps.get_models()]
                ),
            ]),
            width=3,
        ),
        dbc.Col(html.Div([dbc.Card('', id='model-info', body=True)], style={'height': '100%'}),
            width=6,
        ),
    ], style={'margin': 10})


def profile_selector():
    return dbc.Row([
        dbc.Col(
            html.Div([
                html.Label('Profile', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='profile-dropdown'
                ),
            ]),
            width=3,
        ),
        dbc.Col(html.Div([dbc.Card('', id='profile-info', body=True)], style={'height': '100%'}),
            width=6,
        ),
    ], style={'margin': 10})


def model_run_buttons():
    return html.Div([
        dbc.Button("Run Model", id="run_model_button", color="primary", className="mr-1", disabled=True),
        dbc.Button("See Results", id="model_results_button", color="success", className="mr-1",
                   target="_blank", disabled=True),
        dbc.Toast(
            [],
            id="run_model_toast",
            header="Notification",
            duration=3000,
            icon="primary",
            dismissable=True,
            is_open=False
        ),
        dbc.Toast(
            [html.P("No cached results, please run model first", className="mb-0")],
            id="no_results_toast",
            header="Notification",
            duration=3000,
            icon="primary",
            dismissable=True,
            is_open=False
        ),
    ])


def history_table():
    cols = model_runner.history_columns()

    return dbc.Row([
        dbc.Col(
            html.Div([
                html.H3('Run Queue'),
                dash_table.DataTable(
                    id='history_table',
                    columns=[{"name": i, "id": i} for i in cols],
                    data=[{}],
                )
            ]),
            width=6,
        )
    ], style={'margin-top': 100, 'margin-left': 5})


layout = html.Div(
    [
        html.H3('Run Model'),
        camp_selector(),
        model_selector(),
        profile_selector(),
        model_run_buttons(),
        history_table(),
        dcc.Interval(
            id='interval-component',
            interval=2 * 1000  # in milliseconds
        )
    ], style={'margin': 10}
)


@dash_app.callback(
    Output('camp-info', 'children'),
    [Input('camp-dropdown', 'value')])
def display_value(value):
    if value is not None:
        total_population = int(facade.ps.get_camp_params(value).Total_population.dropna().sum())
        return f'Camp "{value}" with total population {total_population} people'
    else:
        return 'Select camp'


@dash_app.callback(
    [Output('model-info', 'children'), Output('profile-dropdown', 'options')],
    [Input('model-dropdown', 'value')])
def display_value(value):
    if value is not None:
        return value, [{'label': i, 'value': i} for i in facade.ps.get_profiles(value)]
    else:
        return 'Select model', []


@dash_app.callback(
    Output('profile-info', 'children'),
    [Input('model-dropdown', 'value'), Input('profile-dropdown', 'value')])
def display_value(model, profile):
    if model is not None and profile is not None:
        params = facade.ps.get_params(model, profile)
        return pprint.pformat(params)
    else:
        return 'Select model and profile'


@dash_app.callback(
    [Output("run_model_toast", "is_open"), Output("run_model_toast", "children")],
    [Input("run_model_button", "n_clicks"), Input('camp-dropdown', 'value'), Input('model-dropdown', 'value'),
     Input('profile-dropdown', 'value')]
)
def on_run_model_click(n, camp, model, profile):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, dash.no_update
    else:
        event_source_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if event_source_id == 'run_model_button':
            res = model_runner.run_model(model, profile, camp)
            if res == ModelScheduleRunResult.SCHEDULED:
                return True, html.P("Model run scheduled", className="mb-0")
            elif res == ModelScheduleRunResult.CAPACITY:
                return True, html.P("Can not run model now, over capacity, try again later", className="mb-0")
            elif res == ModelScheduleRunResult.ALREADY_RUNNING:
                return True, html.P("Already running", className="mb-0")
            else:
                raise RuntimeError("Unsupported result type: "+str(res))
        else:
            return False, dash.no_update


@dash_app.callback(Output('history_table', 'data'),
              [Input('interval-component', 'n_intervals')])
def update_history(n):
    df = model_runner.history_df()
    return df.to_dict('records')


@dash_app.callback(
    [
        Output("no_results_toast", "is_open"),
        Output('run_model_button', 'disabled'),
        Output('model_results_button', 'disabled'),
        Output('model_results_button', 'href')
    ],
    [
        Input('interval-component', 'n_intervals'),
        Input('camp-dropdown', 'value'),
        Input('model-dropdown', 'value'),
        Input('profile-dropdown', 'value')
    ]
)
def on_see_results_click_and_state_update(n, camp, model, profile):
    if camp is None or model is None or profile is None:
        return False, True, True, ''
    else:
        if model_runner.results_exist(model, profile, camp):
            return False, False, False, f'/sim/results?model={model}&profile={profile}&camp={camp}'
        #TODO: elif href == '':
        #    return False, False, True, ''
        else:
            return True, False, True, ''
