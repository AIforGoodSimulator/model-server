import pprint
import dash
import dash_core_components as dcc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
from ai4good.webapp.model_runner import ModelScheduleRunResult
from dash.dependencies import Input, Output, State
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
        dbc.Col(html.Div([dbc.Card(id='profile-info', body=True)], style={'height': '100%'}),
        width=6),
    ], style={'margin': 10})


def model_run_buttons():
    return html.Div([
        dbc.Button("Run Model", id="run_model_button", color="primary", className="mr-1", disabled=True),
        dbc.Button("See Results", id="model_results_button", color="success", className="mr-1",
                   target="_blank", disabled=True, external_link=True, href='none', key='model_results_button_key'),
        dbc.Toast(
            [],
            id="run_model_toast",
            header="Notification",
            duration=3000,
            icon="primary",
            dismissable=True,
            is_open=False
        ),
        html.Div([], id='model_run_tooltip_holder')
    ], id='run_buttons_div')


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
            interval=5 * 1000  # in milliseconds
        )
    ], style={'margin': 10}
)


@dash_app.callback(
    Output('camp-info', 'children'),
    [Input('camp-dropdown', 'value')])
def update_camp_info(value):
    if value is not None:
        total_population = int(facade.ps.get_camp_params(value).Total_population.dropna().sum())
        return f'Camp "{value}" with total population {total_population} people'
    else:
        return 'Select camp'


@dash_app.callback(
    [Output('model-info', 'children'), Output('profile-dropdown', 'options')],
    [Input('model-dropdown', 'value')])
def update_model_info(value):
    if value is not None:
        return value, [{'label': i, 'value': i} for i in facade.ps.get_profiles(value)]
    else:
        return 'Select model', []


@dash_app.callback(
    Output('profile-info', 'children'),
    [Input('model-dropdown', 'value'), Input('profile-dropdown', 'value')])
def update_profile_info(model, profile):
    if model is not None and profile is not None:
        df = facade.ps.get_params(model, profile).drop(columns=['Profile'])
        return dash_table.DataTable(
            id='profile_table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
        )
    else:
        return 'Select profile'


@dash_app.callback(
    [Output("run_model_toast", "is_open"), Output("run_model_toast", "children")],
    [Input("run_model_button", "n_clicks")],
    [State('camp-dropdown', 'value'), State('model-dropdown', 'value'), State('profile-dropdown', 'value')]
)
def on_run_model_click(n, camp, model, profile):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, dash.no_update
    else:
        res = model_runner.run_model(model, profile, camp)
        if res == ModelScheduleRunResult.SCHEDULED:
            return True, html.P("Model run scheduled", className="mb-0")
        elif res == ModelScheduleRunResult.CAPACITY:
            return True, html.P("Can not run model now, over capacity, try again later", className="mb-0")
        elif res == ModelScheduleRunResult.ALREADY_RUNNING:
            return True, html.P("Already running", className="mb-0")
        else:
            raise RuntimeError("Unsupported result type: "+str(res))


@dash_app.callback(Output('history_table', 'data'),
                   [Input('interval-component', 'n_intervals')])
def update_history(n):
    df = model_runner.history_df()
    return df.to_dict('records')


@dash_app.callback(
    [
        Output('run_model_button', 'disabled'),
        Output('model_results_button', 'disabled'),
        Output('model_results_button', 'href'),
        Output("model_run_tooltip_holder", "children")
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
        return True, True, '', \
               dbc.Tooltip('Select camp, model and profile to see results', id='_mr_tt', target="run_buttons_div")
    else:
        if model_runner.results_exist(model, profile, camp):
            return False, False, f'/sim/results?model={model}&profile={profile}&camp={camp}', []
        else:
            return False, True, '', \
                   dbc.Tooltip('No cached results, please run model first', id='_mr_tt', target='run_buttons_div'),
