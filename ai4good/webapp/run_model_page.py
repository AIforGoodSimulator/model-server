import pprint
import dash_core_components as dcc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


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
            [html.P("Model execution succeeded", className="mb-0")],
            id="run_model_toast",
            header="Notification",
            duration=3000,
            icon="primary",
            dismissable=True,
        ),
        dbc.Toast(
            [html.P("No cached results, please run model first", className="mb-0")],
            id="no_results_toast",
            header="Notification",
            duration=3000,
            icon="primary",
            dismissable=True,
        ),
    ])


layout = html.Div(
    [
        html.H3('Run Model'),
        camp_selector(),
        model_selector(),
        profile_selector(),
        model_run_buttons()
        #TODO: add currently running models with progress
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


last_run_model_click = 0


@dash_app.callback(
    Output("run_model_toast", "is_open"),
    [Input("run_model_button", "n_clicks"), Input('camp-dropdown', 'value'), Input('model-dropdown', 'value'),
     Input('profile-dropdown', 'value')]
)
def on_run_model_click(n, camp, model, profile):
    global last_run_model_click
    if n is None:
        return False
    elif n > last_run_model_click:
        last_run_model_click = n
        model_runner.run_model(model, profile, camp)
        return True


@dash_app.callback(
    [
        Output("no_results_toast", "is_open"),
        Output('run_model_button', 'disabled'),
        Output('model_results_button', 'disabled'),
        Output('model_results_button', 'href')
    ],
    [
        Input("model_results_button", "n_clicks"),
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
        else:
            return True, False, True, ''
