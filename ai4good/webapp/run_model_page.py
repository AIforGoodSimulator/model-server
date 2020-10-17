import numpy as np
import pandas as pd
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
        dbc.Col(html.Div([dbc.Card(id='profile-info', body=True, children=html.Div([
            html.Div(id='profile_help'),
            dash_table.DataTable(
                id='profile_table',
                columns=[],
                data=[]
            ),
            dbc.Button("Save", id="save_profile_button",outline=True, color="info", className="mr-1",
                       style={'display': 'none'})
        ]))], style={'height': '100%'}), width=6),

        dbc.Modal([
            dbc.ModalHeader("Save profile"),
            dbc.ModalBody(dbc.Input(id="profile_name_input", placeholder="Specify profile name...", type="text",
                                    autoFocus=True)),
            dbc.ModalFooter([
                dbc.Button(
                    "OK", id="do_save_profile_dialog", className="ml-auto"
                ),
                dbc.Button(
                    "Close", id="close_save_profile_dialog", className="ml-auto"
                )
            ]),
        ], id="save_profile_dialog", centered=True)

    ], style={'margin': 10})


def model_run_buttons():
    return html.Div([
        dbc.Button("Run Model", id="run_model_button", color="primary", className="mr-1", disabled=True),
        dbc.Button("See Results", id="model_results_button", color="success", className="mr-1",
                   target="_blank", disabled=True, external_link=True, href='none', key='model_results_button_key'),
        dbc.Button("See Report", id="model_report_button", color="success", className="mr-1",
                   disabled=True, key='model_report_button_key'),
        dbc.Toast(
            [],
            id="run_model_toast",
            header="Notification",
            duration=3000,
            icon="primary",
            dismissable=True,
            is_open=False
        ),
        html.Div([], id='model_run_tooltip_holder'),
        dbc.Modal([
            dbc.ModalHeader(id='show_report_header'),
            dbc.ModalBody([
                html.Div("Select intervention profiles to compare:"),
                dbc.Checklist(options=[], value=[], id="show_report_intervention_checklist")
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "OK", id="do_show_report", className="ml-auto", target="_blank", external_link=True, href='none'
                ),
                dbc.Button(
                    "Cancel", id="cancel_show_report", className="ml-auto"
                )
            ]),
        ], id="show_report_dialog", centered=True)

    ], id='run_buttons_div')


def history_table():
    cols = model_runner.history_columns()

    return html.Div([
        dbc.Row([
         html.H3('Run Queue ')
        ], style={'margin-top': 100, 'margin-left': 15,'color':'Black','border':'10px'}),
        dbc.Row([
        dbc.Col(
            html.Div([
                dash_table.DataTable(
                    id='history_table',
                    columns=[{"name": i, "id": i} for i in cols],
                    data=[{}],
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)',

                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(89,169,255)',
                        'fontWeight': 'bold',
                        'color': 'rgb(255,255,255)',
                        'text-align': 'center'
                    }

                )
            ],style={'margin-left':20}),
            width=6,
        )
    ])
    ])


def nav_bar():
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("About US", href="#")),

        ],
        brand="AIforGood",
        brand_href="#",
        color="primary",
        dark=True,
    )


layout = html.Div(
    [
        nav_bar(),
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
    [Output('profile_help', 'children'),
     Output('profile_table', 'columns'), Output('profile_table', 'data'),
     Output('save_profile_button', 'style')],
    [Input('model-dropdown', 'value'), Input('profile-dropdown', 'value')])
def update_profile_info(model, profile):
    if model is not None and profile is not None:
        df = facade.ps.get_params(model, profile).drop(columns=['Profile'])
        return '', [{"name": i, "id": i, 'editable': i != 'Parameter'} for i in df.columns], \
               df.to_dict('records'), {'float': 'right', 'margin-top': 12}
    else:
        return ['Select profile', [], [], {'display': 'none'}]


@dash_app.callback(
    Output('save_profile_button', 'disabled'),
    [Input('model-dropdown', 'value'), Input('profile-dropdown', 'value'),
     Input('profile_table', 'data'), Input('profile_table', 'columns')])
def update_save_button_state(model, profile, profile_table_data, profile_table_columns):
    if model is not None and profile is not None:
        original_df = facade.ps.get_params(model, profile).drop(columns=['Profile'])
        new_df = pd.DataFrame(profile_table_data, columns=[c['name'] for c in profile_table_columns])
        return np.array_equal(original_df.values,new_df.values)
    else:
        return True


@dash_app.callback(
    [Output("save_profile_dialog", "is_open"), Output('profile-dropdown', 'value'), Output('model-dropdown', 'value'),
     Output('profile_name_input', 'value')],
    [Input("save_profile_button", "n_clicks"), Input("close_save_profile_dialog", "n_clicks"), Input("do_save_profile_dialog", "n_clicks")],
    [State("save_profile_dialog", "is_open"), State('profile_name_input', 'value'),
     State('model-dropdown', 'value'), State('profile-dropdown', 'value'),
     State('profile_table', 'data'), State('profile_table', 'columns')]
)
def on_save_profile_button_click(n_save, n_close, n_confirm_save, is_open, new_profile_name, model, profile,
                                 profile_table_data, profile_table_columns):
    ctx = dash.callback_context
    if ctx.triggered and (n_save or n_close or n_confirm_save):
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'save_profile_button':
            return True, dash.no_update, dash.no_update, profile
        elif button_id == 'close_save_profile_dialog':
            return False, profile, dash.no_update, dash.no_update
        elif button_id == 'do_save_profile_dialog':
            assert new_profile_name is not None and len(new_profile_name) > 0
            new_profile_df = pd.DataFrame(profile_table_data, columns=[c['name'] for c in profile_table_columns])
            facade.ps.store_params(model, new_profile_name, new_profile_df)
            return False, new_profile_name, model, dash.no_update
    else:
        return is_open, dash.no_update, dash.no_update, dash.no_update


@dash_app.callback(
    [Output("run_model_toast", "is_open"), Output("run_model_toast", "children")],
    [Input("run_model_button", "n_clicks")],
    [State('camp-dropdown', 'value'), State('model-dropdown', 'value'),
    State('profile-dropdown', 'value')]
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
        Output('model_report_button', 'disabled'),
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
        return True, \
               True, '', \
               True, \
               dbc.Tooltip('Select camp, model and profile to see results', id='_mr_tt', target="run_buttons_div")
    else:
        if model_runner.results_exist(model, profile, camp):
            return False, \
                   False, f'/sim/results?model={model}&profile={profile}&camp={camp}', \
                   False, \
                   []
        else:
            return False, \
                   True, '', \
                   True, \
                   dbc.Tooltip('No cached results, please run model first', id='_mr_tt', target='run_buttons_div'),


@dash_app.callback(
    [Output("show_report_dialog", "is_open"),
     Output('show_report_header', 'children'),
     Output('show_report_intervention_checklist', 'options'),
     Output('show_report_intervention_checklist', 'value')
    ],
    [Input("model_report_button", "n_clicks"), Input("do_show_report", "n_clicks"), Input("cancel_show_report", "n_clicks")],
    [State("show_report_dialog", "is_open"), State('model-dropdown', 'value'),
     State('profile-dropdown', 'value'), State('camp-dropdown', 'value')]
)
def on_run_report_button_click(n_open, n_show, n_cancel, is_open, model, profile, camp):

    def is_enabled(p):
        return (p != profile) and model_runner.results_exist(model, p, camp)

    ctx = dash.callback_context
    if ctx.triggered and (n_open or n_show or n_cancel):
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'do_show_report':
            return False, [], [], []
        elif button_id == 'cancel_show_report':
            return False, [], [], []
        elif button_id == 'model_report_button':
            profiles = facade.ps.get_profiles(model)
            selected_profiles = [p for p in profiles if is_enabled(p)]
            profile_options = [{'label': p, 'value': p} for p in selected_profiles]

            if len(selected_profiles) == 0:
                profile_options = [{'label': 'No model runs available', 'value': 'None', 'disabled': True}]
            return True, f'Report for {profile} profile', profile_options, selected_profiles
    else:
        return is_open, dash.no_update, dash.no_update, dash.no_update


@dash_app.callback(
    Output("do_show_report", "href"),
    [Input("show_report_intervention_checklist", "value")],
    [State('model-dropdown', 'value'), State('profile-dropdown', 'value'), State('camp-dropdown', 'value')]
)
def update_run_report_link(interventions, model, profile, camp):
    p = "&".join(map(lambda i: f'intervention={i}', interventions))
    return f'/sim/report?model={model}&profile={profile}&camp={camp}&{p}',

