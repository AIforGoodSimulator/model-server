from urllib.parse import urlparse, parse_qs
from flask import redirect
from flask_login import current_user
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from ai4good.utils.logger_util import get_logger
import ai4good.webapp.login_page as login_page
import ai4good.webapp.logout_page as logout_page
import ai4good.webapp.register_page as register_page
import ai4good.webapp.landing_page as landing_page
import ai4good.webapp.input_page_1 as input_page_1
import ai4good.webapp.input_page_2 as input_page_2
import ai4good.webapp.input_page_3 as input_page_3
import ai4good.webapp.input_page_4 as input_page_4
import ai4good.webapp.waiting_page as waiting_page
import ai4good.webapp.run_model_page as run_model_page
import ai4good.webapp.validate_model_page as validate_model_page
import ai4good.webapp.old_pages.cm_model_results_page as cm_model_results_page
import ai4good.webapp.cm_model_report_page as cm_model_report_page
import ai4good.webapp.cm_admin_page as cm_admin_page
import ai4good.webapp.abm_model_results_page as abm_model_results_page
import ai4good.webapp.abm_model_report_page as abm_model_report_page
import ai4good.webapp.nm_model_results_page as nm_model_results_page
import ai4good.webapp.nm_model_report_page as nm_model_report_page
import ai4good.webapp.nm_admin_page as nm_admin_page
import ai4good.webapp.model_results_scaffolding as model_results_scaffolding
from ai4good.webapp.apps import flask_app, dash_app, dash_auth_app

logger = get_logger(__file__, 'DEBUG')


@flask_app.route('/')
def index():
    return redirect('/auth/')


@flask_app.route('/register/')
def register():
    return redirect('/auth/register/')


dash_auth_app.layout = html.Div([
    dcc.Location(id='url-auth-app', refresh=False),
    html.Div(id='page-content-auth-app')
])


dash_app.layout = html.Div([
    dcc.Location(id='url-app', refresh=False),
    html.Div(id='page-content-app')
])


@dash_auth_app.callback(Output('page-content-auth-app', 'children'),
                   [Input('url-auth-app', 'pathname'), Input('url-auth-app', 'search')])
def display_dash_auth_app_page(pathname, query=None):
    logger.info("Displaying page %s with query %s", pathname, query)
    if pathname in ['/auth/login_page', '/auth/login/', '/auth/']:
        return login_page.layout
    elif pathname in ['/auth/register_page', '/auth/register/']:
        return register_page.layout
    else:
        return '404 auth'


@dash_app.callback([Output('navlink-user', 'children'), Output('navlink-tooltip-user', 'children')], 
                   [Input('url-app', 'pathname'), Input('url-app', 'search')])
def update_dash_app_navbar(pathname, query=None):
    assert current_user
    return [current_user.username.split('@')[0], 'Welcome ' + current_user.username]


@dash_app.callback(Output('page-content-app', 'children'),
                   [Input('url-app', 'pathname'), Input('url-app', 'search')])
def display_dash_app_page(pathname, query=None):
    logger.info("Displaying page %s with query %s", pathname, query)
    if pathname in ['/auth/logout_page', '/auth/logout/', '/logout/']:
        return logout_page.layout
    elif pathname == '/sim/landing' or pathname == '/sim/':
        return landing_page.layout
    elif pathname == '/sim/input_page_1':
        return input_page_1.layout
    elif pathname == '/sim/input_page_2':
        return input_page_2.layout
    elif pathname == '/sim/input_page_3':
        return input_page_3.layout
    elif pathname == '/sim/input_page_4':
        return input_page_4.layout
    elif pathname == '/sim/run_model':
        return run_model_page.layout
    elif pathname == '/sim/validate_model':
        return validate_model_page.layout
    elif pathname == '/sim/results':
        query = parse_qs(urlparse(query).query)
        if query['model'][0] in ['compartmental-model', 'compartmental-model-stochastic']:
            return cm_model_results_page.layout(query['model'][0], query['camp'][0], query['profile'][0])
        elif query['model'][0] == 'agent-based-model':
            return abm_model_results_page.layout(query['camp'][0], query['profile'][0])
        elif query['model'][0] == 'network-model':
            return nm_model_results_page.layout(query['camp'][0], query['profile'][0])
        else:
            return '404'
    elif pathname == '/sim/report':
        query = parse_qs(urlparse(query).query)
        if query['model'][0] in ['compartmental-model', 'compartmental-model-stochastic']:
            interventions = query.get('intervention', [])
            return cm_model_report_page.layout(query['model'][0], query['camp'][0], query['profile'][0], interventions)
        elif query['model'][0] == 'agent-based-model':
            interventions = query.get('intervention', [])
            return abm_model_report_page.layout(query['camp'][0], query['profile'][0], interventions)
        elif query['model'][0] == 'network-model':
            interventions = query.get('intervention', [])
            return nm_model_report_page.layout(query['camp'][0], query['profile'][0], interventions)
        else:
            return '404'
    elif pathname == '/sim/dashboard':
        query = parse_qs(urlparse(query).query)
        return model_results_scaffolding.layout(query['camp'][0])
    elif pathname == '/sim/admin':
        return cm_admin_page.layout()
    elif pathname == '/sim/admin_nm':
        return nm_admin_page.layout()
    elif pathname == '/sim/results_test':
        return model_results_scaffolding.layout()
    elif pathname == '/sim/waiting':
        return waiting_page.layout
    else:
        return '404'

if __name__ == '__main__':
    dash_app.run_server(debug=True)
    dash_auth_app.run_server(debug=True)
