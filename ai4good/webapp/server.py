import logging
from flask import redirect
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from urllib.parse import urlparse, parse_qs
import ai4good.webapp.landing_page as landing_page
import ai4good.webapp.login_page as login_page
import ai4good.webapp.input_page_1 as input_page_1
import ai4good.webapp.input_page_2 as input_page_2
import ai4good.webapp.input_page_3 as input_page_3
import ai4good.webapp.input_page_4 as input_page_4
import ai4good.webapp.run_model_page as run_model_page
import ai4good.webapp.cm_model_results_page as cm_model_results_page
import ai4good.webapp.cm_model_report_page as cm_model_report_page
import ai4good.webapp.cm_admin_page as cm_admin_page
import ai4good.webapp.abm_model_results_page as abm_model_results_page
import ai4good.webapp.abm_model_report_page as abm_model_report_page
import ai4good.webapp.nm_model_results_page as nm_model_results_page
import ai4good.webapp.nm_model_report_page as nm_model_report_page
import ai4good.webapp.nm_admin_page as nm_admin_page
from ai4good.webapp.apps import flask_app, dash_app


logging.basicConfig(level=logging.DEBUG)


@flask_app.route("/")
def index():
    return redirect('/sim/landing')


dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@dash_app.callback(Output('page-content', 'children'),
                   [Input('url', 'pathname'), Input('url', 'search')])
def display_page(pathname, query=None):
    logging.info("Displaying page %s with query %s", pathname, query)
    if pathname == '/sim/landing' or pathname == '/sim/':
        return landing_page.layout
    elif pathname == '/sim/login_page':
        return login_page.layout
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
    elif pathname == '/sim/results':
        query = parse_qs(urlparse(query).query)
        if query['model'][0] == 'compartmental-model':
            return cm_model_results_page.layout(query['camp'][0], query['profile'][0])
        elif query['model'][0] == 'agent-based-model':
            return abm_model_results_page.layout(query['camp'][0], query['profile'][0])
        elif query['model'][0] == 'network-model':
            return nm_model_results_page.layout(query['camp'][0], query['profile'][0])
        else:
            return '404'
    elif pathname == '/sim/report':
        query = parse_qs(urlparse(query).query)
        if query['model'][0] == 'compartmental-model':
            interventions = query.get('intervention', [])
            return cm_model_report_page.layout(query['camp'][0], query['profile'][0], interventions)
        elif query['model'][0] == 'agent-based-model':
            interventions = query.get('intervention', [])
            return abm_model_report_page.layout(query['camp'][0], query['profile'][0], interventions)
        elif query['model'][0] == 'network-model':
            interventions = query.get('intervention', [])
            return nm_model_report_page.layout(query['camp'][0], query['profile'][0], interventions)
        else:
            return '404'
    elif pathname == '/sim/admin':
        query = parse_qs(urlparse(query).query)
        if query['model'][0] == 'network-model':
            return nm_admin_page.layout()
        return cm_admin_page.layout()
    else:
        return '404'

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dash_app.run_server(debug=True)
