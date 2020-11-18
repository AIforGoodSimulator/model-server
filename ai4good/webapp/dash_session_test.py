import dash_bootstrap_components as dbc
import dash_html_components as html
import ai4good.webapp.common_elements as common_elements
import dash_core_components as dcc
from ai4good.webapp.apps import dash_app, facade, model_runner, _redis,flask_app

from flask import Flask,session


text_introduction = 'The Simulator is a web tool for NGOs and local authorities to model COVID-19 outbreak inside refugee camps and prepare timely and proportionate response measures needed to flatten the curve and reduce the number of fatalities. This tool helps to predict the possble outbreak scenarios and their potential outcomes and help NGOs design an optimal intervention strategy. '

layout02 = html.Div([
html.H3('dash_session_test'),
dcc.Dropdown(
id='app-1-dropdown',
options=[
{'label': 'App 1 - {}'.format(i), 'value': i} for i in [
'NYC', 'MTL', 'LA'
]
],
value=session['app-1-display-val-session']
),
html.Div(id='app-1-display-value'),
dcc.Link('Go to App 2', href='/apps/app2')
])

@app.callback(
Output(‘app-1-display-value’, ‘children’),
[Input(‘app-1-dropdown’, ‘value’)])
def display_value(value):
session[‘app-1-display-val-session’] = value
if ‘app-1-display-val-session’ not in session:
return ‘Prova sessione non riuscita’
return ‘You have selected “{}”’.format(session[‘app-1-display-val-session’])