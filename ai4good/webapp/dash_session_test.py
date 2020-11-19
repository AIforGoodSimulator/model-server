import dash_bootstrap_components as dbc
import dash_html_components as html
import ai4good.webapp.common_elements as common_elements
import dash_core_components as dcc
from ai4good.webapp.apps import dash_app, facade, model_runner, _redis,flask_app

from flask import Flask,session


text_introduction = 'The Simulator is a web tool for NGOs and local authorities to model COVID-19 outbreak inside refugee camps and prepare timely and proportionate response measures needed to flatten the curve and reduce the number of fatalities. This tool helps to predict the possble outbreak scenarios and their potential outcomes and help NGOs design an optimal intervention strategy. '

