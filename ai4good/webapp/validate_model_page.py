import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from ai4good.webapp.apps import dash_app, facade, model_runner
import ai4good.webapp.run_model_page as run_model_page
from ai4good.webapp.model_runner import ModelScheduleRunResult
import ai4good.utils.path_utils as pu
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_table
import os
from datetime import date, timedelta
import sklearn

def get_autocorrelation_plots(pred):
    def get_autocorrelation(sequence, shifts=31):
        correlations = []
        
        for shift in range(1, shifts):
            correlation = np.corrcoef(sequence[:-shift], sequence[shift:])[0, 1]
            correlations.append(correlation)
        return [1] + correlations  # correlation with 0 shift -> 1

    def get_partial_autocorrelation(sequence, shifts=31):
        p_correlations = []

        residuals = sequence
        for shift in range(1, shifts):
            correlation = np.corrcoef(sequence[:-shift], residuals[shift:])[0, 1]
            p_correlations.append(correlation)

            m, c =  np.polyfit(sequence[:-shift], residuals[shift:], 1)  # m -> grad.; c -> intercept
            residuals[shift:] = residuals[shift:] - (m * sequence[:-shift] + c)
        return [1] + p_correlations

    ac_df = pd.DataFrame(data={"shift": np.linspace(0, 30, 31), "ac": get_autocorrelation(pred.to_numpy().copy())})
    pac_df = pd.DataFrame(data={"shift": np.linspace(0, 30, 31), "pac": get_partial_autocorrelation(pred.to_numpy())})

    ac_fig = px.bar(ac_df, x="shift", y="ac", title="autocorrelation")
    pac_fig = px.bar(pac_df, x="shift", y="pac", title="partial autocorrelation")

    return ac_fig, pac_fig

def get_random_pred():  # Get random pred for graph demo
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = pd.DataFrame(columns=["date", "actual", "pred"])
    today = date.today()
    begin = today - timedelta(days=99)
    df["date"] = pd.date_range(begin, today)
    data = np.random.randint(100,150,size=(100,2))
    df[["actual", "pred"]] = pd.DataFrame(data, columns=["actual", "pred"])
    pred = df["pred"]
    
    return pred

pred = get_random_pred()
ac_fig, pac_fig = get_autocorrelation_plots(pred)

layout = html.Div(
    [
        run_model_page.nav_bar(),
        
        html.Div([
            dbc.Container([
                dbc.Row(
                    dbc.Col([
                        dbc.Card([
                            html.H4('Model Validation', className='card-title'),
                            html.P('Descriptions',className='card-text'),
                            html.H5('Section 1', className='card-text'),
                            html.Header('Result 1', className='card-text'),
                            html.Header('Result 2', className='card-text'),
                            html.P(''), 
                            html.H5('Section 2', className='card-text'),
                            html.Header('Result 3', className='card-text'),
                            html.Header('Result 4', className='card-text'),
                            dcc.Graph(figure=ac_fig),
                            dcc.Graph(figure=pac_fig),
                            dbc.CardFooter(dbc.Button('Back', id='validate-model-button', color='secondary', href='/sim/run_model', style={'float':'right'})),
                            html.Div(id='validate-model-page-alert'), 
                            ], body=True), 
                        html.Br()], width=8
                    ), justify='center', style={'margin-top':'50px'}
                )
            ])
        ])
    ]
)
