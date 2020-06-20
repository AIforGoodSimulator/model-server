import numpy as np
import pandas as pd
import logging
import dash_core_components as dcc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_html_components as html
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.models.cm.initialise_parameters import Parameters
from ai4good.webapp.apps import dash_app, facade, model_runner, cache, cache_timeout
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import textwrap


def layout(camp, profile):

    _, profile_df, params, _ = get_model_result(camp, profile)

    return html.Div(
        [
            dcc.Markdown(disclaimer(camp), style={'margin': 30}),
            html.H1(f'AI for Good Simulator Model Report for {camp} Camp {profile} profile', style={'margin': 30}),
            dcc.Markdown(glossary(), style={'margin': 30}),
            dcc.Markdown(overview1(camp, params), style={'margin': 30}),
            html.Img(src='/static/cm_model.png'),
            html.Div(overview_population(params), style={'margin': 30}),
            dcc.Loading(html.Div([], id='main_section_part1', style={'margin': 30})),
            dcc.Loading(html.Div([], id='main_section_part2', style={'margin': 30})),
            dcc.Loading(html.Div([], id='main_section_part3', style={'margin': 30})),
            html.Div(camp, id='_camp_param', style={'display': 'none'}),
            html.Div(profile, id='_profile_param', style={'display': 'none'})
        ], style={'margin': 50}
    )


def disclaimer(camp):
    return textwrap.dedent(f'''
    ##### Disclaimer: This report is a draft report from AI for Good Simulator on the COVID-19 situation
    in {camp} camp. The insights are preliminary and they are subject to future model
    fixes and improvements on parameter values
    ''').replace('\n', ' ')


def glossary():
    return textwrap.dedent(f'''
    ## 0. Glossary 
    * **hospitalisation/critical care demand person days**: The residents go into the hospitalisation/critical stage signify they require care but they may not receive the care depending on the capacity so this aggregates the hospitalisation/critical demand by person times number of days. If a unit cost of care is known, the total medical expense can be calculated.
    * **IQR (interquartile range)**: The output of modelling are represented as Interquartile range representing a confidence interval of 25%-75%.
    ''')


def overview1(camp: str, params: Parameters):
    return textwrap.dedent(f'''
    ## 1. Overview
    This report provides simulation-based estimates for COVID-19 epidemic scenarios for the {camp} camp. 
    There are an estimated {int(params.population)} people currently living in the camp. Through epidemiology simulations, 
    we estimated peak counts, the timing of peak counts as well as cumulative counts for new symptomatic cases, hospitalisation demand person-days, 
    critical care demand person-days and deaths for an unmitigated epidemic.  Then we compare the results with different combinations 
    of intervention strategies in place to:
    * Compare the potential efficacies of different interventions and prioritise the ones that are going to help contain the virus.
    * Have a realistic estimate of the clinic capacity, PPE, ICU transfer and other supplies and logistical measures needed

    The graph below represents the disease transition dynamics for each individual included in the modelling studies. 
    The model we use is a deterministic, age-specific compartment model. We produce the analysis based on 
    {params.control_dict['numberOfIterations']} simulation runs over a range of possible parameters.
    ''')


def overview_population(params: Parameters):
    df = params.population_frame.copy()
    df['Population structure'] = df['Population_structure'].map('{:,.1f}%'.format)
    df['Number of residents'] = df['Population_structure'] * params.population / 100.0
    df['Number of residents'] = df['Number of residents'].astype(int)
    df = df[['Age', 'Population structure', 'Number of residents']]

    return [
        dcc.Markdown(textwrap.dedent(f'''
        The deterministic compartmental model requires:
        * Camp specific parameters: population size and age structure of the residents.
        * COVID specific parameters: asymptomatic infection rate, days remaining infectious, etc.
        ''')),
        dbc.Row([
            dbc.Col([
                html.Div(html.B(f'Population breakdown of {params.camp} camp with {params.population} residents')),
                dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
            ], width=4)
        ]),
        dcc.Markdown(textwrap.dedent(f'''
        Epidemiology simulations produce peak prevalence, the timing of peak prevalence as well as cumulative incidences for:
        * Symptomatic cases of infection
        * Hospitalisation demand person-days
        * Critical care demand person-days
        * Deaths
        '''))
    ]


@cache.memoize(timeout=cache_timeout)
def get_model_result(camp: str, profile: str):
    logging.info("Reading data for: " + camp + ", " + profile)
    mr = model_runner.get_result(CompartmentalModel.ID, profile, camp)
    assert mr is not None
    profile_df = facade.ps.get_params(CompartmentalModel.ID, profile).drop(columns=['Profile'])
    params = Parameters(facade.ps, camp, profile_df, {})
    report = load_report(mr, params)
    return mr, profile_df, params, report


@dash_app.callback(
    Output('main_section_part1', 'children'),
    [Input('_camp_param', 'children'), Input('_profile_param', 'children')],
)
def render_main_section_part1(camp, profile):
    mr, profile_df, params, report = get_model_result(camp, profile)

    prevalence = prevalence_all_table(report)
    peak_critical_care_demand = prevalence[prevalence['Outcome'] == 'Critical Care Demand']['Peak Number IQR'].iloc[0]

    prevalence_age = prevalence_age_table(report).reset_index()
    prevalence_age = prevalence_age.rename(columns={"level_1": "Age"})

    return [
        dcc.Markdown(textwrap.dedent(f'''
        ## 2. Base COVID-19 Epidemic Trajectory for profile "{profile}"

        Here we assume the epidemic spreads through the camp without any non-pharmaceutical intervention in place and the peak incidence 
        (number of cases), the timing and the cumulative case counts are all presented by interquartile range values (25%-75% quantiles)
        and they respectively represent the optimistic and pessimistic estimates of the spread of the virus given the uncertainty 
        in parameters estimated from epidemiological studies. In the simulations, we explore a range of reproduction numbers, 
        pre-symptomatic duration, rate of recovery, rate of severe infections and death rates based on estimates in the European 
        and Asian settings to nearly what is estimated from a high population density location like a cruise ship. 
        ''')),
        dbc.Row([
            dbc.Col([
                html.Div(html.B(f'Peak day and peak number for prevalences of different disease states of COVID-19')),
                dbc.Table.from_dataframe(prevalence, striped=True, bordered=True, hover=True)
            ], width=4)
        ]),
        dcc.Markdown(textwrap.dedent(f'''
        The death estimate is based on the fact the patients require critical care will receive appropriate treatment from the 
        {params.icu_count} ICU beds that are currently available in {params.camp}. The prevalence of death could be as 
        high as the peak critical care demand ({peak_critical_care_demand}) if camp residents are denied treatment at the 
        ICU facility on the island.  
        ''')),
        dbc.Row([
            dbc.Col([html.Div([
                html.B(f'Peak day and peak prevalences of different disease states of COVID-19 breakdown by age'),
            ])])
        ]),
        dbc.Row([
            dbc.Col([
                html.B(f'Incident Cases'),
                dbc.Table.from_dataframe(
                    prevalence_age[prevalence_age['level_0'] == 'Incident Cases'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4),
            dbc.Col([
                html.B(f'Hospital Demand'),
                dbc.Table.from_dataframe(
                    prevalence_age[prevalence_age['level_0'] == 'Hospital Demand'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4),
            dbc.Col([
                html.B(f'Critical Demand'),
                dbc.Table.from_dataframe(
                    prevalence_age[prevalence_age['level_0'] == 'Critical Demand'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4),
        ])
    ]


@dash_app.callback(
    Output('main_section_part2', 'children'),
    [Input('_camp_param', 'children'), Input('_profile_param', 'children')],
)
def render_main_section_part2(camp, profile):
    mr, profile_df, params, report = get_model_result(camp, profile)

    t_sim = params.control_dict['t_sim']
    cumulative_all = cumulative_all_table(report, params.population)
    cumulative_age = cumulative_age_table(report).reset_index()
    cumulative_age = cumulative_age.rename(columns={"level_1": "Age"})

    return [
        dbc.Row([
            dbc.Col([
                html.B(f'Cumulative case counts of different disease states of COVID-19 spanning {t_sim} days'),
                dbc.Table.from_dataframe(cumulative_all, striped=True, bordered=True, hover=True)
            ], width=4),
        ]),
        dcc.Markdown(textwrap.dedent(f'''Table above show cumulative counts, hospital-person days can be translated into
          projected medical costs or time required from medical staff if the medical cost and time taken is known for 
          treating one patient for a day.''')),

        dbc.Row([
            dbc.Col([html.Div([
                html.B(f'Cumulative case counts of different disease states of COVID-19 breakdown by age'),
            ])])
        ]),
        dbc.Row([
            dbc.Col([
                html.B(f'Symptomatic Cases'),
                dbc.Table.from_dataframe(
                    cumulative_age[cumulative_age['level_0'] == 'Symptomatic Cases'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4),
            dbc.Col([
                html.B(f'Hospital Person-Days'),
                dbc.Table.from_dataframe(
                    cumulative_age[cumulative_age['level_0'] == 'Hospital Person-Days'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4)
        ]),
        dbc.Row([
            dbc.Col([
                html.B(f'Critical Person-days'),
                dbc.Table.from_dataframe(
                    cumulative_age[cumulative_age['level_0'] == 'Critical Person-days'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4),
            dbc.Col([
                html.B(f'Deaths'),
                dbc.Table.from_dataframe(
                    cumulative_age[cumulative_age['level_0'] == 'Deaths'].drop(columns=['level_0']),
                    striped=True, bordered=True, hover=True)
            ], width=4)
        ])
    ]


@dash_app.callback(
    Output('main_section_part3', 'children'),
    [Input('_camp_param', 'children'), Input('_profile_param', 'children')],
)
def render_main_section_part2(camp, profile):
    mr, profile_df, params, report = get_model_result(camp, profile)

    # fig, ax = plt.subplots(1, 4, sharex='col', figsize=(16, 9), constrained_layout=True)
    #
    # columns_to_plot = ['Infected (symptomatic)', 'Hospitalised', 'Critical', 'Deaths']
    # i = 0
    # fontdict = {'fontsize': 15}
    # for column in columns_to_plot:
    #     sns.lineplot(x="Time", y=column, ci='iqr', data=df, ax=ax[i], estimator=np.median)
    #     ax[i].set_title(column, fontdict)
    #     i += 1
    # fig.suptitle(
    #     'Plots of changes in symptomatically infected cases, hopitalisation cases, critical care cases and death incidents over the course of simulation days',
    #     fontsize=20)
    #
    # fig = make_subplots(rows=1, cols=4)
    # columns_to_plot = ['Infected (symptomatic)', 'Hospitalised', 'Critical', 'Deaths']
    # col = 0
    # for column in columns_to_plot:
    #     # sns.lineplot(x="Time", y=column, ci='iqr', data=df, ax=ax[i], estimator=np.median)
    #     # ax[i].set_title(column, fontdict)
    #
    #     fig.add_trace(
    #         go.Scatter(x=report['Time'], y=report[column]),
    #         row=1, col=col
    #     )
    #     col += 1



    # fig.add_trace(
    #     go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    #     row=1, col=1
    # )
    #
    # fig.add_trace(
    #     go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    #     row=1, col=2
    # )

    # fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")


    #fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])

    return [
        # dcc.Graph(
        #     id='example-graph-2',
        #     figure=fig
        # )
    ]


def load_report(mr, params) -> pd.DataFrame:
    df = mr.get('report')
    df.R0 = df.R0.apply(lambda x: round(complex(x).real, 1))
    df_temp = df.drop(['Time', 'R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'],
                      axis=1)
    df_temp = df_temp * params.population
    df.update(df_temp)
    return df


def prevalence_all_table(df):
    # calculate Peak Day IQR and Peak Number IQR for each of the 'incident' variables to table
    table_params = ['Infected (symptomatic)', 'Hospitalised', 'Critical', 'Change in Deaths']
    grouped = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])
    incident_rs = {}
    for index, group in grouped:
        # for each RO value find out the peak days for each table params
        group = group.set_index('Time')
        incident = {}
        for param in table_params:
            incident[param] = (group.loc[:, param].idxmax(), group.loc[:, param].max())
        incident_rs[index] = incident
    iqr_table = {}
    for param in table_params:
        day = []
        number = []
        for elem in incident_rs.values():
            day.append(elem[param][0])
            number.append(elem[param][1])
        q75_day, q25_day = np.percentile(day, [75, 25])
        q75_number, q25_number = np.percentile(number, [75, 25])
        iqr_table[param] = (
        (int(round(q25_day)), int(round(q75_day))), (int(round(q25_number)), int(round(q75_number))))
    table_columns = {'Infected (symptomatic)': 'Prevalence of Symptomatic Cases',
                     'Hospitalised': 'Hospitalisation Demand',
                     'Critical': 'Critical Care Demand', 'Change in Deaths': 'Prevalence of Deaths'}
    outcome = []
    peak_day = []
    peak_number = []
    for param in table_params:
        outcome.append(table_columns[param])
        peak_day.append(f'{iqr_table[param][0][0]}-{iqr_table[param][0][1]}')
        peak_number.append(f'{iqr_table[param][1][0]}-{iqr_table[param][1][1]}')
    data = {'Outcome': outcome, 'Peak Day IQR': peak_day, 'Peak Number IQR': peak_number}
    return pd.DataFrame.from_dict(data)


def prevalence_age_table(df):
    # calculate age specific Peak Day IQR and Peak Number IQR for each of the 'prevalent' variables to contruct table
    table_params = ['Infected (symptomatic)', 'Hospitalised', 'Critical']
    grouped = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])
    prevalent_age = {}
    params_age = []
    for index, group in grouped:
        # for each RO value find out the peak days for each table params
        group = group.set_index('Time')
        prevalent = {}
        for param in table_params:
            for column in df.columns:
                if column.startswith(param):
                    prevalent[column] = (group.loc[:, column].idxmax(), group.loc[:, column].max())
                    params_age.append(column)
        prevalent_age[index] = prevalent
    params_age_dedup = list(set(params_age))
    prevalent_age_bucket = {}
    for elem in prevalent_age.values():
        for key, value in elem.items():
            if key in prevalent_age_bucket:
                prevalent_age_bucket[key].append(value)
            else:
                prevalent_age_bucket[key] = [value]
    iqr_table_age = {}
    for key, value in prevalent_age_bucket.items():
        day = [x[0] for x in value]
        number = [x[1] for x in value]
        q75_day, q25_day = np.percentile(day, [75, 25])
        q75_number, q25_number = np.percentile(number, [75, 25])
        iqr_table_age[key] = (
        (int(round(q25_day)), int(round(q75_day))), (int(round(q25_number)), int(round(q75_number))))
    arrays = [np.array(['Incident Cases', 'Incident Cases', 'Incident Cases', 'Incident Cases', 'Incident Cases',
                        'Incident Cases', 'Incident Cases', 'Incident Cases', 'Incident Cases', 'Hospital Demand',
                        'Hospital Demand', 'Hospital Demand', 'Hospital Demand', 'Hospital Demand', 'Hospital Demand',
                        'Hospital Demand', 'Hospital Demand', 'Hospital Demand', 'Critical Demand', 'Critical Demand',
                        'Critical Demand', 'Critical Demand', 'Critical Demand', 'Critical Demand', 'Critical Demand',
                        'Critical Demand', 'Critical Demand']),
              np.array(
                  ['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years',
                   '60-69 years', '70+ years', 'all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years',
                   '40-49 years', '50-59 years', '60-69 years', '70+ years', 'all ages', '<9 years', '10-19 years',
                   '20-29 years', '30-39 years', '40-49 years', '50-59 years', '60-69 years', '70+ years'])]
    peak_day = np.empty(27, dtype="S10")
    peak_number = np.empty(27, dtype="S10")
    for key, item in iqr_table_age.items():
        if key == 'Infected (symptomatic)':
            peak_day[0] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
            peak_number[0] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif key == 'Hospitalised':
            peak_day[9] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
            peak_number[9] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif key == 'Critical':
            peak_day[18] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
            peak_number[18] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '0-9' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[1] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[1] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[10] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[10] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[19] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[19] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '10-19' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[2] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[2] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[11] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[11] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[20] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[20] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '20-29' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[3] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[3] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[12] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[12] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[21] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[21] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '30-39' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[4] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[4] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[13] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[13] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[22] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[22] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '40-49' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[5] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[5] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[14] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[14] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[23] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[23] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '50-59' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[6] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[6] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[15] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[15] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[24] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[24] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '60-69' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[7] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[7] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[16] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[16] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[25] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[25] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
        elif '70+' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[8] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[8] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[17] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[17] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[26] = f'{iqr_table_age[key][0][0]}-{iqr_table_age[key][0][1]}'
                peak_number[26] = f'{iqr_table_age[key][1][0]}-{iqr_table_age[key][1][1]}'
    d = {'Peak Day, IQR': peak_day.astype(str), 'Peak Number, IQR': peak_number.astype(str)}
    return pd.DataFrame(data=d, index=arrays)


def cumulative_all_table(df, N):
    # now we try to calculate the total count
    # cases: (N-exposed)*0.5 since the asymptomatic rate is 0.5
    # hopistal days: cumulative count of hospitalisation bucket
    # critical days: cumulative count of critical days
    # deaths: we already have that from the frame
    table_params = ['Susceptible', 'Hospitalised', 'Critical', 'Deaths']
    grouped = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])
    cumulative_all = {}
    for index, group in grouped:
        # for each RO value find out the peak days for each table params
        group = group.set_index('Time')
        cumulative = {}
        for param in table_params:
            if param == 'Susceptible':
                param09 = 'Susceptible: 0-9'
                param1019 = 'Susceptible: 10-19'
                param2029 = 'Susceptible: 20-29'
                param3039 = 'Susceptible: 30-39'
                param4049 = 'Susceptible: 40-49'
                param5059 = 'Susceptible: 50-59'
                param6069 = 'Susceptible: 60-69'
                param7079 = 'Susceptible: 70+'
                cumulative[param] = ((N * 0.2105 - (group[param09].tail(1).values[0])) * 0.4 +
                                     (N * 0.1734 - (group[param1019].tail(1).values[0])) * 0.25 +
                                     (N * 0.2635 - (group[param2029].tail(1).values[0])) * 0.37 +
                                     (N * 0.1716 - (group[param3039].tail(1).values[0])) * 0.42 +
                                     (N * 0.0924 - (group[param4049].tail(1).values[0])) * 0.51 +
                                     (N * 0.0555 - (group[param5059].tail(1).values[0])) * 0.59 +
                                     (N * 0.0254 - (group[param6069].tail(1).values[0])) * 0.72 +
                                     (N * 0.0077 - (group[param7079].tail(1).values[0])) * 0.76)
            elif param == 'Deaths':
                cumulative[param] = (group[param].tail(1).values[0])
            elif param == 'Hospitalised' or param == 'Critical':
                cumulative[param] = (group[param].sum())
        cumulative_all[index] = cumulative
    cumulative_count = []
    for param in table_params:
        count = []
        for elem in cumulative_all.values():
            count.append(elem[param])
        q75_count, q25_count = np.percentile(count, [75, 25])
        cumulative_count.append(f'{int(round(q25_count))}-{int(round(q75_count))}')
    data = {'Totals': ['Symptomatic Cases', 'Hospital Person-Days', 'Critical Person-days', 'Deaths'],
            'Counts': cumulative_count}
    return pd.DataFrame.from_dict(data)


def cumulative_age_table(df):
    # need to have an age break down for this as well
    # 1 month 3 month and 6 month breakdown
    arrays = [np.array(
        ['Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases',
         'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 'Symptomatic Cases', 'Hospital Person-Days',
         'Hospital Person-Days', 'Hospital Person-Days', 'Hospital Person-Days', 'Hospital Person-Days',
         'Hospital Person-Days',
         'Hospital Person-Days', 'Hospital Person-Days', 'Hospital Person-Days', 'Critical Person-days',
         'Critical Person-days',
         'Critical Person-days', 'Critical Person-days', 'Critical Person-days', 'Critical Person-days',
         'Critical Person-days',
         'Critical Person-days', 'Critical Person-days', 'Deaths', 'Deaths', 'Deaths', 'Deaths', 'Deaths', 'Deaths',
         'Deaths', 'Deaths',
         'Deaths']),
              np.array(
                  ['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years',
                   '60-69 years', '70+ years', 'all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years',
                   '40-49 years', '50-59 years', '60-69 years', '70+ years', 'all ages', '<9 years', '10-19 years',
                   '20-29 years', '30-39 years', '40-49 years', '50-59 years', '60-69 years', '70+ years', 'all ages',
                   '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years', '60-69 years',
                   '70+ years'])]
    table_params = ['Susceptible', 'Hospitalised', 'Critical', 'Deaths']
    params_select = ['Susceptible:', 'Deaths']
    params_accu = ['Hospitalised', 'Critical']
    columns_to_select = []
    columns_to_acc = []
    for column in df.columns:
        for param in params_select:
            if column.startswith(param):
                columns_to_select.append(column)
        for param in params_accu:
            if column.startswith(param):
                columns_to_acc.append(column)
    first_month_select = {}
    three_month_select = {}
    six_month_select = {}

    for column in columns_to_select:
        if 'Susceptible:' in column:
            if '0-9' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.4).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.4).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.4).quantile([.25, .75])
            elif '10-19' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.25).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.25).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.25).quantile([.25, .75])
            elif '20-29' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.37).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.37).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.37).quantile([.25, .75])
            elif '30-39' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.42).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.42).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.42).quantile([.25, .75])
            elif '40-49' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.51).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.51).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.51).quantile([.25, .75])
            elif '50-59' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.59).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.59).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.59).quantile([.25, .75])
            elif '60-69' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.72).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.72).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.72).quantile([.25, .75])
            elif '70+' in column:
                first_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_first_month_diff)[column].mul(-0.76).quantile([.25, .75])
                three_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_third_month_diff)[column].mul(-0.76).quantile([.25, .75])
                six_month_select[column] = \
                df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                    [column, 'Time']].apply(find_sixth_month_diff)[column].mul(-0.76).quantile([.25, .75])
        else:
            first_month_select[column] = \
            df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                [column, 'Time']].apply(find_first_month)[column].quantile([.25, .75])
            three_month_select[column] = \
            df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                [column, 'Time']].apply(find_third_month)[column].quantile([.25, .75])
            six_month_select[column] = \
            df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
                [column, 'Time']].apply(find_sixth_month)[column].quantile([.25, .75])

    first_month_select['Susceptible'] = {0.25: 0, 0.75: 0}
    three_month_select['Susceptible'] = {0.25: 0, 0.75: 0}
    six_month_select['Susceptible'] = {0.25: 0, 0.75: 0}
    for column in columns_to_select:
        if 'Susceptible:' in column:
            first_month_select['Susceptible'][0.25] += first_month_select[column][0.25]
            first_month_select['Susceptible'][0.75] += first_month_select[column][0.75]
            three_month_select['Susceptible'][0.25] += three_month_select[column][0.25]
            three_month_select['Susceptible'][0.75] += three_month_select[column][0.75]
            six_month_select['Susceptible'][0.25] += six_month_select[column][0.25]
            six_month_select['Susceptible'][0.75] += six_month_select[column][0.75]
    first_month_accu = {}
    three_month_accu = {}
    six_month_accu = {}
    for column in columns_to_acc:
        first_month_accu[column] = \
        df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
            [column, 'Time']].apply(find_one_month)[column].quantile([.25, .75])
        three_month_accu[column] = \
        df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
            [column, 'Time']].apply(find_three_months)[column].quantile([.25, .75])
        six_month_accu[column] = \
        df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
            [column, 'Time']].apply(find_six_months)[column].quantile([.25, .75])
    first_month = _merge(first_month_select, first_month_accu)
    third_month = _merge(three_month_select, three_month_accu)
    sixth_month = _merge(six_month_select, six_month_accu)
    first_month_count = np.empty(36, dtype="S15")
    for key, item in first_month.items():
        if key == 'Susceptible':
            first_month_count[0] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif key == 'Hospitalised':
            first_month_count[9] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif key == 'Critical':
            first_month_count[18] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif key == 'Deaths':
            first_month_count[27] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '0-9' in key:
            if key.startswith('Susceptible'):
                first_month_count[1] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[10] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[19] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[28] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '10-19' in key:
            if key.startswith('Susceptible'):
                first_month_count[2] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[11] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[20] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[29] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '20-29' in key:
            if key.startswith('Susceptible'):
                first_month_count[3] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[12] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[21] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[30] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '30-39' in key:
            if key.startswith('Susceptible'):
                first_month_count[4] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[13] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[22] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[31] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '40-49' in key:
            if key.startswith('Susceptible'):
                first_month_count[5] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[14] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[23] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[32] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '50-59' in key:
            if key.startswith('Susceptible'):
                first_month_count[6] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[15] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[24] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[33] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '60-69' in key:
            if key.startswith('Susceptible'):
                first_month_count[7] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[16] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[25] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[34] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
        elif '70+' in key:
            if key.startswith('Susceptible'):
                first_month_count[8] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[17] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[26] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[35] = f'{int(round(first_month[key][0.25]))}-{int(round(first_month[key][0.75]))}'
    three_month_count = np.empty(36, dtype="S15")
    for key, item in third_month.items():
        if key == 'Susceptible':
            three_month_count[0] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif key == 'Hospitalised':
            three_month_count[9] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif key == 'Critical':
            three_month_count[18] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif key == 'Deaths':
            three_month_count[27] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '0-9' in key:
            if key.startswith('Susceptible'):
                three_month_count[1] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[10] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[19] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[28] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '10-19' in key:
            if key.startswith('Susceptible'):
                three_month_count[2] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[11] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[20] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[29] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '20-29' in key:
            if key.startswith('Susceptible'):
                three_month_count[3] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[12] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[21] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[30] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '30-39' in key:
            if key.startswith('Susceptible'):
                three_month_count[4] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[13] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[22] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[31] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '40-49' in key:
            if key.startswith('Susceptible'):
                three_month_count[5] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[14] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[23] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[32] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '50-59' in key:
            if key.startswith('Susceptible'):
                three_month_count[6] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[15] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[24] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[33] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '60-69' in key:
            if key.startswith('Susceptible'):
                three_month_count[7] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[16] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[25] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[34] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
        elif '70+' in key:
            if key.startswith('Susceptible'):
                three_month_count[8] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[17] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[26] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[35] = f'{int(round(third_month[key][0.25]))}-{int(round(third_month[key][0.75]))}'
    six_month_count = np.empty(36, dtype="S10")
    for key, item in sixth_month.items():
        if key == 'Susceptible':
            six_month_count[0] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif key == 'Hospitalised':
            six_month_count[9] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif key == 'Critical':
            six_month_count[18] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif key == 'Deaths':
            six_month_count[27] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '0-9' in key:
            if key.startswith('Susceptible'):
                six_month_count[1] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[10] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[19] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[28] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '10-19' in key:
            if key.startswith('Susceptible'):
                six_month_count[2] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[11] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[20] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[29] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '20-29' in key:
            if key.startswith('Susceptible'):
                six_month_count[3] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[12] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[21] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[30] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '30-39' in key:
            if key.startswith('Susceptible'):
                six_month_count[4] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[13] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[22] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[31] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '40-49' in key:
            if key.startswith('Susceptible'):
                six_month_count[5] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[14] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[23] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[32] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '50-59' in key:
            if key.startswith('Susceptible'):
                six_month_count[6] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[15] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[24] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[33] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '60-69' in key:
            if key.startswith('Susceptible'):
                six_month_count[7] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[16] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[25] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[34] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
        elif '70+' in key:
            if key.startswith('Susceptible'):
                six_month_count[8] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[17] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[26] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[35] = f'{int(round(sixth_month[key][0.25]))}-{int(round(sixth_month[key][0.75]))}'
    d = {'First month': first_month_count.astype(str), 'First three months': three_month_count.astype(str),
         'First six months': six_month_count.astype(str)}
    return pd.DataFrame(data=d, index=arrays)


def find_first_month(df):
    return df[df['Time'] == 30]


def find_third_month(df):
    return df[df['Time'] == 90]


def find_sixth_month(df):
    return df[df['Time'] == 180]


def find_first_month_diff(df):
    return df[df['Time'] <= 30].diff(periods=30).tail(1)


def find_third_month_diff(df):
    return df[df['Time'] <= 90].diff(periods=90).tail(1)


def find_sixth_month_diff(df):
    return df[df['Time'] <= 180].diff(periods=180).tail(1)


def find_one_month(df):
    return df[df['Time'] <= 30].cumsum().tail(1)


def find_three_months(df):
    return df[df['Time'] <= 90].cumsum().tail(1)


def find_six_months(df):
    return df[df['Time'] <= 180].cumsum().tail(1)


def _merge(dict1, dict2):
	res = {**dict1, **dict2}
	return res