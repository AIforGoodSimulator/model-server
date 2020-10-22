# from functools import reduce
# import logging
import textwrap
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
# import dash_bootstrap_components as dbc

# from ai4good.models.cm.cm_model import CompartmentalModel
# from ai4good.models.cm.initialise_parameters import Parameters
# from ai4good.webapp.apps import dash_app, facade, model_runner, cache, local_cache, cache_timeout
# from ai4good.webapp.cm_model_report_utils import *


# @cache.memoize(timeout=cache_timeout)
def layout():
    # camp, profile, cmp_profiles
    # _, profile_df, params, _ = get_model_result(camp, profile)
    return html.Div(
        [
            dcc.Markdown(disclaimer(), style={'margin': 30}),
            html.H1(f'AI for Good Simulator: Model Results Dashboard for the Refugee Camp', style={
                    'margin': 30}),
            # dcc.Markdown(overview(camp, params), style={'margin': 30}),
            dcc.Markdown(high_level_message_1(), style={'margin': 30}),
            dcc.Markdown(high_level_message_2(), style={'margin': 30}),
            dcc.Markdown(high_level_message_3(), style={'margin': 30}),
            dcc.Markdown(high_level_message_4(), style={'margin': 30}),
            dcc.Markdown(high_level_message_5(), style={'margin': 30}),
            dcc.Loading(
                html.Div([], id='main_section_part2', style={'margin': 30})),
            # base_profile_chart_section(),
            # dcc.Loading(html.Div([], id='cmp_section', style={'margin': 30})),
            # html.Div(camp, id='_camp_param', style={'display': 'none'}),
            # html.Div(profile, id='_profile_param', style={'display': 'none'}),
            # html.Div('¬'.join(cmp_profiles), id='_cmp_profiles',
            #          style={'display': 'none'})
        ], style={'margin': 50}
    )


def disclaimer():
    return textwrap.dedent('''
    ##### Disclaimer: The model results are produced based on the parameters being inputted and contain abstractions of the reality. The exact values from the model predictions are not validated by real data from your refugee camp but the rough values and ranges of reduction with relevent intervention methods are explored to provide value in planning efforts.
    ''').replace('\n', ' ')


# def overview(camp: str, params: Parameters):
#     return textwrap.dedent(f'''
#     ## 1. Overview
#     This report provides simulation-based estimates for COVID-19 epidemic scenarios for the {camp} camp.
#     There are an estimated {int(params.population)} people currently living in the camp. Through epidemiology simulations,
#     we estimated peak counts, the timing of peak counts as well as cumulative counts for new symptomatic cases, hospitalisation demand person-days,
#     critical care demand person-days and deaths for an unmitigated epidemic.  Then we compare the results with different combinations
#     of intervention strategies in place to:
#     * Compare the potential efficacies of different interventions and prioritise the ones that are going to help contain the virus.
#     * Have a realistic estimate of the clinic capacity, PPE, ICU transfer and other supplies and logistical measures needed.
#     ''')


def high_level_message_1():
    return textwrap.dedent(f'''
    ## 1. The case for a longer, more sustainable program
    As shown by comparing implementing different interventions for 1 month, 3 month and 6 month during the period of epidemic,
    it is important to prioritize long-term non-pharmaceutical intervention over short-term interventions. 
    ''')


def high_level_message_2():
    return textwrap.dedent(f'''
    ## 2. Switch on invervention like lockdown at the right time
    It is also important to switch on interventions at the correct time, rather than having them on all the time.
    This is shown by comparing implementing different interventions starting at different points during the epidemic. 
    ''')


def high_level_message_3():
    return textwrap.dedent(f'''
    ## 3. Reducing activities happening within the camp might not be that effective
    Reducing activities for the residents might not be an effective intervention. A camp lockdown is comparatively more effective 
    but must not be implemented the whole time as people will start becoming more relaxed. 
    ''')


def high_level_message_4():
    return textwrap.dedent(f'''
    ## 4. Isolating the symptomatic might not be that effective
    Isolating the symptomatically infected residents can be effective, but it is resource intensive and there is uncertainty about the final outcome. 
    ''')


def high_level_message_5():
    return textwrap.dedent(f'''
    ## 5. Characteristics of non-pharmaceutical interventions that apply to your camp
    Each non-pharmaceutical intervention has its different characteristics. It is important to implement a combinatorial approach, 
    where using several less-effective policies laid over each other prove to be more effective than using any single intervention on its own. 
    ''')


# @local_cache.memoize(timeout=cache_timeout)
# def get_model_result(camp: str, profile: str):
#     logging.info("Reading data for: " + camp + ", " + profile)
#     mr = model_runner.get_result(CompartmentalModel.ID, profile, camp)
#     assert mr is not None
#     profile_df = facade.ps.get_params(
#         CompartmentalModel.ID, profile).drop(columns=['Profile'])
#     params = Parameters(facade.ps, camp, profile_df, {})
#     report = load_report(mr, params)
#     return mr, profile_df, params, report


# def base_profile_chart_section():
#     options = [
#         {'label': 'All ages', 'value': 'ALL'},
#         {'label': '<9 years', 'value': '0-9'},
#         {'label': '10-19 years', 'value': '10-19'},
#         {'label': '20-29 years', 'value': '20-29'},
#         {'label': '30-39 years', 'value': '30-39'},
#         {'label': '40-49 years', 'value': '40-49'},
#         {'label': '50-59 years', 'value': '50-59'},
#         {'label': '60-69 years', 'value': '60-69'},
#         {'label': '70+ years', 'value': '70+'}
#     ]

#     return html.Div([
#         dbc.Row([
#             dbc.Col([
#                 html.B('Plots of changes in symptomatically infected cases, hopitalisation cases, critical care cases '
#                        'and death incidents over the course of simulation days'),
#                 dcc.Dropdown(
#                     id='charts_age_dropdown',
#                     options=options,
#                     value='ALL',
#                     clearable=False,
#                     style={'margin-top': 5}
#                 ),
#             ], width=6)
#         ]),
#         dbc.Row([
#             dbc.Col([
#                 dcc.Loading(html.Div([], id='chart_section_plot_container')),
#             ], width=12)
#         ])
#     ], style={'margin': 30})


# @dash_app.callback(
#     Output('chart_section_plot_container', 'children'),
#     [Input('_camp_param', 'children'), Input('_profile_param',
#                                              'children'), Input('charts_age_dropdown', 'value')],
# )
# def render_main_section_charts(camp, profile, age_to_plot):
#     mr, profile_df, params, report = get_model_result(camp, profile)
#     logging.info(f"Plotting {age_to_plot}")

#     columns_to_plot = ['Infected (symptomatic)',
#                        'Hospitalised', 'Critical', 'Deaths']
#     fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
#                         vertical_spacing=0.05,
#                         horizontal_spacing=0.05,
#                         subplot_titles=columns_to_plot)

#     for i, col in enumerate(columns_to_plot):
#         row_idx = int(i % 2 + 1)
#         col_idx = int(i / 2 + 1)
#         if age_to_plot != 'ALL':
#             age_cols = [c for c in report.columns if (
#                 age_to_plot in c) and (c.startswith(col))]
#             assert len(age_cols) == 1
#             col = age_cols[0]
#         p1, p2 = plot_iqr(report, col)
#         fig.add_trace(p1, row=row_idx, col=col_idx)
#         fig.add_trace(p2, row=row_idx, col=col_idx)
#         fig.update_yaxes(title_text=col, row=row_idx, col=col_idx)

#     x_title = 'Time, days'
#     fig.update_xaxes(title_text=x_title, row=2, col=1)
#     fig.update_xaxes(title_text=x_title, row=2, col=2)

#     fig.update_traces(mode='lines')
#     fig.update_layout(
#         margin=dict(l=0, r=0, t=30, b=0),
#         height=800,
#         showlegend=False
#     )

#     return [
#         dcc.Graph(
#             id='plot_all_fig',
#             figure=fig,
#             style={'width': '100%'}
#         )
#     ]


# @dash_app.callback(
#     Output('cmp_section', 'children'),
#     [Input('_camp_param', 'children'), Input('_profile_param',
#                                              'children'), Input('_cmp_profiles', 'children')],
# )
# @cache.memoize(timeout=cache_timeout)
# def interventions(camp, profile, cmp_profiles):

#     _, base_profile, base_params, base_df = get_model_result(camp, profile)
#     profiles = cmp_profiles.split('¬')

#     if len(profiles) == 0 or (len(profiles) == 1 and len(profiles[0].strip()) == 0):
#         return []

#     intervention_content = [intervention(
#         camp, p, i+1, base_df, base_params, base_profile, profile) for i, p in enumerate(profiles)]
#     intervention_content = reduce(list.__add__, intervention_content)

#     return [
#         dcc.Markdown(textwrap.dedent(f'''
#         ## 2. Intervention scenarios

#         We compare each intervention scenario to baseline. Baseline charts are in blue as before, intervention charts
#         are in red.
#          ''')),
#         html.Div(intervention_content)
#     ]


# def intervention(camp, cmp_profile_name, idx, base_df, base_params, base_profile, base_profile_name):
#     _, cmp_profile, cmp_params, cmp_df = get_model_result(
#         camp, cmp_profile_name)

#     return intervention_plots(base_df, cmp_df, base_profile_name, cmp_profile_name)


# def intervention_plots(base_df, cmp_df, base_profile_name, cmp_profile_name):
#     columns_to_plot = ['Infected (symptomatic)',
#                        'Hospitalised', 'Critical', 'Deaths']
#     fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
#                         vertical_spacing=0.05,
#                         horizontal_spacing=0.05,
#                         subplot_titles=columns_to_plot)

#     for i, col in enumerate(columns_to_plot):
#         row_idx = int(i % 2 + 1)
#         col_idx = int(i / 2 + 1)

#         b1, b2 = plot_iqr(base_df, col, ci_name_prefix=f'{base_profile_name} ',
#                           estimator_name=f'{base_profile_name} median')
#         c1, c2 = plot_iqr(cmp_df, col, ci_name_prefix=f'{cmp_profile_name} ',
#                           estimator_name=f'{cmp_profile_name} median', color_scheme=color_scheme_secondary)

#         fig.add_trace(b2, row=row_idx, col=col_idx)
#         fig.add_trace(c2, row=row_idx, col=col_idx)
#         fig.add_trace(b1, row=row_idx, col=col_idx)
#         fig.add_trace(c1, row=row_idx, col=col_idx)

#         fig.update_yaxes(title_text=col, row=row_idx, col=col_idx)
#     x_title = 'Time, days'
#     fig.update_xaxes(title_text=x_title, row=2, col=1)
#     fig.update_xaxes(title_text=x_title, row=2, col=2)

#     fig.update_traces(mode='lines')
#     fig.update_layout(
#         margin=dict(l=0, r=0, t=30, b=0),
#         height=800,
#         showlegend=False
#     )

#     return [
#         dcc.Graph(
#             id='intervention_plots_fig',
#             figure=fig,
#             style={'width': '100%'}
#         )
#     ]


# color_scheme_main = ['rgba(0, 176, 246, 0.2)',
#                      'rgba(255, 255, 255,0)', 'rgb(0, 176, 246)']
# color_scheme_secondary = [
#     'rgba(245, 186, 186, 0.5)', 'rgba(255, 255, 255,0)', 'rgb(255, 0, 0)']


# def plot_iqr(df: pd.DataFrame, y_col: str,
#              x_col='Time', estimator=np.median, estimator_name='median', ci_name_prefix='',
#              iqr_low=0.25, iqr_high=0.75,
#              color_scheme=color_scheme_main):
#     grouped = df.groupby(x_col)[y_col]
#     est = grouped.agg(estimator)
#     cis = pd.DataFrame(np.c_[grouped.quantile(iqr_low), grouped.quantile(iqr_high)], index=est.index,
#                        columns=["low", "high"]).stack().reset_index()

#     x = est.index.values.tolist()
#     x_rev = x[::-1]

#     y_upper = cis[cis['level_1'] == 'high'][0].values.tolist()
#     y_lower = cis[cis['level_1'] == 'low'][0].values.tolist()
#     y_lower = y_lower[::-1]

#     p1 = go.Scatter(
#         x=x + x_rev, y=y_upper + y_lower, fill='toself', fillcolor=color_scheme[0],
#         line_color=color_scheme[1], name=f'{ci_name_prefix}{iqr_low*100}% to {iqr_high*100}% interval')
#     p2 = go.Scatter(
#         x=x, y=est, line_color=color_scheme[2], name=f'{y_col} {estimator_name}')
#     return p1, p2