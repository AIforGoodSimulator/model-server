import textwrap
import dash_html_components as html
import dash_core_components as dcc
from ai4good.webapp.apps import dash_app, facade, model_runner, cache, local_cache, cache_timeout
from plotly.subplots import make_subplots
from ai4good.webapp.model_results_config import model_profile_config
from collections import defaultdict
from ai4good.webapp.model_results_utils import load_report
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from ai4good.models.cm.initialise_parameters import Parameters # this needs to be changed later
import plotly.graph_objs as go
from ai4good.utils.logger_util import get_logger
from plotly.colors import DEFAULT_PLOTLY_COLORS

logger = get_logger(__name__)


CAMP = 'Moria'
# this logic needs to be changed later where the user input params will be stored
# can read in the camp parameters here directly from the user input thus avoiding retrieving data from the system elsewhere

# @cache.memoize(timeout=cache_timeout)
def layout():
    # camp, profile, cmp_profiles
    # _, profile_df, params, _ = get_model_result(camp, profile)
    return html.Div(
        [
            html.A(html.Button('Print', className="btn btn-light"), href='javascript:window.print()', className='d-print-none', style={"float": 'right'}),
            dcc.Markdown(disclaimer(), style={'margin': 30}),
            html.H1(f'AI for Good Simulator: Model Results Dashboard for the Refugee Camp', style={
                    'margin': 30}),
            # dcc.Markdown(overview(camp, params), style={'margin': 30}),
            dcc.Markdown(high_level_message_1(), style={'margin': 30}),
            html.Div(render_message_1_plots(), style={'margin': 30}),
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


@local_cache.memoize(timeout=cache_timeout)
def get_model_result_message(message_key):
    logger.info(f"Reading data for high level message: {message_key}")
    model_profile_report_dict = defaultdict(dict)
    for model in model_profile_config[message_key].keys():
        if len(model_profile_config[message_key][model])>0:
            for profile in model_profile_config[message_key][model]:
                try:
                    mr = model_runner.get_result(model, profile, CAMP)
                    profile_df = facade.ps.get_params(model, profile).drop(columns=['Profile'])
                    params = Parameters(facade.ps, CAMP, profile_df, {})
                    report = load_report(mr, params)
                    assert mr is not None
                    model_profile_report_dict[model][profile] = report
                except:
                    logger.info(f"Unable to load result for model: ({model}, {profile}, {CAMP}).")
    return model_profile_report_dict


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

color_scheme_updated = DEFAULT_PLOTLY_COLORS

def render_message_1_plots():
    model_profile_report_dict = get_model_result_message("message_1")
    columns_to_plot = ['Infected (symptomatic)']
    fig = make_subplots()
    col = columns_to_plot[0]
    row_idx = 1
    for profile in model_profile_report_dict["compartmental-model"].keys():
        if profile == 'baseline':
            label_name = 'No interventions'
        elif profile == 'better_hygiene_one_month':
            label_name = 'Wearing facemask for 1 month'
        elif profile == 'better_hygiene_three_month':
            label_name = 'Wearing facemask for 3 month '
        elif profile == 'better_hygiene_six_month':
            label_name = 'Using Hand Sanitizers & facemask for 6 month'

        else:
            label_name = 'default'

        label_to_plot = [label_name]


        p1,p2= plot_iqr(model_profile_report_dict["compartmental-model"][profile], col,label_to_plot, ci_name_prefix=label_name + ' ')
        logger.info(f'plot on {row_idx}')
        fig.add_trace(p1, row=1, col=1)
        fig.add_trace(p2, row=1, col=1)
        fig.update_yaxes(title_text=col, row=row_idx, col=1)
        if row_idx < len(DEFAULT_PLOTLY_COLORS):
            fig["data"][2 * (row_idx - 1)]["line"]["color"] = color_scheme_updated[row_idx - 1] #Curve Color
            fig["data"][(2 * row_idx) - 1]["line"]["color"] = color_scheme_updated[row_idx - 1] #IQR Colour
            fig["data"][2 * (row_idx - 1)]["opacity"] = 0.2 #IQR Opacity
        row_idx += 1
        
    x_title = 'Time, days'
    fig.update_xaxes(title_text=x_title, row=1, col=1)
    # fig.update_xaxes(title_text=x_title, row=2, col=1)
    # fig.update_xaxes(title_text=x_title, row=3, col=1)
    # fig.update_xaxes(title_text=x_title, row=4, col=1)
    fig.update_traces(mode='lines')
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        showlegend=True

    )

    return [
        dcc.Graph(
            id='plot_all_fig',
            figure=fig,
            style={'width': '100%'}
        )
    ]

color_scheme_main = ['rgba(0, 176, 246, 0.2)', 'rgba(130, 190, 240, 1)', 'rgb(0, 176, 246)']
color_scheme_secondary = ['rgba(245, 240, 240, 0.5)']


def plot_iqr(df: pd.DataFrame, y_col: str,graph_label:str,
             x_col='Time', estimator=np.median, estimator_name='median', ci_name_prefix='',
             iqr_low=0.25, iqr_high=0.75,
             color_scheme=color_scheme_main):
    grouped = df.groupby(x_col)[y_col]
    est = grouped.agg(estimator)
    cis = pd.DataFrame(np.c_[grouped.quantile(iqr_low), grouped.quantile(iqr_high)], index=est.index,
                       columns=["low", "high"]).stack().reset_index()

    x = est.index.values.tolist()
    x_rev = x[::-1]

    y_upper = cis[cis['level_1'] == 'high'][0].values.tolist()
    y_lower = cis[cis['level_1'] == 'low'][0].values.tolist()
    y_lower = y_lower[::-1]
    
    p1 = go.Scatter(
        x=x + x_rev, y=y_upper + y_lower, fill='toself',  line_color=color_scheme[1],
         name=f'{ci_name_prefix}{iqr_low*100}% to {iqr_high*100}% interval', hoverinfo="skip")
    p2 = go.Scatter(x=x, y=est,  name=f'{graph_label}', line=dict(width=6))
    return p1,p2

# @dash_app.callback(
#     Output('cmp_section', 'children'),
#     [Input('_camp_param', 'children'), Input('_profile_param', 'children'), Input('_cmp_profiles', 'children')],
# )
# @cache.memoize(timeout=cache_timeout)
# def interventions(camp, profile, cmp_profiles):

#     _, base_profile, base_params, base_df = get_model_result(camp, profile)
#     profiles = cmp_profiles.split('¬')

#     if len(profiles) == 0 or (len(profiles) == 1 and len(profiles[0].strip()) == 0):
#         return []

#     intervention_content = [intervention(camp, p, i+1, base_df, base_params, base_profile, profile) for i, p in enumerate(profiles)]
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
#     _, cmp_profile, cmp_params, cmp_df = get_model_result(camp, cmp_profile_name)

#     tbl = diff_table(base_df, cmp_df, cmp_params.population)
#     profile_diff = profile_diff_tbl(base_profile, base_params, cmp_profile, cmp_params)

#     return [
#         dcc.Markdown(textwrap.dedent(f'''
#         ## 2.{idx} {cmp_profile_name} intervention comparison
        
#         Here we compare {cmp_profile_name} model profile with base {base_profile_name} profile explored in the main 
#         section. Compared to base profile {cmp_profile_name} has following changes:
#         ''')),
#         dbc.Row([
#             dbc.Col([
#                 html.B(f'{cmp_profile_name} parameter changes compared to {base_profile_name}'),
#                 dbc.Table.from_dataframe(profile_diff, bordered=True, hover=True,  striped=True),
#             ], width=4)
#         ]),
#         dcc.Markdown(textwrap.dedent(f'''
#         #### Comparison results
#         ''')),
#         dbc.Row([
#             dbc.Col([
#                 html.B(f'{cmp_profile_name} to {base_profile_name} comparison table'),
#                 dbc.Table.from_dataframe(tbl, bordered=True, hover=True)
#             ], width=4)
#         ])
#     ] + intervention_plots(base_df, cmp_df, base_profile_name, cmp_profile_name)


# def intervention_plots(base_df, cmp_df, base_profile_name, cmp_profile_name):
#     columns_to_plot = ['Infected (symptomatic)', 'Hospitalised', 'Critical', 'Deaths']
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

