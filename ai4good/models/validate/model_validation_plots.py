import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

# x is a list of days in data set starting at 1
# y_baseline is a pandas dataframe of the model output for baseline
# y_pred     is a pandas dataframe of the model output for baseline

# Generates the y data for the graph using the collumn name and df
def generate_y_data(model_output_df, col_age): 
    y_data = model_output_df[col_age]
    return y_data

# Generates a string of the column title we want from the df
def generate_col_age(model, col, age):
    if model.upper() == "CM":
        col_age = f"{col}: {age}"
    elif model.upper() == "ABM":
        col_age = f"{col}_AGE{age}"
    elif model.upper() == "NM":
        col_age = f"{col}"
    return col_age

def generate_model_age_df(model, category, age, df_model, time_col, n_simul, population):
    # Process case columns
    cols = [generate_col_age(model, category, age), time_col]
    df_model_age_simul = df_model[cols]
    # Calculate averages for all simulations
    df_model_age_sum = df_model_age_simul.groupby([time_col]).sum() * population
    df_model_age = df_model_age_sum / n_simul   
    return df_model_age 

def gen_traces_to_show(traces, index):
        traces_to_show = traces
        actual_index = 2*index # as two plots per age category. first index must be 0
        traces_to_show[actual_index]     = True
        traces_to_show[(actual_index+1)] = True
        return traces_to_show

# Generates the drop down list for selecting age category graphs
def generate_drop_down_list(traces_to_show_all_false, age_categories):
    buttons_list = []
    traces_to_show_all_true = []
    for i in range (0,len(traces_to_show_all_false)):
        traces_to_show_all_true.append(not traces_to_show_all_false[i])

    buttons_list.append(
        dict(label = "All",
        method = "update",
        args = [{"visible": traces_to_show_all_true},
            {"showlegend": True}])
    )

    # adds drop down option for each age category
    for i in range(len(age_categories)):
        traces_to_show_all_false_copy =traces_to_show_all_false
        traces_to_show = gen_traces_to_show(traces_to_show_all_false_copy, i)
        buttons_list.append(
            dict(label = age_categories[i],
            method = "update",
            args = [{"visible": traces_to_show},
                {"showlegend": True}])
        )
    return buttons_list

def plot_series(model, x, df_baseline, df_model, category, age_categories, time_col, baseline_n_simul, n_simul, population):
    # plotting the series
    fig = go.Figure()
    traces_to_show = []
    for age in age_categories:

        df_baseline_age = generate_model_age_df(model, category, age, df_baseline, time_col,  baseline_n_simul, population)
        df_model_age    = generate_model_age_df(model, category, age, df_model, time_col, n_simul, population)

        col_age = generate_col_age(model, category, age) # generated once for each age group for efficiency
        
        fig.add_trace(go.Scatter(x=x, y=generate_y_data(df_baseline_age, col_age),
            mode = "lines+markers",
            name = f"Baseline {age}"))

        fig.add_trace(go.Scatter(x=x, y=generate_y_data(df_model_age, col_age),
            mode = "lines+markers",
            name = f"Model {age}"))

        traces_to_show.append(False) # Probably a cleaner way of doing this but need an item in list for every trace with default value of True
        traces_to_show.append(False)

    # Add title and axis labels
    fig.update_layout(
        title=f"Comparison of {category} over different age groups",
        xaxis_title="Day",
        yaxis_title="No. of cases",
    )

    return fig

def plot_histogram(model, x, df_baseline, df_model, category, age_categories, time_col, baseline_n_simul, n_simul, population):

    fig = go.Figure()
    traces_to_show = []
    
    # determine histogram bin size
    df_max = 0
    for age in age_categories:
        col_age = generate_col_age(model, category, age)

        df_baseline_age = generate_model_age_df(model, category, age, df_baseline, time_col, baseline_n_simul, population)
        df_model_age    = generate_model_age_df(model, category, age, df_model, time_col,  n_simul, population)
        
        df_max = max(df_max, np.max(generate_y_data(df_baseline_age, col_age)), np.max(generate_y_data(df_model_age, col_age)))

    hist_bin_min = 0
    hist_bin_max = df_max
    hist_bin_size = (hist_bin_max - hist_bin_min)/20

    for age in age_categories:
        col_age = generate_col_age(model, category, age)

        df_baseline_age = generate_model_age_df(model, category, age, df_baseline, time_col, baseline_n_simul, population)
        df_model_age    = generate_model_age_df(model, category, age, df_model, time_col, n_simul, population)
        
        # Add histogram for baseline
        fig.add_histogram(x=generate_y_data(df_baseline_age, col_age), 
                          xbins=dict(start=hist_bin_min, end=hist_bin_max, size=hist_bin_size), 
                          autobinx=False, 
                          name = f"Baseline {age}"
        )
        # Add histogram for model
        fig.add_histogram(x=generate_y_data(df_model_age, col_age), 
                          xbins=dict(start=hist_bin_min, end=hist_bin_max, size=hist_bin_size), 
                          autobinx=False, 
                          name = f"Model {age}"
        )
        
        traces_to_show.append(False) # Probably a cleaner way of doing this but need an item in list for every trace with default value of True
        traces_to_show.append(False)

    fig.update_layout(
        # Add title and axis labels
        title=f"Histogram of {category} over different age groups",
        xaxis_title="No. of cases",
        yaxis_title="Frequency",
        # Overlay both histograms
        barmode="overlay",
    )

    #Reduce Opacity to see both histograms
    fig.update_traces(opacity=0.75)

    return fig

# Plots a graph of kde distribution
def plot_distribution(model, x, df_baseline, df_model, category, age_categories, time_col, baseline_n_simul, n_simul, population):

    traces_to_show = []
    data_to_plot = []
    group_labels = []
    for age in age_categories:

        df_baseline_age = generate_model_age_df(model, category, age, df_baseline, time_col, baseline_n_simul, population)
        df_model_age    = generate_model_age_df(model, category, age, df_model, time_col, n_simul, population)

        col_age = generate_col_age(model, category, age) # generated once for each age group for efficiency

        data_to_plot.append(generate_y_data(df_baseline_age, col_age))
        data_to_plot.append(generate_y_data(df_model_age, col_age))

        group_labels.append(f"Baseline {age}")
        group_labels.append(f"Model {age}")
        
        traces_to_show.append(False) # Probably a cleaner way of doing this but need an item in list for every trace with default value of True
        traces_to_show.append(False)
    
    fig = ff.create_distplot(data_to_plot, group_labels, show_hist=False)
    # Add title and axis labels
    fig.update_layout(
        title=f"Distribution of {category} over different age groups",
        xaxis_title="No. of cases",
        yaxis_title="Output",
    )
    return fig

def plot_autocorrelation(model, df_baseline_age, df_model_age, col, age_categories, category, shifts=31):

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

    autocorrelations, p_autocorrelations = [], []
    for age in age_categories:
        col_age = generate_col_age(model, category, age)
        y = df_baseline_age[col_age]
        pred = df_model_age[col_age]
        input = y - pred

        autocorrelations.append([np.linspace(0, shifts-1, shifts), get_autocorrelation(pred.to_numpy().copy(), shifts=shifts), [age for __ in range(shifts)]])
        p_autocorrelations.append([np.linspace(0, shifts-1, shifts), get_partial_autocorrelation(pred.to_numpy(), shifts=shifts), [age for __ in range(shifts)]])

    autocorrelations, p_autocorrelations = np.asarray(autocorrelations), np.asarray(p_autocorrelations)

    ac_df = pd.DataFrame(data={"shift": autocorrelations[:,0].flatten(), "ac": autocorrelations[:,1].flatten(), "colour": autocorrelations[:,2].flatten()})
    pac_df = pd.DataFrame(data={"shift": p_autocorrelations[:,0].flatten(), "pac": p_autocorrelations[:,1].flatten(), "colour": p_autocorrelations[:,2].flatten()})

    ac_fig = px.line(ac_df, x="shift", y="ac", color="colour", title=f"Autocorrelation of {category} over different age groups")
    pac_fig = px.line(pac_df, x="shift", y="pac", color="colour", title=f"Partial Autocorrelation of {category} over different age groups")

    return ac_fig, pac_fig


def model_validation_plots(population:int, model:str, age_categories:list, case_cols:list, df_baseline:pd.DataFrame, df_model:pd.DataFrame):

    # Process case columns
    if model.upper() == "CM":
        time_col = "Time"
    elif model.upper() == "ABM":
        time_col = "DAY"
    elif model.upper() == "NM":
        time_col = "Time"
    cols_overall = [time_col] + case_cols

    # Process baseline csv
    df = df_baseline[time_col]
    baseline_n_days = df.nunique()  # Count distinct observations over requested axis.
    baseline_n_rows = df.shape[0]
    # num of simuls
    baseline_n_simul = df[df == 0].count()

    # Get df for population
    # Use this as the benchmark for the age group
    df_baseline_all_simul = df_baseline[cols_overall]
    df_baseline_all_sum = df_baseline_all_simul.groupby([time_col]).sum() * population
    df_baseline_all = df_baseline_all_sum / baseline_n_simul

    # Process Model Output and compare with baseline;
    df = df_model[time_col]
    n_days = df.nunique()
    n_rows = df.shape[0]
    # num of simuls
    n_simul = df[df == 0].count()

    x = [i+1 for i in range(n_days)]
    graph_divs = []
    for col in case_cols:
        graph_divs.append(html.Div(dcc.Graph(figure=plot_series(model, x, df_baseline, df_model, col, age_categories, time_col, baseline_n_simul, n_simul, population))))
        graph_divs.append(html.Div(dcc.Graph(figure=plot_histogram(model, x, df_baseline, df_model, col, age_categories, time_col, baseline_n_simul, n_simul, population))))
        graph_divs.append(html.Div(dcc.Graph(figure=plot_distribution(model, x, df_baseline, df_model, col, age_categories, time_col, baseline_n_simul, n_simul, population))))
        ac_fig, pac_fig = plot_autocorrelation(model, df_baseline, df_model, col, age_categories, col)
        graph_divs.append(html.Div(dcc.Graph(figure=ac_fig)))
        graph_divs.append(html.Div(dcc.Graph(figure=pac_fig)))

    return (graph_divs)
