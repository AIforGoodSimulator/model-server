import time

import numpy as np
import pandas as pd

from ai4good.models.cm.simulator import AGE_SEP
DIGIT_SEP = ' to '  # em dash to separate from minus sign

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.1f s' % (f.__name__, (time2-time1)))
        return ret
    return wrap

def load_report(mr, params) -> pd.DataFrame:
    return normalize_report(mr.get('report'), params)


def normalize_report(df, params):
    df = df.copy()
    df.R0 = df.R0.apply(lambda x: round(complex(x).real, 1))
    df_temp = df.drop(['Time', 'R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'],
                      axis=1)
    df_temp = df_temp * params.population
    df.update(df_temp)
    return df

@timing
def prevalence_all_table(df):
    # calculate Peak Day IQR and Peak Number IQR for each of the 'incident' variables to table
    df = df.filter(regex='^Time$|^R0$|^latentRate$|^removalRate$|^hospRate$|^deathRateICU$|^deathRateNoIcu$|^Infected \(symptomatic\)$|^Hospitalised$|^Critical$|^Change in Deaths$')
    groupby_columns = ['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu']
    grouped = df.groupby(groupby_columns)
    indices_to_drop = groupby_columns + ['Time']
    peak_days = get_quantile_report(grouped.apply(lambda x: x.set_index('Time').idxmax()), indices_to_drop)
    peak_numbers = get_quantile_report(grouped.max(), indices_to_drop)

    resultdf = pd.DataFrame.from_dict({'Peak Day IQR': peak_days, 'Peak Number IQR': peak_numbers})
    resultdf.index.name = 'Outcome'

    table_columns = {'Infected (symptomatic)': 'Prevalence of Symptomatic Cases',
                     'Hospitalised': 'Hospitalisation Demand',
                     'Critical': 'Critical Care Demand', 'Change in Deaths': 'Prevalence of Deaths'}

    return resultdf.reindex(index=table_columns.keys()).rename(index=table_columns).reset_index()

def get_quantile_report(x, indices_to_drop):
    return x.stack().groupby(level=-1).quantile([.25, .75])\
        .apply(round).astype(int).astype(str).groupby(level=0).apply(lambda x: DIGIT_SEP.join(x.values)).drop(index = indices_to_drop, errors='ignore')

@timing
def prevalence_age_table(df):
    # calculate age specific Peak Day IQR and Peak Number IQR for each of the 'prevalent' variables to contruct table
    df = df.filter(regex='^Time$|^R0$|^latentRate$|^removalRate$|^hospRate$|^deathRateICU$|^deathRateNoIcu$|^Infected \(symptomatic\)|^Hospitalised|^Critical')
    groupby_columns = ['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu']
    grouped = df.groupby(groupby_columns)
    indices_to_drop = groupby_columns + ['Time']
    peak_days = get_quantile_report(grouped.apply(lambda x: x.set_index('Time').idxmax()), indices_to_drop)
    peak_numbers = get_quantile_report(grouped.max(), indices_to_drop)

    resultdf = pd.DataFrame.from_dict({'Peak Day, IQR': peak_days, 'Peak Number, IQR': peak_numbers})

    arrays = [np.array(['Incident Cases']*9 + ['Hospital Demand']*9 + ['Critical Demand']*9),
              np.array(
                  ['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years',
                   '60-69 years', '70+ years']*3)]
    sorted_index = resultdf.sort_index().index.values
    my_comp_order = ['Infected (symptomatic)', 'Hospitalised', 'Critical']
    my_sorted_index = sum([list(filter(lambda column: comp in column, sorted_index)) for comp in my_comp_order], [])
    sortedresultdf = resultdf.reindex(index=my_sorted_index)
    sortedresultdf.index = pd.MultiIndex.from_arrays(arrays)
    return sortedresultdf

@timing
def cumulative_all_table(df, population, camp_params):
    # now we try to calculate the total count
    # cases: (N-exposed)*0.5 since the asymptomatic rate is 0.5
    # hopistal days: cumulative count of hospitalisation bucket
    # critical days: cumulative count of critical days
    # deaths: we already have that from the frame

    df = df.filter(regex='^Time$|^R0$|^latentRate$|^removalRate$|^hospRate$|^deathRateICU$|^deathRateNoIcu$|Susceptible'+AGE_SEP+'|^Deaths$|^Hospitalised$|^Critical$|^Deaths$')
    groups = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])
    groups_tails = groups.apply(lambda x: x.set_index('Time').tail(1))

    susceptible = groups_tails.filter(like='Susceptible'+AGE_SEP).rename(columns=lambda x: x.split(AGE_SEP)[1])[camp_params['Age']]
    susceptible = ((population * camp_params['Population_structure'].values / 100 - susceptible) * camp_params['p_symptomatic'].values).sum(axis=1)
    susceptible.index = susceptible.index.droplevel('Time')

    deaths = groups_tails['Deaths']
    deaths.index = deaths.index.droplevel('Time')

    cumulative = {
        'Susceptible': susceptible,
        'Hospitalised': groups['Hospitalised'].sum(),
        'Critical': groups['Critical'].sum(),
        'Deaths': deaths
    }
    cumulative_all = pd.DataFrame(cumulative)
    cumulative_count = cumulative_all.quantile([.25, .75]).apply(round).astype(int).astype(str).apply(lambda x: DIGIT_SEP.join(x.values), axis=0).values
    data = {'Totals': ['Symptomatic Cases', 'Hospital Person-Days', 'Critical Person-days', 'Deaths'],
            'Counts': cumulative_count}
    return pd.DataFrame.from_dict(data)

@timing
def cumulative_age_table(df, camp_params):
    # need to have an age break down for this as well
    # 1 month 3 month and 6 month breakdown
    arrays = [np.array(
        ['Symptomatic Cases'] * 9 + ['Hospital Person-Days'] * 9 + ['Critical Person-days'] * 9 + ['Deaths'] * 9),
        np.array(
            ['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years',
             '60-69 years', '70+ years'] * 4)]
    params_select = ['Susceptible:', 'Deaths']
    params_accu = ['Hospitalised', 'Critical']
    columns_to_acc, columns_to_select, multipliers = collect_columns(df.columns, params_accu, params_select, camp_params)

    first_month_diff = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
        columns_to_select + ['Time']].apply(find_first_month_diff)
    third_month_diff = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
        columns_to_select + ['Time']].apply(find_third_month_diff)
    sixth_month_diff = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
        columns_to_select + ['Time']].apply(find_sixth_month_diff)
    first_month_select = first_month_diff[columns_to_select].mul(multipliers).quantile([.25, .75])
    three_month_select = third_month_diff[columns_to_select].mul(multipliers).quantile([.25, .75])
    six_month_select = sixth_month_diff[columns_to_select].mul(multipliers).quantile([.25, .75])

    first_month_select['Susceptible'] = first_month_select.filter(like='Susceptible:').sum(axis=1)
    three_month_select['Susceptible'] = three_month_select.filter(like='Susceptible:').sum(axis=1)
    six_month_select['Susceptible'] = six_month_select.filter(like='Susceptible:').sum(axis=1)

    one_month_cumsum = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
        columns_to_acc + ['Time']].apply(find_one_month)
    three_month_cumsum = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
        columns_to_acc + ['Time']].apply(find_three_months)
    six_month_cumsum = df.groupby(['R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'])[
        columns_to_acc + ['Time']].apply(find_six_months)
    first_month_accu = one_month_cumsum[columns_to_acc].quantile([.25, .75])
    three_month_accu = three_month_cumsum[columns_to_acc].quantile([.25, .75])
    six_month_accu = six_month_cumsum[columns_to_acc].quantile([.25, .75])

    first_month = pd.concat([first_month_select, first_month_accu], axis=1)
    third_month = pd.concat([three_month_select, three_month_accu], axis=1)
    sixth_month = pd.concat([six_month_select, six_month_accu], axis=1)

    sorted_columns = first_month.columns.sort_values()
    my_comp_order = ['Susceptible', 'Hospitalised', 'Critical', 'Deaths']
    my_sorted_columns = sum([list(filter(lambda column: comp in column, sorted_columns)) for comp in my_comp_order], [])

    first_month_count = first_month[my_sorted_columns]\
        .apply(round).astype(int).astype(str) \
        .apply(lambda x: DIGIT_SEP.join(x.values), axis=0).values
    three_month_count = third_month[my_sorted_columns]\
        .apply(round).astype(int).astype(str) \
        .apply(lambda x: DIGIT_SEP.join(x.values), axis=0).values
    six_month_count = sixth_month[my_sorted_columns]\
        .apply(round).astype(int).astype(str) \
        .apply(lambda x: DIGIT_SEP.join(x.values), axis=0).values

    d = {'First month': first_month_count, 'First three months': three_month_count,
         'First six months': six_month_count}
    return pd.DataFrame(data=d, index=arrays)


def collect_columns(columns, params_accu, params_select, camp_params):
    columns_to_select = list(filter(lambda column: any(column.startswith(s) for s in params_select), columns))
    columns_to_acc = list(filter(lambda column: any(column.startswith(s) for s in params_accu), columns))
    multipliers = list(
        map(lambda column: -camp_params[camp_params['Age'].apply(lambda x: x in column)]['p_symptomatic'].values[0] if 'Susceptible:' in column else 1,
            columns_to_select))
    return columns_to_acc, columns_to_select, multipliers

def diff_table(baseline, intervention, N):
    t1 = effectiveness_cum_table(baseline, intervention, N)
    t2 = effectiveness_peak_table(baseline, intervention)

    r1 = [
        'Symptomatic Cases',
        t1.loc['Symptomatic Cases']['Reduction'],
        t2.loc['Prevalence of Symptomatic Cases']['Delay in Peak Day'],
        t2.loc['Prevalence of Symptomatic Cases']['Reduction in Peak Number']
    ]
    r2 = [
        'Hospital Person-Days',
        t1.loc['Hospital Person-Days']['Reduction'],
        t2.loc['Hospitalisation Demand']['Delay in Peak Day'],
        t2.loc['Hospitalisation Demand']['Reduction in Peak Number']
    ]
    r3 = [
        'Critical Person-days',
        t1.loc['Critical Person-days']['Reduction'],
        t2.loc['Critical Care Demand']['Delay in Peak Day'],
        t2.loc['Critical Care Demand']['Reduction in Peak Number']
    ]
    r4 = [
        'Deaths',
        t1.loc['Deaths']['Reduction'],
        t2.loc['Prevalence of Deaths']['Delay in Peak Day'],
        t2.loc['Prevalence of Deaths']['Reduction in Peak Number']
    ]
    df = pd.DataFrame([r1, r2, r3, r4],
                      columns=['Outcome', 'Overall reduction', 'Delay in Peak Day', 'Reduction in Peak Number'])
    return df


def effectiveness_cum_table(baseline, intervention, N):
    table_params = ['Symptomatic Cases', 'Hospital Person-Days', 'Critical Person-days', 'Deaths']
    cum_table_baseline = cumulative_all_table(baseline, N)
    # print("CUM: "+str(cum_table_baseline.loc[:, 'Counts']))
    baseline_numbers = cum_table_baseline.loc[:, 'Counts'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)])

    baseline_numbers_separate = pd.DataFrame(baseline_numbers.tolist(), columns=['25%', '75%'])
    comparisonTable = {}

    cumTable = cumulative_all_table(intervention, N)
    # print("Counts: \n"+str(cumTable.loc[:, 'Counts']))

    intervention_numbers = pd.DataFrame(
        cumTable.loc[:, 'Counts'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(),
        columns=['25%', '75%'])
    differencePercentage = (baseline_numbers_separate - intervention_numbers) / baseline_numbers_separate * 100
    prettyOutput = []
    for _, row in differencePercentage.round(0).astype(int).iterrows():
        output1, output2 = row['25%'], row['75%']
        prettyOutput.append(format_diff_row(output1, output2))

    comparisonTable['Reduction'] = prettyOutput
    comparisonTable['Total'] = table_params
    return pd.DataFrame.from_dict(comparisonTable).set_index('Total')


def format_diff_row(o1, o2, unit='%'):
    if o1 == o2:
        return f'{o1} {unit}'
    elif o2 > o1:
        return f'{o1} to {o2} {unit}'
    else:
        return f'{o2} to {o1} {unit}'


def effectiveness_peak_table(baseline, intervention):
    # the calcuation here is a little bit hand wavy and flimsy, the correct way of implementing should be to compare each intervention
    # with the baseline with the same set up parameters and then in that range pick 25% to 75% data or else it is not correct.
    interventionPeak_baseline = prevalence_all_table(baseline)
    table_columns = interventionPeak_baseline.Outcome.tolist()
    peakDay_baseline = pd.DataFrame(
        interventionPeak_baseline.loc[:, 'Peak Day IQR'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(),
        columns=['25%', '75%'])
    peakNumber_baseline = pd.DataFrame(
        interventionPeak_baseline.loc[:, 'Peak Number IQR'].apply(
            lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(),
        columns=['25%', '75%'])

    comparisonSubdict = {}
    interventionPeak = prevalence_all_table(intervention)
    peakDay = pd.DataFrame(
        interventionPeak.loc[:, 'Peak Day IQR'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(),
        columns=['25%', '75%'])
    peakNumber = pd.DataFrame(
        interventionPeak.loc[:, 'Peak Number IQR'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(),
        columns=['25%', '75%'])
    differenceDay = (peakDay - peakDay_baseline)

    peakNumber_baseline = peakNumber_baseline + 0.01  # Shift to avoid div/0
    peakNumber = peakNumber + 0.01

    differenceNumberPercentage = (peakNumber_baseline - peakNumber) / peakNumber_baseline * 100
    # differenceNumberPercentage = differenceNumberPercentage.replace([np.inf, -np.inf], 100.0)
    prettyOutputDay = []
    prettyOutputNumber = []
    for _, row in differenceDay.round(0).astype(int).iterrows():
        output1, output2 = row['25%'], row['75%']
        prettyOutputDay.append(format_diff_row(output1, output2, 'days'))

    for _, row in differenceNumberPercentage.round(0).astype(int).iterrows():
        output1, output2 = row['25%'], row['75%']
        prettyOutputNumber.append(format_diff_row(output1, output2))

    comparisonSubdict['Delay in Peak Day'] = prettyOutputDay
    comparisonSubdict['Reduction in Peak Number'] = prettyOutputNumber

    comparisondf = pd.DataFrame(comparisonSubdict).set_index(pd.Index(table_columns), 'States')
    return comparisondf


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
