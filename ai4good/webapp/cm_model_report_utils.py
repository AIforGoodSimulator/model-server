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
        peak_day.append(f'{iqr_table[param][0][0]}{DIGIT_SEP}{iqr_table[param][0][1]}')
        peak_number.append(f'{iqr_table[param][1][0]}{DIGIT_SEP}{iqr_table[param][1][1]}')
    data = {'Outcome': outcome, 'Peak Day IQR': peak_day, 'Peak Number IQR': peak_number}
    return pd.DataFrame.from_dict(data)

@timing
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
    arrays = [np.array(['Incident Cases']*9 + ['Hospital Demand']*9 + ['Critical Demand']*9),
              np.array(
                  ['all ages', '<9 years', '10-19 years', '20-29 years', '30-39 years', '40-49 years', '50-59 years',
                   '60-69 years', '70+ years']*3)]
    peak_day = np.empty(27, dtype="S10")
    peak_number = np.empty(27, dtype="S10")
    for key, item in iqr_table_age.items():
        if key == 'Infected (symptomatic)':
            peak_day[0] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
            peak_number[0] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif key == 'Hospitalised':
            peak_day[9] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
            peak_number[9] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif key == 'Critical':
            peak_day[18] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
            peak_number[18] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '0-9' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[1] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[1] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[10] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[10] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[19] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[19] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '10-19' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[2] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[2] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[11] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[11] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[20] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[20] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '20-29' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[3] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[3] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[12] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[12] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[21] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[21] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '30-39' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[4] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[4] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[13] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[13] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[22] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[22] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '40-49' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[5] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[5] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[14] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[14] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[23] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[23] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '50-59' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[6] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[6] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[15] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[15] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[24] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[24] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '60-69' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[7] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[7] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[16] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[16] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[25] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[25] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
        elif '70+' in key:
            if key.startswith('Infected (symptomatic)'):
                peak_day[8] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[8] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Hospitalised'):
                peak_day[17] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[17] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
            elif key.startswith('Critical'):
                peak_day[26] = f'{iqr_table_age[key][0][0]}{DIGIT_SEP}{iqr_table_age[key][0][1]}'
                peak_number[26] = f'{iqr_table_age[key][1][0]}{DIGIT_SEP}{iqr_table_age[key][1][1]}'
    d = {'Peak Day, IQR': peak_day.astype(str), 'Peak Number, IQR': peak_number.astype(str)}
    return pd.DataFrame(data=d, index=arrays)

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
