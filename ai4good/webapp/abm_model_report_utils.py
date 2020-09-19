import numpy as np
import pandas as pd

DIGIT_SEP = ' to '  # em dash to separate from minus sign


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
        cumulative_count.append(f'{int(round(q25_count))}{DIGIT_SEP}{int(round(q75_count))}')
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
            first_month_count[0] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif key == 'Hospitalised':
            first_month_count[9] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif key == 'Critical':
            first_month_count[18] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif key == 'Deaths':
            first_month_count[27] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '0-9' in key:
            if key.startswith('Susceptible'):
                first_month_count[1] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[10] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[19] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[28] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '10-19' in key:
            if key.startswith('Susceptible'):
                first_month_count[2] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[11] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[20] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[29] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '20-29' in key:
            if key.startswith('Susceptible'):
                first_month_count[3] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[12] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[21] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[30] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '30-39' in key:
            if key.startswith('Susceptible'):
                first_month_count[4] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[13] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[22] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[31] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '40-49' in key:
            if key.startswith('Susceptible'):
                first_month_count[5] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[14] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[23] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[32] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '50-59' in key:
            if key.startswith('Susceptible'):
                first_month_count[6] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[15] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[24] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[33] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '60-69' in key:
            if key.startswith('Susceptible'):
                first_month_count[7] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[16] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[25] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[34] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
        elif '70+' in key:
            if key.startswith('Susceptible'):
                first_month_count[8] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                first_month_count[17] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Critical'):
                first_month_count[26] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                first_month_count[35] = f'{int(round(first_month[key][0.25]))}{DIGIT_SEP}{int(round(first_month[key][0.75]))}'
    three_month_count = np.empty(36, dtype="S15")
    for key, item in third_month.items():
        if key == 'Susceptible':
            three_month_count[0] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif key == 'Hospitalised':
            three_month_count[9] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif key == 'Critical':
            three_month_count[18] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif key == 'Deaths':
            three_month_count[27] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '0-9' in key:
            if key.startswith('Susceptible'):
                three_month_count[1] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[10] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[19] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[28] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '10-19' in key:
            if key.startswith('Susceptible'):
                three_month_count[2] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[11] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[20] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[29] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '20-29' in key:
            if key.startswith('Susceptible'):
                three_month_count[3] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[12] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[21] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[30] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '30-39' in key:
            if key.startswith('Susceptible'):
                three_month_count[4] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[13] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[22] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[31] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '40-49' in key:
            if key.startswith('Susceptible'):
                three_month_count[5] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[14] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[23] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[32] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '50-59' in key:
            if key.startswith('Susceptible'):
                three_month_count[6] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[15] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[24] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[33] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '60-69' in key:
            if key.startswith('Susceptible'):
                three_month_count[7] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[16] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[25] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[34] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
        elif '70+' in key:
            if key.startswith('Susceptible'):
                three_month_count[8] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                three_month_count[17] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Critical'):
                three_month_count[26] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                three_month_count[35] = f'{int(round(third_month[key][0.25]))}{DIGIT_SEP}{int(round(third_month[key][0.75]))}'
    six_month_count = np.empty(36, dtype="S10")
    for key, item in sixth_month.items():
        if key == 'Susceptible':
            six_month_count[0] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif key == 'Hospitalised':
            six_month_count[9] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif key == 'Critical':
            six_month_count[18] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif key == 'Deaths':
            six_month_count[27] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '0-9' in key:
            if key.startswith('Susceptible'):
                six_month_count[1] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[10] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[19] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[28] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '10-19' in key:
            if key.startswith('Susceptible'):
                six_month_count[2] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[11] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[20] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[29] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '20-29' in key:
            if key.startswith('Susceptible'):
                six_month_count[3] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[12] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[21] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[30] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '30-39' in key:
            if key.startswith('Susceptible'):
                six_month_count[4] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[13] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[22] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[31] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '40-49' in key:
            if key.startswith('Susceptible'):
                six_month_count[5] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[14] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[23] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[32] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '50-59' in key:
            if key.startswith('Susceptible'):
                six_month_count[6] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[15] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[24] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[33] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '60-69' in key:
            if key.startswith('Susceptible'):
                six_month_count[7] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[16] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[25] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[34] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
        elif '70+' in key:
            if key.startswith('Susceptible'):
                six_month_count[8] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Hospitalised'):
                six_month_count[17] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Critical'):
                six_month_count[26] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
            elif key.startswith('Deaths'):
                six_month_count[35] = f'{int(round(sixth_month[key][0.25]))}{DIGIT_SEP}{int(round(sixth_month[key][0.75]))}'
    d = {'First month': first_month_count.astype(str), 'First three months': three_month_count.astype(str),
         'First six months': six_month_count.astype(str)}
    return pd.DataFrame(data=d, index=arrays)


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
    #print("CUM: "+str(cum_table_baseline.loc[:, 'Counts']))
    baseline_numbers = cum_table_baseline.loc[:, 'Counts'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)])

    baseline_numbers_separate = pd.DataFrame(baseline_numbers.tolist(), columns=['25%', '75%'])
    comparisonTable = {}

    cumTable = cumulative_all_table(intervention, N)
    #print("Counts: \n"+str(cumTable.loc[:, 'Counts']))

    intervention_numbers = pd.DataFrame(
        cumTable.loc[:, 'Counts'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(), columns=['25%', '75%'])
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
        interventionPeak_baseline.loc[:, 'Peak Number IQR'].apply(lambda x: [int(i) for i in x.split(DIGIT_SEP)]).tolist(),
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

    peakNumber_baseline = peakNumber_baseline + 0.01 # Shift to avoid div/0
    peakNumber = peakNumber + 0.01

    differenceNumberPercentage = (peakNumber_baseline - peakNumber) / peakNumber_baseline * 100
    #differenceNumberPercentage = differenceNumberPercentage.replace([np.inf, -np.inf], 100.0)
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


