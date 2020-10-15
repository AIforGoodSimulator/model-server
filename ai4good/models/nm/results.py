import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


sns.set_style("darkgrid")


files = [file for file in os.listdir('results') if '.csv' in file]

networks = []
for fname in files:
    df = pd.read_csv('results/' + fname, index_col=0)
    df['Infected'] = df['Infected_Presymptomatic']\
                     + df['Infected_Symptomatic']\
                     + df['Infected_Asymptomatic']
    df['delta_Fatalities'] = df['Fatalities'].diff()
    df.fillna(value=0, inplace=True)

    networks.append(df)

infected_nums = pd.DataFrame()
hospitalized_nums = pd.DataFrame()
fatalities_nums = pd.DataFrame()
fatalities_rates = pd.DataFrame()
summary_df = pd.DataFrame(columns=['model',
                                   'highest_infections_day',
                                   'highest_infections',
                                   'highest_hospitalizations_day',
                                   'highest_hospitalizations',
                                   'hospital_person_days',
                                   'highest_deaths_day',
                                   'highest_deaths',
                                   'total_deaths'])
for i in range(len(files)):
    infected_nums[files[i]] = networks[i]['Infected'].values
    hospitalized_nums[files[i]] = networks[i]['Hospitalized'].values
    fatalities_nums[files[i]] = networks[i]['Fatalities'].values
    fatalities_rates[files[i]] = networks[i]['delta_Fatalities'].values

    summary_df.loc[i] = [files[i],
                         np.argmax(networks[i]['Infected'].values),
                         max(networks[i]['Infected']),
                         np.argmax(networks[i]['Hospitalized'].values),
                         max(networks[i]['Hospitalized']),
                         sum(networks[i]['Hospitalized'].values),
                         np.argmax(networks[i]['delta_Fatalities'].values),
                         max(networks[i]['delta_Fatalities']),
                         sum(networks[i]['delta_Fatalities'])]

base_graphs = []
intervention_graphs = []
fq4_graphs = []
fq8_graphs = []
fq16_graphs = []
fq4int150_graphs = []
fq8int150_graphs = []
fq16int150_graphs = []
fq4int60_graphs = []
fq8int60_graphs = []
fq16int60_graphs = []
for col in infected_nums.columns:
    if 'BaseSympModel' in col:
        base_graphs.append(col)
    elif 'InterventionsBaseModel' in col:
        intervention_graphs.append(col)
    elif 'MultFoodQueue1' in col:
        fq4_graphs.append(col)
    elif 'MultFoodQueue2' in col:
        fq8_graphs.append(col)
    elif 'MultFoodQueue4' in col:
        fq16_graphs.append(col)
    elif 'InterventionsMultFQ1' in col and 'Qtime=0-150' in col:
        fq4int150_graphs.append(col)
    elif 'InterventionsMultFQ2' in col and 'Qtime=0-150' in col:
        fq8int150_graphs.append(col)
    elif 'InterventionsMultFQ4' in col and 'Qtime=0-150' in col:
        fq16int150_graphs.append(col)
    elif 'InterventionsMultFQ1' in col and 'Qtime=3-63' in col:
        fq4int60_graphs.append(col)
    elif 'InterventionsMultFQ2' in col and 'Qtime=3-63' in col:
        fq8int60_graphs.append(col)
    elif 'InterventionsMultFQ4' in col and 'Qtime=3-63' in col:
        fq16int60_graphs.append(col)

infected_nums['Avg Base Network'] = infected_nums[base_graphs].mean(axis=1)
infected_nums['Avg Base Network with Quarantine'] = infected_nums[intervention_graphs].mean(axis=1)
infected_nums['Avg Base Network with 4 Food Queues'] = infected_nums[fq4_graphs].mean(axis=1)
infected_nums['Avg Base Network with 8 Food Queues'] = infected_nums[fq8_graphs].mean(axis=1)
infected_nums['Avg Base Network with 16 Food Queues'] = infected_nums[fq16_graphs].mean(axis=1)
infected_nums['Combined (4 Food Queues)'] = infected_nums[fq4int150_graphs].mean(axis=1)
infected_nums['Combined (8 Food Queues)'] = infected_nums[fq8int150_graphs].mean(axis=1)
infected_nums['Combined (16 Food Queues)'] = infected_nums[fq16int150_graphs].mean(axis=1)
infected_nums['Avg Base Network with 4 Food Queues and 2 months of Quarantine'] = infected_nums[fq4int60_graphs].mean(axis=1)
infected_nums['Avg Base Network with 8 Food Queues and 2 months of Quarantine'] = infected_nums[fq8int60_graphs].mean(axis=1)
infected_nums['Avg Base Network with 16 Food Queues and 2 months of Quarantine'] = infected_nums[fq16int60_graphs].mean(axis=1)


hospitalized_nums['Avg Base Network'] = hospitalized_nums[base_graphs].mean(axis=1)
hospitalized_nums['Avg Base Network with Quarantine'] = hospitalized_nums[intervention_graphs].mean(axis=1)
hospitalized_nums['Avg Base Network with 4 Food Queues'] = hospitalized_nums[fq4_graphs].mean(axis=1)
hospitalized_nums['Avg Base Network with 8 Food Queues'] = hospitalized_nums[fq8_graphs].mean(axis=1)
hospitalized_nums['Avg Base Network with 16 Food Queues'] = hospitalized_nums[fq16_graphs].mean(axis=1)
hospitalized_nums['Combined (4 Food Queues)'] = hospitalized_nums[fq4int150_graphs].mean(axis=1)
hospitalized_nums['Combined (8 Food Queues)'] = hospitalized_nums[fq8int150_graphs].mean(axis=1)
hospitalized_nums['Combined (16 Food Queues)'] = hospitalized_nums[fq16int150_graphs].mean(axis=1)


fatalities_nums['Avg Base Network'] = fatalities_nums[base_graphs].mean(axis=1)
fatalities_nums['Avg Base Network with Quarantine'] = fatalities_nums[intervention_graphs].mean(axis=1)
fatalities_nums['Avg Base Network with 4 Food Queues'] = fatalities_nums[fq4_graphs].mean(axis=1)
fatalities_nums['Avg Base Network with 8 Food Queues'] = fatalities_nums[fq8_graphs].mean(axis=1)
fatalities_nums['Avg Base Network with 16 Food Queues'] = fatalities_nums[fq16_graphs].mean(axis=1)
fatalities_nums['Combined (4 Food Queues)'] = fatalities_nums[fq4int150_graphs].mean(axis=1)
fatalities_nums['Combined (8 Food Queues)'] = fatalities_nums[fq8int150_graphs].mean(axis=1)
fatalities_nums['Combined (16 Food Queues)'] = fatalities_nums[fq16int150_graphs].mean(axis=1)


fatalities_rates['Avg Base Network'] = fatalities_rates[base_graphs].mean(axis=1)
fatalities_rates['Avg Base Network with Quarantine'] = fatalities_rates[intervention_graphs].mean(axis=1)
fatalities_rates['Avg Base Network with 4 Food Queues'] = fatalities_rates[fq4_graphs].mean(axis=1)
fatalities_rates['Avg Base Network with 8 Food Queues'] = fatalities_rates[fq8_graphs].mean(axis=1)
fatalities_rates['Avg Base Network with 16 Food Queues'] = fatalities_rates[fq16_graphs].mean(axis=1)
fatalities_rates['Combined (4 Food Queues)'] = fatalities_rates[fq4int150_graphs].mean(axis=1)
fatalities_rates['Combined (8 Food Queues)'] = fatalities_rates[fq8int150_graphs].mean(axis=1)
fatalities_rates['Combined (16 Food Queues)'] = fatalities_rates[fq16int150_graphs].mean(axis=1)


to_graph = ['Avg Base Network',
            'Avg Base Network with 4 Food Queues',
            'Avg Base Network with 8 Food Queues',
            'Avg Base Network with 16 Food Queues']

plt.figure()
plt.title('Total Infected')
fig1 = sns.lineplot(data=infected_nums[to_graph])
fig1.figure.savefig('plotted_results/infected_totals.png')

plt.figure()
plt.title('Total Hospitalized per Day')
fig2 = sns.lineplot(data=hospitalized_nums[to_graph])
fig2.figure.savefig('plotted_results/hospitalized_totals.png')

plt.figure()
plt.title('Total Fatalities')
fig3 = sns.lineplot(data=fatalities_nums[to_graph])
fig3.figure.savefig('plotted_results/fatalities_totals.png')

plt.figure()
plt.title('New Fatalities per Day')
fig4 = sns.lineplot(data=fatalities_rates[to_graph])
fig4.figure.savefig('plotted_results/fatalities_delta.png')

summary_df.to_csv('summary.csv', index=False)
