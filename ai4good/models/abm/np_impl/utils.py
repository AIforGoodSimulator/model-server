import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_progress(file):
    df = pd.read_csv(file)
    # df = df.loc[:100, :]

    t = df.loc[:, 'DAY']

    plt.plot(t, df.loc[:, 'SUSCEPTIBLE'], label='Susceptible')
    plt.plot(t, df.loc[:, 'EXPOSED'], label='Exposed')
    plt.plot(t, df.loc[:, 'PRESYMPTOMATIC'], label='Presymptomatic')
    plt.plot(t, df.loc[:, 'SYMPTOMATIC'], label='Symptomatic')
    plt.plot(t, df.loc[:, 'MILD'], label='Mild')
    plt.plot(t, df.loc[:, 'SEVERE'], label='Severe')
    plt.plot(t, df.loc[:, 'ASYMPTOMATIC1'], label='Asymptomatic1')
    plt.plot(t, df.loc[:, 'ASYMPTOMATIC2'], label='Asymptomatic2')
    plt.plot(t, df.loc[:, 'RECOVERED'], label='Recovered')
    plt.plot(t, df.loc[:, 'HOSPITALIZED'], label='Hospitalized')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    f_name = sys.argv[1]
    plot_progress(f_name)
