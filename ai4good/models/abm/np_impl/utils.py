"""
This file contains small helper utility functions
"""

import sys
import math
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_age_gender(num_ppl):
    """
    Read file containing people's age and gender.

    Parameters
    ----------
        num_ppl : Number of records to return

    Returns
    -------
        out : Numpy array of size (`num_ppl`, 2) containing rows: (age, gender)

    """
    # Data frame. V1 = age, V2 is sex (1 = male?, 0  = female?).

    from ai4good.utils.path_utils import get_am_aug_pop

    age_and_gender = pd.read_csv(get_am_aug_pop())
    age_and_gender = age_and_gender.loc[:, ~age_and_gender.columns.str.contains('^Unnamed')]
    age_and_gender = age_and_gender.values

    if age_and_gender.shape[0] < num_ppl:
        logging.warning("Number of agents are more than data provided in age_and_gender.csv by {}".
                        format(num_ppl - age_and_gender.shape[0]))

    age_and_gender = age_and_gender[np.random.randint(age_and_gender.shape[0], size=num_ppl)]
    return age_and_gender


def get_incubation_period(num_ppl):
    # The time from exposure until symptoms appear (i.e., the incubation period) is
    # drawn from a Weibull distribution with a mean of 6.4 days and a standard deviation of 2.3
    # days (Backer et al. 2020)
    k = (2.3/6.4)**(-1.086)
    l = 6.4 / (math.gamma(1 + 1/k))
    return np.around([random.weibullvariate(l, k) for _ in np.arange(num_ppl)]).astype(np.int32)


def plot_progress(file):
    df = pd.read_csv(file)

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


def plot_sir(file):
    # Group disease states under susceptible, infected and recovered
    df = pd.read_csv(file)

    t = df.loc[:, 'DAY']

    sus = df.loc[:, 'SUSCEPTIBLE']
    inf = df.loc[:, ['EXPOSED', 'PRESYMPTOMATIC', 'SYMPTOMATIC', 'MILD', 'SEVERE', 'ASYMPTOMATIC1', 'ASYMPTOMATIC2']]\
        .sum(axis=1)
    rec = df.loc[:, 'RECOVERED']

    plt.plot(t, sus, label='Susceptible')
    plt.plot(t, inf, label='Infected')
    plt.plot(t, rec, label='Recovered')

    plt.legend()
    plt.show()


def plot_hospitalized(file):
    df = pd.read_csv(file)

    t = df.loc[:, 'DAY']

    plt.plot(t, df.loc[:, 'HOSPITALIZED'], label='Hospitalized')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # If this file is called directly from command line, one can pass the progress file name to get a time series plot
    # of disease states
    f_name = sys.argv[1]  # use like > python utils.py path/to/filename.csv
    plot_progress(f_name)
    plot_sir(f_name)
    plot_hospitalized(f_name)
