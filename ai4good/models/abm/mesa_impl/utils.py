import time
import math
import random
import logging
import numpy as np
import pandas as pd
from numba import njit

from ai4good.utils.path_utils import get_am_aug_pop
from ai4good.models.abm.mesa_impl.common import CAMP_SIZE


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


def log(name="function"):
    # decorator for logging completion time
    # use this decorator like @log(name='func_name')
    def wrapped1(func):
        def wrapped2(self, *args, **kwargs):
            t1 = time.time()
            func(self, *args, **kwargs)
            t2 = time.time()
            logging.info("Completed {} in {} seconds".format(name, t2-t1))
        return wrapped2
    return wrapped1


def plot():
    pass
