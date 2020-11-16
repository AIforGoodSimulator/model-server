"""
This file sets up the parameters for model validation
"""

import numpy as np
import pandas as pd
import json
from ai4good.utils import path_utils as pu


# parameters for model validation
class Parameters:
    model = ['CM', 'ABM', 'NM']
    age_categories = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

    def case_cols(self, model:str) -> list:
        if model.upper() == "CM":
            return ['Exposed', 'Infected (symptomatic)', 'Asymptomatically Infected', 'Recovered', 'Hospitalised', 'Critical', 'Deaths']
        elif model.upper() == "ABM":
            return ['EXPOSED', 'PRESYMPTOMATIC', 'SYMPTOMATIC', 'MILD', 'SEVERE', 'ASYMPTOMATIC1', 'ASYMPTOMATIC2', 'RECOVERED', 'DECEASED']
        elif model.upper() == "NM":
            return ['Exposed', 'Susceptible', 'Exposed', 'Infected_Presymptomatic', 'Infected_Symptomatic', 'Infected_Asymptomatic', 'Hospitalized', 'Recovered', 'Fatalities']
