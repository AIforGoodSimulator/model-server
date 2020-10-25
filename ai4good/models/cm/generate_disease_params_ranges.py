# run this script if the underlying covid19 disease parameters have been modified
from ai4good.params.disease_params import covid_specific_parameters
import ai4good.utils.path_utils as pu
import pandas as pd
import numpy as np


if __name__=="__main__":
    generated_params = {}
    np.random.seed(42)
    generated_params['R0'] = np.random.normal(covid_specific_parameters["R0_medium"], 1, 1000)
    generated_params['LatentPeriod'] = np.random.normal(covid_specific_parameters["Latent_period"], 1, 1000)
    generated_params['RemovalPeriod'] = np.random.normal(covid_specific_parameters["Infectious_period"], 1, 1000)
    generated_params['HospPeriod'] = np.random.normal(covid_specific_parameters["Hosp_period"], 1, 1000)
    generated_params['DeathICUPeriod'] = np.random.normal(covid_specific_parameters["Death_period_withICU"], 1, 1000)
    generated_params['DeathNoICUPeriod'] = np.random.normal(covid_specific_parameters["Death_period"], 1, 1000)
    generated_params_df = pd.DataFrame(generated_params)
    generated_params_df[generated_params_df<0] = np.nan
    generated_params_df = generated_params_df.dropna()
    generated_params_df.to_csv(pu.params_path('generated_params.csv'))
