# Here is a set of parameters that will be updated when the latest medical studies come in and all models will be taking parameters from this config file
covid_specific_parameters = {
    "R0_low": 3,
    "R0_low_cv": 0.5,
    "R0_medium": 4,
    "R0_medium_cv": 0.5,
    "R0_high": 5,
    "R0_high_cv": 0.5,
    "Latent_period": 4,
    "Latent_period_cv": 0.5,
    "Infectious_period": 5,
    "Infectious_period_cv": 0.5,
    "Hosp_period": 8,
    "Hosp_period_cv": 0.5,
    "Death_period": 2,
    "Death_period_cv": 0.5,
    "Death_period_withICU": 10,
    "Death_period_withICU_cv": 0.5,
    "Death_prob_withICU": 0.75,
    "Death_prob_withICU_cv": 0.5,
    "Infectiousness_asymptomatic": 0.5,
    "p_symptomatic" : [0.4, 0.25, 0.37, 0.42, 0.51, 0.59, 0.72, 0.76],  # for 8 age groups
    "p_hosp_given_symptomatic": [0.0076, 0.0081, 0.0099, 0.0185, 0.0543, 0.1505, 0.3329, 0.6176],  # for 8 age groups
    "p_critical_given_hospitalised": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # for 8 age groups

}