"""
This file sets up the parameters for NM models
"""

import pandas as pd
import json
import hashlib
from ai4good.params.param_store import ParamStore
from seirsplus.utilities import *
from ai4good.models.nm.utils.network_utils import *
from ai4good.models.nm.parameters.camp_params import *


# TODO: BIG TODO!!!! We're using default parameters from the seirsplus model, and hard coding them here
# Hopefully, we will be able to read them from a csv instead of this


class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}, t_steps=2):
        self.ps = ps

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Camp parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # self.camp = camp   TODO: Update for both camps when ready
        self.camp = "Moria"
        self.camp_params = ps.get_camp_params(self.camp)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Disease parameters                    # Do we need this?
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disease_params = ps.get_disease_params()
        parameter_csv = disease_params
        model_params = parameter_csv[parameter_csv['Type'] == 'Model Parameter']
        model_params = model_params.loc[:, ['Name', 'Value']]
        control_data = parameter_csv[parameter_csv['Type'] == 'Control']

        self.better_hygiene = np.float(control_data.Value[control_data.Name == 'Better hygiene'])
        self.shield_decrease = np.float(
            control_data[control_data['Name'] == 'Reduction in contact between groups'].Value)
        self.shield_increase = np.float(control_data[control_data['Name'] == 'Increase in contact within group'].Value)

        self.r_0_list = np.asarray(model_params[model_params['Name'] == 'R0'].Value)

        self.latent_period = np.float(model_params[model_params['Name'] == 'latent period'].Value)
        self.infectious_period = np.float(model_params[model_params['Name'] == 'infectious period'].Value)
        self.hosp_period = np.float(model_params[model_params['Name'] == 'hosp period'].Value)
        self.quarantine_period = np.float(model_params[model_params['Name'] == 'quarantine period'].Value)

        self.death_period = np.float(model_params[model_params['Name'] == 'death period'].Value)
        self.death_period_with_icu = np.float(model_params[model_params['Name'] == 'death period with ICU'].Value)

        self.infectiousness_of_asymptomatic = np.float(
            model_params[model_params['Name'] == 'infectiousness of asymptomatic'].Value)
        self.death_prob_with_icu = np.float(model_params[model_params['Name'] == 'death prob with ICU'].Value)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Model Parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        profile.set_index('Parameter', inplace=True)

        # Sample the population age, and parameter rates
        self.total_population = int(self.camp_params.Total_population.dropna().sum())  # n_pop
        self.sample_pop = stats_utils.sample_population(self.total_population,
                                                        "ai4good/models/nm/data/augmented_population.csv")

        # Tents
        self.number_of_tents = int(profile.loc['number_of_tents', 'Value'])  # n_tents
        self.total_population_in_tents = int(profile.loc['total_population_in_tents', 'Value'])  # pop_tents
        self.number_of_people_in_one_tent = int(profile.loc['number_of_people_in_one_tent', 'Value'])  # pop_per_tent

        # The radius of nearby structures whose inhabitants a person has connections with
        self.neighbor_proximity = int(profile.loc['neighbor_proximity', 'Value'])

        # Edge weight for friendship connections
        self.neighbor_weight = float(profile.loc['neighbor_weight', 'Value'])

        # Edge weight for connections in the food queue
        self.food_weight = float(profile.loc['food_weight', 'Value'])

        # Edge weight for connections within each structure
        self.household_weight = float(profile.loc['household_weight', 'Value'])

        # Others
        self.number_of_ethnic_groups = int(profile.loc['number_of_ethnic_groups', 'Value'])  # n_ethnic_groups

        # Distribution based parameters
        self.latentPeriod_mean, latentPeriod_coeffvar = 3.0, 0.6
        self.presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar = 2.2, 0.5
        self.symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 4.0, 0.4
        self.onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
        self.hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
        self.hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
        self.R0_mean, R0_coeffvar = 2.5, 0.2

        self.sigma = 1 / \
                     gamma_dist(self.latentPeriod_mean, latentPeriod_coeffvar, self.total_population)
        self.lamda = 1 / \
                     gamma_dist(self.presymptomaticPeriod_mean,
                                presymptomaticPeriod_coeffvar, self.total_population)
        self.gamma = 1 / \
                     gamma_dist(self.symptomaticPeriod_mean,
                                symptomaticPeriod_coeffvar, self.total_population)
        self.eta = 1 / gamma_dist(self.onsetToHospitalizationPeriod_mean,
                                  onsetToHospitalizationPeriod_coeffvar, self.total_population)
        self.gamma_H = 1 / gamma_dist(self.hospitalizationToDischargePeriod_mean,
                                      hospitalizationToDischargePeriod_coeffvar,
                                      self.total_population)
        self.mu_H = 1 / gamma_dist(self.hospitalizationToDeathPeriod_mean,
                                   hospitalizationToDeathPeriod_coeffvar, self.total_population)
        self.R0 = gamma_dist(self.R0_mean, R0_coeffvar, self.total_population)

        infectiousPeriod = 1 / self.lamda + 1 / self.gamma
        self.beta = 1 / infectiousPeriod * self.R0
        self.beta_pairwise_mode = 'infected'
        self.delta_pairwise_mode = 'mean'

        # Constant parameters
        self.p_global_interaction = 0.2
        self.init_exposed = int(self.total_population / 100)
        self.init_infected = int(self.total_population / 100)

        self.transmission_mean = self.beta.mean()
        self.progressionToInfectious_mean = self.sigma.mean()
        self.progressionToSymptomatic_mean = self.lamda.mean()
        self.recovery_mean = self.gamma.mean()
        self.hospitalization_mean = self.eta.mean()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Reduced interaction parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.reduction_rate = 0.3
        self.beta_q = self.beta * (self.reduction_rate / self.R0_mean)
        self.q_global_interactions = 0.05
        self.t_steps = t_steps
        self.q_start = 30
        self.m_start = 30
        self.q_end = 60
        self.m_end = 90

    def initialise_age_parameters(self, graph):
        # Age based parameters
        # TODO: Take from camp_params

        ageGroup_pctAsymp = {"0-19": 0.8, "20+": 0.2}
        ageGroup_pctHospitalized = {'0-9': 0.0000,
                                    '10-19': 0.0004,
                                    '20-29': 0.0104,
                                    '30-39': 0.0343,
                                    '40-49': 0.0425,
                                    '50-59': 0.0816,
                                    '60-69': 0.118,
                                    '70-79': 0.166,
                                    '80+': 0.184}

        ageGroup_hospitalFatalityRate = {'0-9': 0.0000,
                                         '10-19': 0.3627,
                                         '20-29': 0.0577,
                                         '30-39': 0.0426,
                                         '40-49': 0.0694,
                                         '50-59': 0.1532,
                                         '60-69': 0.3381,
                                         '70-79': 0.5187,
                                         '80+': 0.7283}
        ageGroup_susceptibility = {"0-19": 0.5, "20+": 1.0}

        self.pct_asymptomatic = get_values_per_node(ageGroup_pctAsymp, graph)
        self.pct_hospitalized = get_values_per_node(
            ageGroup_pctHospitalized, graph)
        self.pct_fatality = get_values_per_node(
            ageGroup_hospitalFatalityRate, graph)
        self.alpha = get_values_per_node(ageGroup_susceptibility, graph)

    def sha1_hash(self) -> str:
        hash_params = [
            self.total_population,
            self.camp,
            # self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash
