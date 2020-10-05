"""
This file sets up the parameters for NM models
"""

import json
import hashlib
from ai4good.params.param_store import ParamStore
from seirsplus.utilities import *
from ai4good.models.nm.utils.network_utils import *
from ai4good.models.nm.utils.stats_utils import *
from scipy.stats import poisson

# TODO: (Partially Done) We're using default parameters from the seirsplus model, and hard coding them here
# Unsure how to map some parameters in CSV

class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}):
        self.ps = ps
        self.profile_name = profile['Profile'].iloc[0]

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Camp parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # self.camp = camp   TODO: Update for both camps when ready
        self.camp = camp
        self.camp_params = ps.get_camp_params(self.camp)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Model Parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        profile.set_index('Parameter', inplace=True)

        # Sample the population age, and parameter rates
        self.total_population = int(self.camp_params.Total_population.dropna().sum())  # n_pop
        self.sample_pop = sample_population(self.total_population,
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

        # Isoboxes
        self.number_of_people_in_isoboxes = int(profile.loc['number_of_people_in_isoboxes', 'Value'])  # pop_isoboxes
        self.number_of_people_in_one_isobox = int(profile.loc['number_of_people_in_one_isobox', 'Value'])  # pop_per_isobox

        self.t_steps = int(profile.loc['number_of_steps', 'Value'])

        # Grid info for isoboxes
        dimensions_of_isoboxes = profile.loc['dimensions_of_isoboxes', 'Value'].split(',')
        self.dims_isoboxes = (int(dimensions_of_isoboxes[0]), int(dimensions_of_isoboxes[1]))

        # Grid info for tents
        dimensions_of_block1 = profile.loc['dimensions_of_block1', 'Value'].split(',')
        self.dims_block1 = (int(dimensions_of_block1[0]), int(dimensions_of_block1[1]))
        dimensions_of_block2 = profile.loc['dimensions_of_block2', 'Value'].split(',')
        self.dims_block2 = (int(dimensions_of_block2[0]), int(dimensions_of_block2[1]))
        dimensions_of_block3 = profile.loc['dimensions_of_block3', 'Value'].split(',')
        self.dims_block3 = (int(dimensions_of_block3[0]), int(dimensions_of_block3[1]))

        self.n_isoboxes = self.dims_isoboxes[0] * self.dims_isoboxes[1]

        # Define the maximum population per structures (tents and isoboxes) drawn from a poisson distribution

        max_pop_per_struct_isoboxes = list(
            poisson.rvs(mu=self.number_of_people_in_one_isobox, size=self.n_isoboxes))
        max_pop_per_struct_block1 = list(poisson.rvs(
            mu=self.number_of_people_in_one_tent, size=self.dims_block1[0] * self.dims_block1[1]))
        max_pop_per_struct_block2 = list(poisson.rvs(
            mu=self.number_of_people_in_one_tent, size=self.dims_block2[0] * self.dims_block2[1]))
        max_pop_per_struct_block3 = list(poisson.rvs(
            mu=self.number_of_people_in_one_tent, size=self.dims_block3[0] * self.dims_block3[1]))

        self.max_pop_per_struct = max_pop_per_struct_isoboxes \
                                  + max_pop_per_struct_block1 \
                                  + max_pop_per_struct_block2 \
                                  + max_pop_per_struct_block3

        self.n_structs = len(self.max_pop_per_struct)

        self.grid_isoboxes = create_grid(
            self.dims_isoboxes[0], self.dims_isoboxes[1], 0)
        self.grid_block1 = create_grid(
            self.dims_block1[0], self.dims_block1[1], self.grid_isoboxes[-1][-1] + 1)
        self.grid_block2 = create_grid(
            self.dims_block2[0], self.dims_block2[1], self.grid_block1[-1][-1] + 1)
        self.grid_block3 = create_grid(
            self.dims_block3[0], self.dims_block3[1], self.grid_block2[-1][-1] + 1)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Distribution based parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        disease_params = ps.get_disease_params_network_model()
        model_params = disease_params.loc[:, ['Name', 'Value']]

        self.latentPeriod_mean = np.float(model_params[model_params['Name'] == 'latent_period_mean'].Value)
        self.latentPeriod_coeffvar = np.float(model_params[model_params['Name'] == 'latent_period_standard_deviation'].Value)

        self.presymptomaticPeriod_mean = np.float(model_params[model_params['Name'] == 'presymptomatic_period_mean'].Value)
        self.presymptomaticPeriod_coeffvar = np.float(model_params[model_params['Name'] == 'presymptomatic_period_standard_deviation'].Value)

        self.symptomaticPeriod_mean = np.float(model_params[model_params['Name'] == 'symptomatic_period_mean'].Value)
        self.symptomaticPeriod_coeffvar = np.float(model_params[model_params['Name'] == 'symptomatic_period_standard_deviation'].Value)

        self.onsetToHospitalizationPeriod_mean = np.float(model_params[model_params['Name'] == 'onset_to_hospitalization_period_mean'].Value)
        self.onsetToHospitalizationPeriod_coeffvar = np.float(model_params[model_params['Name'] == 'onset_to_hospitalization_period_standard_deviation'].Value)

        self.hospitalizationToDischargePeriod_mean = np.float(model_params[model_params['Name'] == 'hospitalization_to_discharge_period_mean'].Value)
        self.hospitalizationToDischargePeriod_coeffvar = np.float(model_params[model_params['Name'] == 'hospitalization_to_discharge_period_standard_deviation'].Value)

        self.hospitalizationToDeathPeriod_mean = np.float(model_params[model_params['Name'] == 'hospitalization_to_death_period_mean'].Value)
        self.hospitalizationToDeathPeriod_coeffvar = np.float(model_params[model_params['Name'] == 'hospitalization_to_death_period_standard_deviation'].Value)

        self.R0_mean = np.float(model_params[model_params['Name'] == 'R0_mean'].Value)
        self.R0_coeffvar = np.float(model_params[model_params['Name'] == 'R0_standard_deviation'].Value)

        self.sigma = 1 / \
                     gamma_dist(self.latentPeriod_mean, self.latentPeriod_coeffvar, self.total_population)
        self.lamda = 1 / \
                     gamma_dist(self.presymptomaticPeriod_mean,
                                self.presymptomaticPeriod_coeffvar, self.total_population)
        self.gamma = 1 / \
                     gamma_dist(self.symptomaticPeriod_mean,
                                self.symptomaticPeriod_coeffvar, self.total_population)
        self.eta = 1 / gamma_dist(self.onsetToHospitalizationPeriod_mean,
                                  self.onsetToHospitalizationPeriod_coeffvar, self.total_population)
        self.gamma_H = 1 / gamma_dist(self.hospitalizationToDischargePeriod_mean,
                                      self.hospitalizationToDischargePeriod_coeffvar,
                                      self.total_population)
        self.mu_H = 1 / gamma_dist(self.hospitalizationToDeathPeriod_mean,
                                   self.hospitalizationToDeathPeriod_coeffvar, self.total_population)
        self.R0 = gamma_dist(self.R0_mean, self.R0_coeffvar, self.total_population)

        self.infectiousPeriod = 1 / self.lamda + 1 / self.gamma
        self.beta = 1 / self.infectiousPeriod * self.R0
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