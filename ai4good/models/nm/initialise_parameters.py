"""
This file sets up the parameters for NM models
"""

import hashlib
import json

from scipy.stats import poisson
from seirsplus.utilities import (
    dist_info,
    gamma_dist,
    network_info,
    numpy,
    pyplot,
    results_summary,
)

from ai4good.models.nm.utils.network_utils import create_grid, get_values_per_node
from ai4good.models.nm.utils.stats_utils import sample_population
from ai4good.params.param_store import ParamStore


class Parameters:
    def __init__(
        self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}
    ):
        self.ps = ps
        if profile is not None:
            self.profile_name = profile["Profile"].iloc[0]
        else:
            self.profile_name = "None"

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Camp parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.camp = camp
        self.camp_params = ps.get_camp_params_network_model(self.camp)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Model Parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        profile.set_index("Parameter", inplace=True)

        # Sample the population age, and parameter rates
        self.total_population = int(
            self.camp_params.Total_population.dropna().sum()
        )  # n_pop
        self.sample_pop = sample_population(
            self.total_population, "ai4good/models/nm/data/augmented_population.csv"
        )

        # Tents
        self.number_of_tents = int(profile.loc["number_of_tents", "Value"])  # n_tents
        self.total_population_in_tents = int(
            profile.loc["total_population_in_tents", "Value"]
        )  # pop_tents
        self.number_of_people_in_one_tent = int(
            profile.loc["number_of_people_in_one_tent", "Value"]
        )  # pop_per_tent

        # The radius of nearby structures whose inhabitants a person has connections with
        self.neighbor_proximity = int(profile.loc["neighbor_proximity", "Value"])

        # Edge weight for friendship connections
        self.neighbor_weight = float(profile.loc["neighbor_weight", "Value"])

        # Edge weight for connections in the food queue
        self.food_weight = float(profile.loc["food_weight", "Value"])

        # Edge weight for connections within each structure
        self.household_weight = float(profile.loc["household_weight", "Value"])

        # Others
        self.number_of_ethnic_groups = int(
            profile.loc["number_of_ethnic_groups", "Value"]
        )  # n_ethnic_groups

        # Isoboxes
        self.number_of_people_in_isoboxes = int(
            profile.loc["number_of_people_in_isoboxes", "Value"]
        )  # pop_isoboxes
        self.number_of_people_in_one_isobox = int(
            profile.loc["number_of_people_in_one_isobox", "Value"]
        )  # pop_per_isobox

        self.t_steps = int(profile.loc["number_of_steps", "Value"])

        # Grid info for isoboxes
        dimensions_of_isoboxes = profile.loc["dimensions_of_isoboxes", "Value"].split(
            ","
        )
        self.dims_isoboxes = (
            int(dimensions_of_isoboxes[0]),
            int(dimensions_of_isoboxes[1]),
        )

        # Grid info for tents
        dimensions_of_block1 = profile.loc["dimensions_of_block1", "Value"].split(",")
        self.dims_block1 = (int(dimensions_of_block1[0]), int(dimensions_of_block1[1]))
        dimensions_of_block2 = profile.loc["dimensions_of_block2", "Value"].split(",")
        self.dims_block2 = (int(dimensions_of_block2[0]), int(dimensions_of_block2[1]))
        dimensions_of_block3 = profile.loc["dimensions_of_block3", "Value"].split(",")
        self.dims_block3 = (int(dimensions_of_block3[0]), int(dimensions_of_block3[1]))

        self.n_isoboxes = self.dims_isoboxes[0] * self.dims_isoboxes[1]

        # Define the maximum population per structures (tents and isoboxes) drawn from a poisson distribution
        max_pop_per_struct_isoboxes = list(
            poisson.rvs(mu=self.number_of_people_in_one_isobox, size=self.n_isoboxes)
        )
        max_pop_per_struct_block1 = list(
            poisson.rvs(
                mu=self.number_of_people_in_one_tent,
                size=self.dims_block1[0] * self.dims_block1[1],
            )
        )
        max_pop_per_struct_block2 = list(
            poisson.rvs(
                mu=self.number_of_people_in_one_tent,
                size=self.dims_block2[0] * self.dims_block2[1],
            )
        )
        max_pop_per_struct_block3 = list(
            poisson.rvs(
                mu=self.number_of_people_in_one_tent,
                size=self.dims_block3[0] * self.dims_block3[1],
            )
        )

        self.max_pop_per_struct = (
            max_pop_per_struct_isoboxes
            + max_pop_per_struct_block1
            + max_pop_per_struct_block2
            + max_pop_per_struct_block3
        )

        self.n_structs = len(self.max_pop_per_struct)

        self.grid_isoboxes = create_grid(
            self.dims_isoboxes[0], self.dims_isoboxes[1], 0
        )
        self.grid_block1 = create_grid(
            self.dims_block1[0], self.dims_block1[1], self.grid_isoboxes[-1][-1] + 1
        )
        self.grid_block2 = create_grid(
            self.dims_block2[0], self.dims_block2[1], self.grid_block1[-1][-1] + 1
        )
        self.grid_block3 = create_grid(
            self.dims_block3[0], self.dims_block3[1], self.grid_block2[-1][-1] + 1
        )

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Distribution based parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        disease_params = ps.get_disease_params_network_model()
        model_params = disease_params.loc[:, ["Name", "Value"]]

        self.latentPeriod_mean = np.float(
            model_params[model_params["Name"] == "latent_period_mean"].Value
        )
        self.latentPeriod_coeffvar = np.float(
            model_params[
                model_params["Name"] == "latent_period_standard_deviation"
            ].Value
        )

        self.presymptomaticPeriod_mean = np.float(
            model_params[model_params["Name"] == "presymptomatic_period_mean"].Value
        )
        self.presymptomaticPeriod_coeffvar = np.float(
            model_params[
                model_params["Name"] == "presymptomatic_period_standard_deviation"
            ].Value
        )

        self.symptomaticPeriod_mean = np.float(
            model_params[model_params["Name"] == "symptomatic_period_mean"].Value
        )
        self.symptomaticPeriod_coeffvar = np.float(
            model_params[
                model_params["Name"] == "symptomatic_period_standard_deviation"
            ].Value
        )

        self.onsetToHospitalizationPeriod_mean = np.float(
            model_params[
                model_params["Name"] == "onset_to_hospitalization_period_mean"
            ].Value
        )
        self.onsetToHospitalizationPeriod_coeffvar = np.float(
            model_params[
                model_params["Name"]
                == "onset_to_hospitalization_period_standard_deviation"
            ].Value
        )

        self.hospitalizationToDischargePeriod_mean = np.float(
            model_params[
                model_params["Name"] == "hospitalization_to_discharge_period_mean"
            ].Value
        )
        self.hospitalizationToDischargePeriod_coeffvar = np.float(
            model_params[
                model_params["Name"]
                == "hospitalization_to_discharge_period_standard_deviation"
            ].Value
        )

        self.hospitalizationToDeathPeriod_mean = np.float(
            model_params[
                model_params["Name"] == "hospitalization_to_death_period_mean"
            ].Value
        )
        self.hospitalizationToDeathPeriod_coeffvar = np.float(
            model_params[
                model_params["Name"]
                == "hospitalization_to_death_period_standard_deviation"
            ].Value
        )

        self.R0_mean = np.float(model_params[model_params["Name"] == "R0_mean"].Value)
        self.R0_coeffvar = np.float(
            model_params[model_params["Name"] == "R0_standard_deviation"].Value
        )

        self.sigma = 1 / gamma_dist(
            self.latentPeriod_mean, self.latentPeriod_coeffvar, self.total_population
        )
        self.lamda = 1 / gamma_dist(
            self.presymptomaticPeriod_mean,
            self.presymptomaticPeriod_coeffvar,
            self.total_population,
        )
        self.gamma = 1 / gamma_dist(
            self.symptomaticPeriod_mean,
            self.symptomaticPeriod_coeffvar,
            self.total_population,
        )
        self.eta = 1 / gamma_dist(
            self.onsetToHospitalizationPeriod_mean,
            self.onsetToHospitalizationPeriod_coeffvar,
            self.total_population,
        )
        self.gamma_H = 1 / gamma_dist(
            self.hospitalizationToDischargePeriod_mean,
            self.hospitalizationToDischargePeriod_coeffvar,
            self.total_population,
        )
        self.mu_H = 1 / gamma_dist(
            self.hospitalizationToDeathPeriod_mean,
            self.hospitalizationToDeathPeriod_coeffvar,
            self.total_population,
        )
        self.R0 = gamma_dist(self.R0_mean, self.R0_coeffvar, self.total_population)

        self.infectiousPeriod = 1 / self.lamda + 1 / self.gamma
        self.beta = 1 / self.infectiousPeriod * self.R0
        self.beta_pairwise_mode = "infected"
        self.delta_pairwise_mode = "mean"

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
        self.reduction_rate = np.float(
            model_params[model_params["Name"] == "reduction_rate"].Value
        )
        self.beta_q = self.beta * (self.reduction_rate / self.R0_mean)
        self.q_global_interactions = np.float(
            model_params[model_params["Name"] == "q_globalintxn"].Value
        )
        self.q_start = int(profile.loc["quarantine_start", "Value"])
        self.m_start = int(profile.loc["masks_start", "Value"])
        self.q_end = int(profile.loc["quarantine_end", "Value"])
        self.m_end = int(profile.loc["masks_end", "Value"])

    def initialise_age_parameters(self, graph):
        pctAsymptomatic_df = self.camp_params[["Age", "pctAsymptomatic"]].copy(
            deep=False
        )
        ageGroup_pctAsymp = pctAsymptomatic_df.set_index(
            "Age", drop=True, append=False
        ).to_dict()["pctAsymptomatic"]

        pctHospitalized_df = self.camp_params[["Age", "pctHospitalized"]].copy(
            deep=False
        )
        ageGroup_pctHospitalized = pctHospitalized_df.set_index(
            "Age", drop=True, append=False
        ).to_dict()["pctHospitalized"]

        hospitalFatalityRate_df = self.camp_params[
            ["Age", "hospitalFatalityRate"]
        ].copy(deep=False)
        ageGroup_hospitalFatalityRate = hospitalFatalityRate_df.set_index(
            "Age", drop=True, append=False
        ).to_dict()["hospitalFatalityRate"]

        susceptibility_df = self.camp_params[["Age", "susceptibility"]].copy(deep=False)
        ageGroup_susceptibility = susceptibility_df.set_index(
            "Age", drop=True, append=False
        ).to_dict()["susceptibility"]

        self.pct_asymptomatic = get_values_per_node(ageGroup_pctAsymp, graph)
        self.pct_hospitalized = get_values_per_node(ageGroup_pctHospitalized, graph)
        self.pct_fatality = get_values_per_node(ageGroup_hospitalFatalityRate, graph)
        self.alpha = get_values_per_node(ageGroup_susceptibility, graph)

    def update_parameters(self, graph):
        new_pop_size = len(graph.nodes)
        self.total_population = new_pop_size
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Distribution based parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.sigma = 1 / gamma_dist(
            self.latentPeriod_mean, self.latentPeriod_coeffvar, new_pop_size
        )
        self.lamda = 1 / gamma_dist(
            self.presymptomaticPeriod_mean,
            self.presymptomaticPeriod_coeffvar,
            new_pop_size,
        )
        self.gamma = 1 / gamma_dist(
            self.symptomaticPeriod_mean, self.symptomaticPeriod_coeffvar, new_pop_size
        )
        self.eta = 1 / gamma_dist(
            self.onsetToHospitalizationPeriod_mean,
            self.onsetToHospitalizationPeriod_coeffvar,
            new_pop_size,
        )
        self.gamma_H = 1 / gamma_dist(
            self.hospitalizationToDischargePeriod_mean,
            self.hospitalizationToDischargePeriod_coeffvar,
            self.total_population,
        )
        self.mu_H = 1 / gamma_dist(
            self.hospitalizationToDeathPeriod_mean,
            self.hospitalizationToDeathPeriod_coeffvar,
            new_pop_size,
        )
        self.R0 = gamma_dist(self.R0_mean, self.R0_coeffvar, new_pop_size)

        self.infectiousPeriod = 1 / self.lamda + 1 / self.gamma
        self.beta = 1 / self.infectiousPeriod * self.R0
        self.beta_pairwise_mode = "infected"
        self.delta_pairwise_mode = "mean"

        # Constant parameters
        self.p_global_interaction = 0.2
        self.init_exposed = int(new_pop_size / 100)
        self.init_infected = int(new_pop_size / 100)

        self.transmission_mean = self.beta.mean()
        self.progressionToInfectious_mean = self.sigma.mean()
        self.progressionToSymptomatic_mean = self.lamda.mean()
        self.recovery_mean = self.gamma.mean()
        self.hospitalization_mean = self.eta.mean()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Reduced interaction parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.beta_q = self.beta * (self.reduction_rate / self.R0_mean)

    def sha1_hash(self) -> str:
        hash_params = [
            self.total_population,
            self.camp,
            self.neighbor_proximity,
            self.neighbor_weight,
            self.number_of_ethnic_groups,
            self.quarantine_start,
            self.quarantine_end,
            self.masks_start,
            self.masks_end,
            self.transmission_reduction,
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode("UTF-8"))
        _hash = hash_object.hexdigest()
        return _hash
