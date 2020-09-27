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
    def __init__(self,  camp: str = "Moria", t_steps=200):
        #self.ps = ps
        self.camp = camp
        # disease_params = ps.get_disease_params() # Do we need this?
        # camp_params = ps.get_camp_params(camp)   TODO: This SHOULD link to camp_params.py somehow
        # n_pop = camp_params.n_pop

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Baseline parameters
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # Distribution based parameters
        self.latentPeriod_mean, latentPeriod_coeffvar = 3.0, 0.6
        self.presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar = 2.2, 0.5
        self.symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 4.0, 0.4
        self.onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
        self.hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
        self.hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
        self.R0_mean, R0_coeffvar = 2.5, 0.2

        self.sigma = 1 / \
            gamma_dist(self.latentPeriod_mean, latentPeriod_coeffvar, n_pop)
        self.lamda = 1 / \
            gamma_dist(self.presymptomaticPeriod_mean,
                       presymptomaticPeriod_coeffvar, n_pop)
        self.gamma = 1 / \
            gamma_dist(self.symptomaticPeriod_mean,
                       symptomaticPeriod_coeffvar, n_pop)
        self.eta = 1 / gamma_dist(self.onsetToHospitalizationPeriod_mean,
                                  onsetToHospitalizationPeriod_coeffvar, n_pop)
        self.gamma_H = 1 / gamma_dist(self.hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar,
                                      n_pop)
        self.mu_H = 1 / gamma_dist(self.hospitalizationToDeathPeriod_mean,
                                   hospitalizationToDeathPeriod_coeffvar, n_pop)
        self.R0 = gamma_dist(self.R0_mean, R0_coeffvar, n_pop)

        infectiousPeriod = 1 / self.lamda + 1 / self.gamma
        self.beta = 1 / infectiousPeriod * self.R0
        self.beta_pairwise_mode = 'infected'
        self.delta_pairwise_mode = 'mean'

        # Constant parameters
        self.p_global_interaction = 0.2
        self.init_exposed = int(n_pop / 100)
        self.init_infected = int(n_pop / 100)

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

    def initialise_age_parameters(self, graph):
        # Age based parameters
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
            n_pop,
            self.camp,
            # self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash
