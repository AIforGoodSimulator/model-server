"""
This file sets up the parameters for NM models
"""

import numpy as np
import pandas as pd
import json
import hashlib
from ai4good.params.param_store import ParamStore
from ai4good.models.nm import abm
import math


class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}):
        self.ps = ps
        self.camp = camp
        disease_params = ps.get_disease_params()
        camp_params = ps.get_camp_params(camp)

        # SEIRS+ Model parameters
        transmission_rate_list = [1.28]
        progression_rate_list = [round(1 / 5.1, 3)]
        recovery_rate_list = [0.056]  # Approx 1/18 -> Recovery occurs after 18 days
        hosp_rate_list = [round(1 / 11.4, 3)]  # 1/6.3 # From Tucker Model
        # crit_rate = 0.3 # From camp_params
        crit_rate = list((sample_pop["death_rate"] / sample_pop["prob_symptomatic"]) / sample_pop["prob_hospitalisation"])
        death_rate_list = [0.75]
        prob_global_contact = 1
        prob_detected_global_contact = 1
        # prob_hosp_to_critical = list(sample_pop["death_rate"]/sample_pop["prob_hospitalisation"])
        prob_death = list(sample_pop["death_rate"])
        prob_asymptomatic = list(1 - sample_pop["prob_symptomatic"])
        prob_symp_to_hosp = list(sample_pop["prob_hospitalisation"])
        init_symp_cases = 1
        init_asymp_cases = 1


    def sha1_hash(self) -> str:
        hash_params = [
            {i: self.control_dict[i] for i in self.control_dict if i != 'nProcesses'},
            self.population.tolist(),
            self.camp,
            self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash

