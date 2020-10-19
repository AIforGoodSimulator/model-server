"""
This file sets up the parameters for ABM models used in the cov_functions_AI.py
"""

import json
import hashlib
import logging
import numpy as np
import pandas as pd

from ai4good.params.param_store import ParamStore
from ai4good.utils.path_utils import get_am_aug_pop

VALUE = "Value"  # Name of the column in parameter file containing parameter value


class Parameters(object):

    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict):
        self.ps = ps
        self.camp = camp
        disease_params = ps.get_disease_params()
        camp_params = ps.get_camp_params(camp)

        parameter_csv = disease_params
        model_params = parameter_csv[parameter_csv['Type'] == 'Model Parameter']
        model_params = model_params.loc[:, ['Name', VALUE]]
        self.model_params = model_params

        profile.set_index('Parameter', inplace=True)

        ###############################################################################################################
        # Parameters about the simulation

        # Number of days of simulation
        self.number_of_steps = int(profile.loc['number_of_steps', VALUE])

        ###############################################################################################################
        # Parameters about the camp

        # Total number of people in the iso-boxes
        self.number_of_people_in_isoboxes = int(profile.loc['number_of_people_in_isoboxes', VALUE])
        # Iso-box capacity
        self.number_of_people_in_one_isobox = int(profile.loc['number_of_people_in_one_isobox', VALUE])
        # Total number of people in the tents
        self.number_of_people_in_tents = int(profile.loc['number_of_people_in_tents', VALUE])
        # Tent capacity
        self.number_of_people_in_one_tent = int(profile.loc['number_of_people_in_one_tent', VALUE])
        # Proportion of area covered by iso-boxes
        self.area_covered_by_isoboxes = float(profile.loc['area_covered_by_isoboxes', VALUE])
        # Average number of toilets visit per day
        self.num_toilet_visit = int(profile.loc['num_toilet_visit', VALUE])
        # Average number of food line visits per day
        self.num_food_visit = int(profile.loc['num_food_visit', VALUE])
        # TODO
        self.pct_food_visit = float(profile.loc['pct_food_visit', VALUE])
        # Grid size for toilet placement
        tb = profile.loc['toilets_blocks', VALUE].split(',')
        self.toilets_blocks = (int(tb[0]), int(tb[1]))
        # Grid size for food line placement
        fb = profile.loc['foodline_blocks', VALUE].split(',')
        self.foodline_blocks = (int(fb[0]), int(fb[1]))
        # Ethnic groups and their proportions in the camp
        self.ethnic_groups = [
            # Currently, all agents have same ethnicity, this will be added as input from camp later on
            ['ethnic_group1', 1.0]
        ]
        # Radius around a person where infection spread can happen. If people engage more in the camp, then this value
        # will be higher (e.g. more handshakes etc.)
        try:
            # TODO: add to parameters after confirming with Gaia and Vera
            self.infection_radius = float(profile.loc['infection_radius', VALUE])
        except KeyError:
            self.infection_radius = 0.0001

        ###############################################################################################################
        # Parameters about the agents

        # Probability that agent >16 yrs old becomes asymptomatic
        self.permanently_asymptomatic_cases = float(profile.loc['permanently_asymptomatic_cases', VALUE])
        # Age and gender data (read from file)
        self.age_and_gender = self.read_age_gender(self.number_of_people_in_isoboxes + self.number_of_people_in_tents)
        # Home range for agents who are either 1)female or 2)less than 10 yrs old
        self.smaller_movement_radius = float(profile.loc['smaller_movement_radius', VALUE])
        # Home range for rest of the agents
        self.larger_movement_radius = float(profile.loc['larger_movement_radius', VALUE])
        # Account for relative encounter rate between agents of same ethnicity
        self.relative_strength_of_interaction = float(profile.loc['relative_strength_of_interaction', VALUE])

        ###############################################################################################################
        # Parameters relevant for interventions

        # Factor to scale probability of spread (by wearing masks, etc.)
        self.transmission_reduction = float(profile.loc['transmission_reduction', VALUE])
        # Probability that camp managers can detect symptomatic people in the camp so as to isolate them
        self.probability_spotting_symptoms_per_day = float(profile.loc['probability_spotting_symptoms_per_day', VALUE])
        # Number of days needed to be cleared of isolation
        self.clear_day = int(profile.loc['clearday', VALUE])
        # Probability that people will violate lockdown
        self.prop_violating_lockdown = float(profile.loc["prop_violating_lockdown", VALUE])

        self.percentage_of_toilet_queue_cleared_at_each_step = 0.8
        # TODO: add to the new list of parameters
        # float(profile.loc['percentage_of_toilet_queue_cleared_at_each_step', VALUE])

        self.validate()

    def sha1_hash(self) -> str:
        hash_params = [
            self.camp,
            self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash

    def read_age_gender(self, num_ppl):
        # Data frame. V1 = age, V2 is sex (1 = male?, 0  = female?).

        age_and_gender = pd.read_csv(get_am_aug_pop())
        age_and_gender = age_and_gender.loc[:, ~age_and_gender.columns.str.contains('^Unnamed')]
        age_and_gender = age_and_gender.values

        if age_and_gender.shape[0] < num_ppl:
            logging.warning("Number of agents ({}) are more than data provided in age_and_gender.csv ({})".
                            format(num_ppl, age_and_gender.shape[0]))

        age_and_gender = age_and_gender[np.random.randint(age_and_gender.shape[0], size=num_ppl)]
        return age_and_gender

    def validate(self):
        # Validate parameter values so that only valid simulations are run

        assert self.number_of_steps > 0, "Parameter must be positive integer"

        assert self.number_of_people_in_isoboxes > 0, "Parameter must be a positive integer"
        assert self.number_of_people_in_one_isobox > 0, "Parameter must be a positive integer"
        assert self.number_of_people_in_tents > 0, "Parameter must be a positive integer"
        assert self.number_of_people_in_one_tent > 0, "Parameter must be a positive integer"
        assert 0.0 < self.area_covered_by_isoboxes < 1.0, "Parameter value must be between (0,1)"
        assert self.num_toilet_visit > 0, "Parameter must be a positive integer"
        assert self.num_food_visit > 0, "Parameter must be a positive integer"
        assert 0.0 < self.pct_food_visit < 1.0, "Parameter value must be between (0,1)"
        assert self.toilets_blocks[0] == self.toilets_blocks[1], "UnImplemented: Toilet blocks must be square grid"
        assert self.foodline_blocks[0] == self.foodline_blocks[1], "UnImplemented: Food line blocks must be square grid"
        tot_eth = 0.0
        for eth in self.ethnic_groups:
            assert 0.0 <= eth[1] <= 1.0, "Parameter must be between [0, 1]"
            tot_eth += eth[1]
        assert tot_eth == 1.0, "Total ethnic distribution must add up to 1.0"
        assert 0.0 < self.infection_radius < 1.0, "Parameter value must be between (0,1)"

        assert 0.0 < self.permanently_asymptomatic_cases < 1.0, "Parameter value must be between (0,1)"
        assert 0.0 < self.smaller_movement_radius < 1.0, "Parameter value must be between (0,1)"
        assert 0.0 < self.larger_movement_radius < 1.0, "Parameter value must be between (0,1)"
        assert 0.0 < self.relative_strength_of_interaction <= 1.0, "Parameter value must be between (0,1]"

        assert 0.0 < self.transmission_reduction <= 1.0, "Probability value must be between (0,1]"
        assert 0.0 <= self.probability_spotting_symptoms_per_day <= 1.0, "Probability value must be between [0,1]"
        assert self.clear_day > 0, "Parameter must be a positive integer"
        assert 0.0 <= self.prop_violating_lockdown <= 1.0, "Probability value must be between [0,1]"