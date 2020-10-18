"""
This file sets up the parameters for ABM models used in the cov_functions_AI.py
"""

import numpy as np
import pandas as pd
import json
import hashlib
from ai4good.params.param_store import ParamStore
from . import abm
import math


class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}):
        self.ps = ps
        self.camp = camp
        disease_params = ps.get_disease_params()
        camp_params = ps.get_camp_params(camp)
        # ------------------------------------------------------------
        # disease params
        parameter_csv = disease_params
        model_params = parameter_csv[parameter_csv['Type']
                                     == 'Model Parameter']
        model_params = model_params.loc[:, ['Name', 'Value']]
        control_data = parameter_csv[parameter_csv['Type'] == 'Control']
        self.model_params = model_params

        profile.set_index('Parameter', inplace=True)

        self.number_of_people_in_isoboxes = int(
            profile.loc['number_of_people_in_isoboxes', 'Value'])
        self.number_of_people_in_one_isobox = int(
            profile.loc['number_of_people_in_one_isobox', 'Value'])
        self.number_of_isoboxes = self.number_of_people_in_isoboxes / \
            self.number_of_people_in_one_isobox

        self.number_of_people_in_tents = int(
            profile.loc['number_of_people_in_tents', 'Value'])
        self.number_of_people_in_one_tent = int(
            profile.loc['number_of_people_in_one_tent', 'Value'])
        self.number_of_tents = self.number_of_people_in_tents / \
            self.number_of_people_in_one_tent

        self.total_population = self.number_of_people_in_isoboxes + \
            self.number_of_people_in_tents
        # float(profile.loc['permanently_asymptomatic_cases','Value'])
        self.permanently_asymptomatic_cases = 0.179

        self.age_and_gender = abm.read_age_gender(self.total_population)

        # float(profile.loc['area_covered_by_isoboxes','Value'])
        self.area_covered_by_isoboxes = 0.5
        # float(profile.loc['relative_strength_of_interaction','Value'])
        self.relative_strength_of_interaction = 0.2

        # float(profile.loc['smaller_movement_radius','Value'])
        self.smaller_movement_radius = 0.02
        # float(profile.loc['larger_movement_radius','Value'])
        self.larger_movement_radius = 0.1
        # float(profile.loc['overlapping_rages_radius','Value'])
        self.overlapping_rages_radius = 0.02

        self.number_of_steps = int(profile.loc['number_of_steps', 'Value'])
        self.number_of_states = 14
        self.track_states = np.zeros(
            (self.number_of_steps, self.number_of_states))
        # self.ACTIVATE_INTERVENTION = profile.loc['ACTIVATE_INTERVENTION', 'Value']
        # int(profile.loc['total_number_of_hospitalized','Value'])
        self.total_number_of_hospitalized = 0

        self.num_toilet_visit = int(profile.loc['num_toilet_visit', 'Value'])
        self.num_toilet_contact = int(
            profile.loc['num_toilet_contact', 'Value'])
        self.num_food_visit = int(profile.loc['num_food_visit', 'Value'])
        self.num_food_contact = int(profile.loc['num_food_contact', 'Value'])
        self.pct_food_visit = float(profile.loc['pct_food_visit', 'Value'])

        # float(profile.loc['transmission_reduction','Value'])
        self.transmission_reduction = 1

        # float(profile.loc['probability_infecting_person_in_household_per_day','Value'])
        self.probability_infecting_person_in_household_per_day = 0.33
        # float(profile.loc['probability_infecting_person_in_foodline_per_day','Value'])
        self.probability_infecting_person_in_foodline_per_day = 0.407
        # float(profile.loc['probability_infecting_person_in_toilet_per_day','Value'])
        self.probability_infecting_person_in_toilet_per_day = 0.099
        # float(profile.loc['probability_infecting_person_in_moving_per_day','Value'])
        self.probability_infecting_person_in_moving_per_day = 0.017

        # float(profile.loc['probability_spotting_symptoms_per_day','Value'])
        self.probability_spotting_symptoms_per_day = 0.05
        self.clearday = int(profile.loc['clearday', 'Value'])
        tb = profile.loc['toilets_blocks', 'Value'].split(',')
        self.toilets_blocks = (int(tb[0]), int(tb[1]))
        fb = profile.loc['foodline_blocks', 'Value'].split(',')
        self.foodline_blocks = (int(fb[0]), int(fb[1]))

        self.population = abm.form_population_matrix(
            self.total_population,
            self.number_of_isoboxes,
            self.number_of_people_in_isoboxes,
            self.number_of_tents,
            self.number_of_people_in_tents,
            self.permanently_asymptomatic_cases,
            self.age_and_gender
        )

        self.households_location = abm.place_households(self.population[:, 0].astype(
            int), self.area_covered_by_isoboxes, self.number_of_isoboxes)

        self.toilets_location, self.toilets_numbers, self.toilets_sharing = \
            abm.position_toilet(self.households_location,
                                self.toilets_blocks[0], self.toilets_blocks[1])
        self.foodpoints_location, self.foodpoints_numbers, self.foodpoints_sharing = \
            abm.position_foodline(self.households_location, self.foodline_blocks[0], self.foodline_blocks[1])
        self.ethnical_corellations = abm.create_ethnic_groups(self.households_location, self.relative_strength_of_interaction)
        self.local_interaction_space = abm.interaction_neighbours(self.households_location, self.smaller_movement_radius, self.larger_movement_radius, self.overlapping_rages_radius, self.ethnical_corellations)

        # The probability that a person's disease state will change from mild->recovered (0.37)
        # Liu et al 2020 The Lancet.
        self.mild_rec = np.random.uniform(0, 1, self.total_population) > math.exp(0.2 * math.log(0.1))

        # The probability that a person's disease state will change from severe->recovered (0.071)
        # Cai et al.
        self.sev_rec = np.random.uniform(0, 1, self.total_population) > math.exp(math.log(63 / 153) / 12)

        # Get random numbers to determine health states
        self.pick_sick = np.random.uniform(0, 1, self.total_population)

        self.ethnic_group1 = float(profile.loc['ethnic_group1', 'Value'])
        self.ethnic_group2 = float(profile.loc['ethnic_group2', 'Value'])
        self.ethnic_group3 = float(profile.loc['ethnic_group3', 'Value'])
        self.ethnic_group4 = float(profile.loc['ethnic_group4', 'Value'])
        self.ethnic_group5 = float(profile.loc['ethnic_group5', 'Value'])
        self.ethnic_group6 = float(profile.loc['ethnic_group6', 'Value'])
        self.ethnic_group7 = float(profile.loc['ethnic_group7', 'Value'])
        self.ethnic_others = float(profile.loc['ethnic_others', 'Value'])

        self.percentage_of_toilet_queue_cleared_at_each_step = \
            float(profile.loc['percentage_of_toilet_queue_cleared_at_each_step', 'Value'])

        self.infection_radius = float(profile.loc['infection_radius', 'Value'])
        self.pct_female = float(profile.loc['pct_female', 'Value'])

        # Age proportions by age slot i.e. 0-9, 10-19, 20-29, ... 80-89, 90+
        self.pct_0_9 = float(profile.loc['pct_0_9', 'Value'])
        self.pct_10_19 = float(profile.loc['pct_10_19', 'Value'])
        self.pct_20_29 = float(profile.loc['pct_20_29', 'Value'])
        self.pct_30_39 = float(profile.loc['pct_30_39', 'Value'])
        self.pct_40_49 = float(profile.loc['pct_40_49', 'Value'])
        self.pct_50_59 = float(profile.loc['pct_50_59', 'Value'])
        self.pct_60_69 = float(profile.loc['pct_60_69', 'Value'])
        self.pct_70_79 = float(profile.loc['pct_70_79', 'Value'])
        self.pct_80_89 = float(profile.loc['pct_80_89', 'Value'])
        self.pct_90 = float(profile.loc['pct_90', 'Value'])

        self.validate()

        self.control_dict = {}

    def sha1_hash(self) -> str:
        hash_params = [
            # {i: self.control_dict[i] for i in self.control_dict if i != 'nProcesses'},
            # self.track_states.tolist(),
            # self.population.tolist(),
            self.camp,
            self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash

    def validate(self):
        # Validate parameter values
        # TODO: add more validations on the parameter values here

        assert 0.0 <= self.infection_radius <= 1.0, "Infection radius must be between 0 and 1"

        # Proportion of females in the camp provided must be 0->1
        assert 0.0 <= self.pct_female <= 1.0, "Female proportion in the camp must be a number between 0 and 1"

        # Sum of proportions of agents in different age slots must be ~1
        assert 0.9 <= (self.pct_0_9 + self.pct_10_19 + self.pct_20_29 + self.pct_30_39 + self.pct_40_49 + self.pct_50_59 +
                self.pct_60_69 + self.pct_70_79 + self.pct_80_89 + self.pct_90) <= 1.0, \
            "Age proportions must add up to ~1"
