"""
This file sets up the parameters for ABM models used in the cov_functions_AI.py
"""

import numpy as np
import pandas as pd
import json
import hashlib
from ai4good.params.param_store import ParamStore
from . import abm


class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}):
        self.ps = ps
        self.camp = camp
        disease_params = ps.get_disease_params()
        camp_params = ps.get_camp_params(camp)
        # ------------------------------------------------------------
        # disease params
        parameter_csv = disease_params
        model_params = parameter_csv[parameter_csv['Type'] == 'Model Parameter']
        model_params = model_params.loc[:, ['Name', 'Value']]
        control_data = parameter_csv[parameter_csv['Type'] == 'Control']
        self.model_params = model_params
        
        self.number_of_people_in_isoboxes = self.model_params['number_of_people_in_isoboxes']
        self.number_of_people_in_one_isobox = self.model_params['number_of_people_in_one_isobox']
        self.number_of_isoboxes = self.number_of_people_in_isoboxes / self.number_of_people_in_one_isobox

        self.number_of_people_in_tents = self.model_params['number_of_people_in_tents']
        self.number_of_people_in_one_tent = self.model_params['number_of_people_in_one_tent']
        self.number_of_tents = self.number_of_people_in_tents / self.number_of_people_in_one_tent

        self.total_population = self.number_of_people_in_isoboxes + self.number_of_people_in_tents
        self.permanently_asymptomatic_cases = self.model_params['permanently_asymptomatic_cases']
        self.age_and_gender = abm.read_age_gender(self.total_population)

        self.area_covered_by_isoboxes = self.model_params['area_covered_by_isoboxes']
        self.relative_strength_of_interaction = self.model_params['relative_strength_of_interaction']

        self.smaller_movement_radius = self.model_params['smaller_movement_radius']
        self.larger_movement_radius = self.model_params['larger_movement_radius']
        self.overlapping_rages_radius = self.model_params['overlapping_rages_radius']

        self.number_of_steps = self.model_params['number_of_steps']
        self.number_of_states = 14
        self.track_states = np.zeros((self.number_of_steps, self.number_of_states))
        self.ACTIVATE_INTERVENTION = self.model_params['ACTIVATE_INTERVENTION']
        self.total_number_of_hospitalized = self.model_params['total_number_of_hospitalized']

        self.num_toilet_visit = self.model_params['num_toilet_visit']
        self.num_toilet_contact = self.model_params['num_toilet_contact']
        self.num_food_visit = self.model_params['num_food_visit']
        self.num_food_contact = self.model_params['num_food_contact']
        self.pct_food_visit = self.model_params['pct_food_visit']

        self.transmission_reduction = self.model_params['transmission_reduction']

        self.probability_infecting_person_in_household_per_day = self.model_params['probability_infecting_person_in_household_per_day']
        self.probability_infecting_person_in_foodline_per_day = self.model_params['probability_infecting_person_in_foodline_per_day']
        self.probability_infecting_person_in_toilet_per_day = self.model_params['probability_infecting_person_in_toilet_per_day']
        self.probability_infecting_person_in_moving_per_day = self.model_params['probability_infecting_person_in_moving_per_day']

        self.probability_spotting_symptoms_per_day = self.model_params['probability_spotting_symptoms_per_day']
        self.clearday = self.model_params['clearday']
        self.toilets_blocks = self.model_params['toilets_blocks']
        self.foodline_blocks = self.model_params['foodline_blocks']

        self.population = abm.form_population_matrix(
            self.total_population,
            self.number_of_isoboxes,
            self.number_of_people_in_isoboxes,
            self.number_of_tents,
            self.number_of_people_in_tents,
            self.permanently_asymptomatic_cases,
            self.age_and_gender
        )

        self.households_location = abm.place_households(self.population[:, 0].astype(int), self.area_covered_by_isoboxes, self.number_of_isoboxes)

        self.toilets_location, self.toilets_numbers, self.toilets_sharing = \
            abm.position_toilet(self.households_location , self.toilets_blocks[0], self.toilets_blocks[1])
        self.foodpoints_location, self.foodpoints_numbers, self.foodpoints_sharing = \
            abm.position_foodline(self.households_location, self.foodline_blocks[0], self.foodline_blocks[1])
        self.ethnical_corellations = abm.create_ethnic_groups(self.households_location, self.relative_strength_of_interaction)
        self.local_interaction_space = abm.interaction_neighbours(self.households_location, self.smaller_movement_radius, self.larger_movement_radius, self.overlapping_rages_radius, self.ethnical_corellations)

        self.mild_rec = np.random.uniform(0, 1, self.total_population) > math.exp(0.2 * math.log(0.1))  # Liu et al 2020 The Lancet.
        self.sev_rec = np.random.uniform(0, 1, self.total_population) > math.exp(math.log(63 / 153) / 12)  # Cai et al.
        self.pick_sick = np.random.uniform(0, 1, self.total_population)  # Get random numbers to determine health states.


    def sha1_hash(self) -> str:
        hash_params = [
            {i: self.control_dict[i] for i in self.control_dict if i != 'nProcesses'},
            self.track_states,
            self.population,
            self.camp,
            self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash

