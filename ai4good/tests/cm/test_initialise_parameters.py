import os
import unittest
import numpy as np
import pandas as pd
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.runner.facade import Facade
from ai4good.models.cm.initialise_parameters import Parameters


# Set a default user input for debugging purposes:
user_input_params = ' {"name-camp": "Moria", "location": "Greece", "country-dropdown": "Greece", ' \
                    '"total-area": "null", "total-population": 18700, "age-population-0-5": 1968.175, ' \
                    '"age-population-6-9": 1968.175, "age-population-10-19": 3242.58, ' \
                    '"age-population-20-29": 4927.45, "age-population-30-39": 3208.92, ' \
                    '"age-population-40-49": 1727.88, "age-population-50-59": 1037.85, ' \
                    '"age-population-60-69": 474.98, "age-population-70+": 143.99, "gender-perc-female": 50,' \
                    '"gender-perc-male": 50, "ethnic-no-1": "null", "ethnic-no-2": "null", "ethnic-no-3": "null",' \
                    '"ethnic-no-4": "null", "ethnic-no-5": "null", "ethnic-no-6": "null", "ethnic-no-7": "null",' \
                    '"ethnic-no-8": "null", "ethnic-no-9": "null", "ethnic-no-10": "null",' \
                    '"accommodation-area-type1": "null", "accommodation-area-type2": "null",' \
                    '"accommodation-area-type3": "null", "accommodation-no-unit-type1": "null",' \
                    '"accommodation-no-unit-type2": "null", "accommodation-no-unit-type3": "null",' \
                    '"accommodation-no-person-type1": "null", "accommodation-no-person-type2": "null",' \
                    '"accommodation-no-person-type3": "null", "available-ICU-beds": "null",' \
                    '"increased-ICU-beds": "null", "remove-high-risk-off-site": "null",' \
                    '"age-min-moved-off-site": 60, "age-max-moved-off-site": 100,' \
                    '"number-known-comorbidity": "null", "isolation-centre-capacity": "null",' \
                    '"days-quarantine-tested-positive": "null", "community-shielding": "null",' \
                    '"community-surveillance-program": "null", "radio-intervene-social": "null",' \
                    '"radio-intervene-face": "null", "radio-intervene-handwashing": "null",' \
                    '"radio-intervene-testing": "null", "radio-intervene-lockdown": "null",' \
                    '"activity-no-place-admin": "null", "activity-no-place-food": "null",' \
                    '"activity-no-place-health": "null", "activity-no-place-recreational": "null",' \
                    '"activity-no-place-religious": "null", "activity-no-person-admin": "null",' \
                    '"activity-no-person-food": "null", "activity-no-person-health": "null",' \
                    '"activity-no-person-recreational": "null", "activity-no-person-religious": "null",' \
                    '"activity-no-visit-admin": "null", "activity-no-visit-food": "null",' \
                    '"activity-no-visit-health": "null", "activity-no-visit-recreational": "null",' \
                    '"activity-no-visit-religious": "null"}'


class InitialiseParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.facade = Facade.simple()
        self.profile_df = self.facade.ps.get_params(CompartmentalModel.ID, 'baseline')

    def test_cm_category(self):
        params = Parameters(self.facade.ps, user_input_params,
                            self.profile_df, {})

        self.assertCountEqual(params.change_in_categories, ['C' + x for x in params.calculated_categories])
        self.assertEqual(len(params.categories), 2*len(params.calculated_categories) + 1)
        self.assertEqual(params.control_dict['better_hygiene']['value'],
                         params.better_hygiene)

    def test_cm_custom_profile(self):
        custom_profile_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources/profile.csv'))
        params = Parameters(self.facade.ps, user_input_params,
                            custom_profile_df, {})
        self.assertEqual(params.control_dict['ICU_capacity']['value'],
                         int(custom_profile_df[custom_profile_df['Parameter'] == 'ICU_capacity']['Value'])/params.population)

        self.assertEqual(params.control_dict['better_hygiene']['value'],
                         float(custom_profile_df[custom_profile_df['Parameter'] == 'better_hygiene']['Value']))

        self.assertIsNotNone(params.control_dict['remove_symptomatic']['timing'])

        self.assertIsNotNone(params.control_dict['remove_high_risk']['n_categories_removed'])

        self.assertEqual(params.control_dict['numberOfIterations'], 2)
        self.assertEqual(params.control_dict['t_sim'], 200)
        expected = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources/Greece_contact_matrix.csv'), index_col=0)

        actual = pd.DataFrame(params.infection_matrix)
        self.assertTrue(np.allclose(expected.values, actual.values))
