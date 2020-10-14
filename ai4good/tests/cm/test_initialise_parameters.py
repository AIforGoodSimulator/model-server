import os
import unittest
import pandas as pd
from ai4good.runner.facade import Facade
from ai4good.models.model_registry import get_models, create_params


class InitialiseParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.facade = Facade.simple()
        self.mdl = get_models()['compartmental-model'](self.facade.ps)

    def test_cm_category(self):
        params = create_params(self.facade.ps, 'compartmental-model',
                               self.facade.ps.get_profiles('compartmental-model')[0],
                               'Moria', None)

        self.assertCountEqual(params.change_in_categories, ['C' + x for x in params.calculated_categories])
        self.assertEqual(len(params.categories), 2*len(params.calculated_categories) + 1)
        self.assertEqual(params.control_dict['better_hygiene']['value'],
                         params.better_hygiene)

    def test_cm_custom_profile(self):
        custom_profile_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources/profile.csv'))
        params = create_params(self.facade.ps, 'compartmental-model',
                               custom_profile_df, 'Moria', None)
        self.assertEqual(params.control_dict['ICU_capacity']['value'],
                         int(custom_profile_df[custom_profile_df['Parameter'] == 'ICU_capacity']['Value'])/params.population)

        self.assertEqual(params.control_dict['better_hygiene']['value'],
                         float(custom_profile_df[custom_profile_df['Parameter'] == 'better_hygiene']['Value']))

        self.assertIsNotNone(params.control_dict['remove_symptomatic']['timing'])

        self.assertIsNotNone(params.control_dict['remove_high_risk']['n_categories_removed'])

        self.assertEqual(params.control_dict['numberOfIterations'], 2)
        self.assertEqual(params.control_dict['t_sim'], 200)
