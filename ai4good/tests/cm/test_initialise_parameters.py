import os
import unittest
import numpy as np
import pandas as pd
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.runner.facade import Facade
from ai4good.models.cm.initialise_parameters import Parameters


class InitialiseParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.facade = Facade.simple()
        self.profile_df = self.facade.ps.get_params(CompartmentalModel.ID, 'baseline')

    def test_cm_category(self):
        params = Parameters(self.facade.ps, 'Moria', '',
                            self.profile_df, {})

        self.assertCountEqual(params.change_in_categories, ['C' + x for x in params.calculated_categories])
        self.assertEqual(len(params.categories), 2*len(params.calculated_categories) + 1)
        self.assertEqual(params.control_dict['better_hygiene']['value'],
                         params.better_hygiene)

    def test_cm_custom_profile(self):
        custom_profile_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources/profile.csv'))
        params = Parameters(self.facade.ps, 'Moria', '',
                            custom_profile_df, {})
        self.assertEqual(params.control_dict['ICU_capacity']['value'],
                         int(custom_profile_df[custom_profile_df['Parameter'] == 'ICU_capacity']['Value'])/params.population)

        self.assertEqual(params.control_dict['better_hygiene']['value'],
                         float(custom_profile_df[custom_profile_df['Parameter'] == 'better_hygiene']['Value']))

        self.assertIsNotNone(params.control_dict['remove_symptomatic']['timing'])

        self.assertIsNotNone(params.control_dict['remove_high_risk']['n_categories_removed'])

        self.assertEqual(params.control_dict['numberOfIterations'], 2)
        self.assertEqual(params.control_dict['t_sim'], 200)
        expected = pd.read_csv('cm/resources/Albania_contact_matrix.csv', index_col=0)
        actual = pd.DataFrame(params.infection_matrix)
        np.array_equal(expected.values, actual.values)
        np.array_equal(params.camp_params.to_numpy(), self.facade.ps.get_camp_params('Moria').to_numpy())

    def test_specific_country_contact_matrix(self):
        params = Parameters(self.facade.ps, 'Moria', 'Albania',
                            self.profile_df, {})
        expected = pd.read_csv('cm/resources/Albania_contact_matrix.csv', index_col=0)
        actual = pd.DataFrame(params.infection_matrix)
        np.array_equal(expected.values, actual.values)
        np.array_equal(params.camp_params.to_numpy(), self.facade.ps.get_camp_params('Moria').to_numpy())
