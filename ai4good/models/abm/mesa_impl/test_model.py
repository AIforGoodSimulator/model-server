import unittest

import ai4good.models.abm.mesa_impl.model as mm
from ai4good.runner.facade import Facade
from ai4good.models.model_registry import create_params
import numpy as np


def get_params():
    _model = 'agent-based-model'
    _profile = 'small'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


class MyTestCase(unittest.TestCase):

    def test_model(self):

        params = get_params()
        mod = mm.Camp(params=params)

        test = 1

        np.savetxt('agents_age.csv', mod.agents_age, delimiter=',', header='agent_age')
        np.savetxt('agents_disease_states.csv', mod.agents_disease_states, delimiter=',',
                   header='agents_disease_states')
        np.savetxt('agents_gender.csv', mod.agents_gender, delimiter=',',
                   header='agents_gender')
        np.savetxt('agents_home_ranges.csv', mod.agents_home_ranges, delimiter=',',
                   header='agents_home_ranges')
        np.savetxt('agents_households.csv', mod.agents_households, delimiter=',',
                   header='agents_households')
        np.savetxt('agents_incubation_periods.csv', mod.agents_incubation_periods, delimiter=',',
                   header='agents_incubation_periods')
        np.savetxt('agents_pos.csv', mod.agents_pos, delimiter=',',
                   header='agents_pos1,agents_pos2')
        np.savetxt('agents_route.csv', mod.agents_route, delimiter=',',
                   header='agents_route')
        print("foodline_queue: " + str(mod.foodline_queue))
        np.savetxt('toilets.csv', mod.toilets, delimiter=',',
                   header='toilets1,toilets2')
        print("toilets_queue: " + str(mod.toilets_queue))

        np.savetxt('foodlines.csv', mod.foodlines, delimiter=',',
                   header='foodlines1,foodlines2')
        np.savetxt('households.csv', mod.households, delimiter=',',
                   header='households1,households2')

        np.savetxt('agents_ethnic_groups.csv', mod.agents_ethnic_groups, delimiter=',',
                   header='agents_ethnic_groups')
        # print("people_count: " + str(mod.people_count))
        # print("running: " + str(mod.running))

        # np.savetxt('agents_incubation_periods.csv', mod.agents_incubation_periods, delimiter=',',
        #            header='agents_incubation_periods')
        # np.savetxt('agents_incubation_periods.csv', mod.agents_incubation_periods, delimiter=',',
        #            header='agents_incubation_periods')
        # np.savetxt('agents_incubation_periods.csv', mod.agents_incubation_periods, delimiter=',',
        #            header='agents_incubation_periods')

        # print("ethnic groups: " + mod.agents_ethnic_groups)

        # print("agents_ethnic_groups: " + str(mod.agents_ethnic_groups))


        print(mod)
        # test = 1

        result = []
        expect = []
        self.assertEqual(result, expect)


if __name__ == '__main__':
    unittest.main()
