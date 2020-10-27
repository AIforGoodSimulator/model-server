import unittest

from ai4good.runner.facade import Facade
from ai4good.models.abm.np_impl.moria import *
from ai4good.models.model_registry import create_params
from ai4good.utils.logger_util import get_logger

logger = get_logger(__name__)


def get_params(_profile='BaselineHTHI'):
    _model = 'agent-based-model'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params(facade.ps, _model, _profile, camp, overrides)
    return params


class CampTester(unittest.TestCase):

    """
    To run the unit tests, run following command from project root:
    > python -m unittest discover -s ai4good\models\abm\np_impl\test
    """

    PROFILE = "BaselineHTHI"

    def setUp(self):
        param = get_params()
        self.camp = Moria(params=param, profile=CampTester.PROFILE)

        n = random.randint(10, 20)
        logger.info("===============================================")
        logger.info("Running {} simulation steps as setup".format(n))

        # Run few simulation steps before testing something specific
        for i in range(n):
            self.camp.day()

    def test_count(self):
        logger.info("Running test: test_count")
        # Agents could be only in one of the defined 10 states
        total_state_agents = np.count_nonzero(
            (self.camp.agents[:, A_DISEASE] == INF_SUSCEPTIBLE) |
            (self.camp.agents[:, A_DISEASE] == INF_EXPOSED) |
            (self.camp.agents[:, A_DISEASE] == INF_PRESYMPTOMATIC) |
            (self.camp.agents[:, A_DISEASE] == INF_SYMPTOMATIC) |
            (self.camp.agents[:, A_DISEASE] == INF_MILD) |
            (self.camp.agents[:, A_DISEASE] == INF_SEVERE) |
            (self.camp.agents[:, A_DISEASE] == INF_ASYMPTOMATIC1) |
            (self.camp.agents[:, A_DISEASE] == INF_ASYMPTOMATIC2) |
            (self.camp.agents[:, A_DISEASE] == INF_RECOVERED) |
            (self.camp.agents[:, A_DISEASE] == INF_DECEASED)
        )
        self.assertTrue(total_state_agents == self.camp.agents.shape[0])

    def test_simulate_wander(self):
        logger.info("Running test: test_simulate_wander")
        n = self.camp.agents.shape[0]  # number of agents in the camp
        n_wander = 100  # number of agents to wander
        assert n >= n_wander
        to_wander = self.camp.agents[random.sample(range(n), n_wander)]  # send agents wandering

        # Values before execution
        pre_total_susc = np.count_nonzero(to_wander[:, A_DISEASE] == INF_SUSCEPTIBLE)
        pre_total_exps = np.count_nonzero(to_wander[:, A_DISEASE] == INF_EXPOSED)

        # Perform simulation
        to_wander, num_new_inf = Moria.simulate_wander(to_wander, CAMP_SIZE,
                                                      self.camp.params.relative_strength_of_interaction,
                                                      self.camp.params.infection_radius * CAMP_SIZE,
                                                      self.camp.params.prob_spread_wander)

        # Values after execution
        post_total_susc = np.count_nonzero(to_wander[:, A_DISEASE] == INF_SUSCEPTIBLE)
        post_total_exps = np.count_nonzero(to_wander[:, A_DISEASE] == INF_EXPOSED)

        # Test: Basic tests
        # Number of new infections must have bounds
        self.assertTrue(0 <= num_new_inf <= n_wander)
        # Number of agents returned must match number of input agents
        self.assertTrue(to_wander.shape[0] == n_wander)
        # Simulate function could only do transmission of susceptible->exposed
        self.assertTrue(pre_total_susc == (post_total_susc + num_new_inf))
        # Number of exposed agents should not decrease
        self.assertTrue(pre_total_exps >= post_total_exps)

        # Test: Agents are inside their home ranges
        dist_from_hh = np.sqrt(
            ((to_wander[:, [A_X, A_Y]] - to_wander[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]]) ** 2).sum(axis=1)
        )
        self.assertTrue(np.all(dist_from_hh <= to_wander[:, A_HOME_RANGE]), msg="Agents wandering outside home range")

    def test_simulate_house(self):
        logger.info("Running test: test_simulate_house")
        n = self.camp.agents.shape[0]  # number of agents in the camp
        n_house = 100  # number of agents to send to household
        assert n >= n_house
        to_house = self.camp.agents[random.sample(range(n), n_house)]  # send agents to households

        # Values before execution
        pre_total_susc = np.count_nonzero(to_house[:, A_DISEASE] == INF_SUSCEPTIBLE)
        pre_total_exps = np.count_nonzero(to_house[:, A_DISEASE] == INF_EXPOSED)

        # Perform simulation
        to_house, num_new_inf = Moria.simulate_households(to_house, self.camp.params.prob_spread_house)

        # Values after execution
        post_total_susc = np.count_nonzero(to_house[:, A_DISEASE] == INF_SUSCEPTIBLE)
        post_total_exps = np.count_nonzero(to_house[:, A_DISEASE] == INF_EXPOSED)

        # Test: Basic tests
        # Number of new infections must have bounds
        self.assertTrue(0 <= num_new_inf <= n_house)
        # Number of agents returned must match number of input agents
        self.assertTrue(to_house.shape[0] == n_house)
        # Simulate function could only do transmission of susceptible->exposed
        self.assertTrue(pre_total_susc == (post_total_susc + num_new_inf))
        # Number of exposed agents should not decrease
        self.assertTrue(pre_total_exps >= post_total_exps)

        # Test: Agents are inside their households
        dist_from_hh = np.sqrt(
            ((to_house[:, [A_X, A_Y]] - to_house[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]]) ** 2).sum(axis=1)
        )
        self.assertTrue(np.all(dist_from_hh <= SMALL_ERROR), msg="Invalid agent co-ordinates after going to household")
