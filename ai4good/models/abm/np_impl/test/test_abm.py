import unittest

from ai4good.runner.facade import Facade
from ai4good.models.abm.np_impl.moria import *
from ai4good.models.model_registry import create_params_abm
from ai4good.utils.logger_util import get_logger

logger = get_logger(__name__)

"""
   To run the unit tests, run following command from project root:
   > python -m unittest discover -s ai4good/models/abm/np_impl/test
"""


def get_params(_profile='BaselineHTHI'):
    _model = 'agent-based-model'
    camp = 'Moria'
    overrides = '{"numberOfIterations": 1, "nProcesses": 1}'
    facade = Facade.simple()
    params = create_params_abm(facade.ps, _model, _profile, camp, overrides)
    return params


class OpTester(unittest.TestCase):

    def test_position_blocks(self):
        logger.info("Running test: test_position_blocks")

        camp_size = 100
        euclidean = lambda p1, p2: ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5  # helper function for distance

        # Base test when grid size is 1
        out = OptimizedOps.position_blocks(1, camp_size)
        self.assertTrue(out.shape[0] == 1)
        self.assertTrue(out[0, :].tolist() == [camp_size/2, camp_size/2])

        # Complex test when grid size is > 1
        grid_size = 3
        out = OptimizedOps.position_blocks(grid_size, camp_size)
        self.assertTrue(out.shape[0] == (grid_size * grid_size))

        out = out.reshape((grid_size, grid_size, 2))
        d = euclidean(out[0, 0, :], out[0, 1, :])  # distance between two slots horizontally

        # Assert that each consecutively horizontal or vertical slots are equidistant
        for i in range(grid_size):
            for j in range(1, grid_size):
                d_ = euclidean(out[i, j-1, :], out[i, j, :])
                self.assertAlmostEqual(d_, d)
        for j in range(grid_size):
            for i in range(1, grid_size):
                d_ = euclidean(out[i-1, j, :], out[i, j, :])
                self.assertAlmostEqual(d_, d)

    def test_find_nearest(self):
        logger.info("Running test: test_find_nearest")

        # Test on custom data
        pos = np.array([0, 0])
        others = np.array([[0.5, 0], [0, 1], [-2, 0], [0, -3]])

        # without any condn, (0.5,0) is closest to (0,0)
        min_idx, min_val = OptimizedOps.find_nearest(pos, others)
        self.assertTrue(min_idx == 0)
        self.assertTrue(min_val == 0.5 ** 2)  # find_nearest returns min distance ^ 2 instead of min distance

        # with condn = [False, True, True, True], nearest point would be (0,1) since for (0.5,0) condition is False
        min_idx, min_val = OptimizedOps.find_nearest(pos, others, condn=np.array([0, 1, 1, 1]))
        self.assertTrue(min_idx == 1)
        self.assertTrue(min_val == 1.0 ** 2)

    def test_showing_symptoms(self):
        logger.info("Running test: test_showing_symptoms")

        # Only symptomatic, mild and severe agents show symptoms
        inp = np.array([INF_SUSCEPTIBLE, INF_EXPOSED, INF_PRESYMPTOMATIC, INF_SYMPTOMATIC, INF_MILD, INF_SEVERE,
                        INF_ASYMPTOMATIC1, INF_ASYMPTOMATIC2, INF_RECOVERED, INF_DECEASED])
        exp = np.array([0, 0, 0, 1, 1, 1,
                        0, 0, 0, 0])
        out = OptimizedOps.showing_symptoms(inp)
        self.assertTrue(np.all(out == exp))

    def test_is_infected(self):
        logger.info("Running test: test_is_infected")

        # Presymptomatic, symptomatic, mild, severe, asymptomatic1, asymptomatic2 are infected
        inp = np.array([INF_SUSCEPTIBLE, INF_EXPOSED, INF_PRESYMPTOMATIC, INF_SYMPTOMATIC, INF_MILD, INF_SEVERE,
                        INF_ASYMPTOMATIC1, INF_ASYMPTOMATIC2, INF_RECOVERED, INF_DECEASED])
        exp = np.array([0, 0, 1, 1, 1, 1,
                        1, 1, 0, 0])
        out = OptimizedOps.is_infected(inp)
        self.assertTrue(np.all(out == exp))


class CampTester(unittest.TestCase):

    PROFILE = "BaselineHTHI"

    def setUp(self):
        # This function is called before every test* function
        param = get_params()
        self.camp = Moria(params=param, profile=CampTester.PROFILE)

        n = random.randint(5, 10)
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
        self.assertTrue(pre_total_exps <= post_total_exps)

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
        to_house, num_new_inf = Moria.simulate_households(to_house, self.camp.params.prob_spread_house,
                                                          ACTIVITY_HOUSEHOLD)

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
        self.assertTrue(pre_total_exps <= post_total_exps)

        # Test: Agents are inside their households
        dist_from_hh = np.sqrt(
            ((to_house[:, [A_X, A_Y]] - to_house[:, [A_HOUSEHOLD_X, A_HOUSEHOLD_Y]]) ** 2).sum(axis=1)
        )
        self.assertTrue(np.all(dist_from_hh <= SMALL_ERROR), msg="Invalid agent co-ordinates after going to household")

    def test_update_queues(self):
        logger.info("Running test: test_update_queues")

        # number of agents in the camp
        n = self.camp.agents.shape[0]

        for n_toilet in [0, 3, 5]:  # loop through test values of "number of agents to be put in the queue"
            # select few agents to be put in the queues
            to_toilet_ids = np.array(random.sample(range(n), n_toilet), dtype=np.int32)

            # First send the selected agents to the queue
            _ = self.camp.simulate_queues(to_toilet_ids, "toilet", self.camp.toilets)

            t_queues = self.camp.agents[to_toilet_ids, A_TOILET].astype(np.int32)  # Id of the queue of each agent
            t_queues_pos = self.camp.toilets[t_queues, :]  # Co-ordinates of the queue in the camp for each agent

            # Run tests: `to_toilet_ids` agents should be sent to toilet queue
            self.assertTrue(np.all(self.camp.agents[to_toilet_ids, A_ACTIVITY] == ACTIVITY_TOILET),
                            "Some agents didn't visit queue")
            self.assertTrue(np.all(self.camp.agents[to_toilet_ids][:, [A_X, A_Y]] == t_queues_pos),
                            "Agent's position didn't update properly")
            self.assertTrue(np.all(self.camp.agents[to_toilet_ids, A_ACTIVITY_BEFORE_QUEUE] != -1),
                            "A_ACTIVITY_BEFORE_QUEUE not updating properly")

            # Now, update the queues (de-queue everyone)
            self.camp.update_queues(1.0)
            # Run test: no one should be still in queue
            num_in_toilet_queue = np.count_nonzero(self.camp.agents[:, A_ACTIVITY] == ACTIVITY_TOILET)
            self.assertTrue(num_in_toilet_queue == 0, "After dequeue everyone, still people's position shows in queue")
