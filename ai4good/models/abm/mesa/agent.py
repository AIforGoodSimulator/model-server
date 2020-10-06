import random
import numpy as np
from mesa import Agent

from ai4good.models.abm.mesa.model import Camp
from ai4good.models.abm.mesa.helper import PersonHelper
from ai4good.models.abm.mesa.common import DiseaseStage, Route
from ai4good.models.abm.mesa.utils import get_incubation_period


class Person(Agent, PersonHelper):
    """
    Modelling a person living in the camp
    """

    def __init__(self, unique_id: int, model: Camp):
        super().__init__(unique_id, model)
        self.model = model

        self.day_counter = 0
        self.disease_state = model.agents_disease_states[unique_id]  # initial disease state
        self.gender = model.agents_gender[unique_id]  # agent's gender
        self.age = model.agents_age[unique_id]  # agent's age
        self.pos = model.agents_pos[unique_id]  # agent's position (x, y) in the camp
        self.route = model.agents_route[unique_id]  # current route of the agent: household, food-line, toilet, wander

        # each person is allocated a household initially (iso-box or tent). This does not change during model simulation
        self.household_id = model.agents_households[unique_id]  # household id of the person (fixed)

        # for movement, household center and home range are required
        self.household_center = self.model.households[self.household_id, 1:]  # center co-ordinate of household
        self.home_range = model.agents_home_ranges[unique_id]  # radius of circle centered at household for movement

        # calculate if asymptomatic
        # All children under the age of 16 become asymptomatic (ref), and others become asymptomatic
        # with probability 0.178 (Mizumoto et al. 2020)
        self.is_asymptomatic = self.age < 16 or random.random() <= 0.178

        # calculate if high risk
        self.is_high_risk = self.age >= 80

        # number of days from exposure until symptoms appear
        self.incubation_period = get_incubation_period(self.model.people_count)

    def step(self) -> None:
        # simulate 1 day in the camp

        if self.disease_state not in [DiseaseStage.SUSCEPTIBLE, DiseaseStage.RECOVERED, DiseaseStage.DECEASED]:
            self.day_counter += 1

        if self.route == Route.QUARANTINED:
            return

        # 3 rounds of move and visiting toilet
        for _ in range(3):
            self.move()
            self.infection_dynamics()
            self.visit_toilet()
            self.infection_dynamics()

        # going to food line
        self.visit_foodline()
        self.infection_dynamics()

        # 3 rounds of move and visiting toilet
        for _ in range(3):
            self.move()
            self.infection_dynamics()
            self.visit_toilet()
            self.infection_dynamics()

        # update disease progress values
        self.disease_progression()

        # at the end of the day, send the individual back to his/her household
        self.route = Route.HOUSEHOLD
        self.pos = self.household_center

        # update the model data
        self.model.agents_disease_states[self.unique_id] = self.disease_state
        self.model.agents_pos[self.unique_id] = self.pos
        self.model.agents_route[self.unique_id] = self.route

    def move(self) -> None:
        # We assume that each individual occupies a circular home range centred on its household, and uses all
        # parts of its home range equally

        # We assume that only individuals without symptoms interact in their home ranges
        if self.is_showing_symptoms():
            return

        # set current route
        self.route = Route.WANDERING

        # set new position
        # this internal calls the helper class where the numba optimized code is executed
        self.pos = self._move(self.household_center, self.home_range)

    def visit_foodline(self, prob_attend=0.75) -> None:
        """
        Simulating agent's visit to foodline. This is not probabilistic/stochastic, but fixed i.e. 3 times per day

        Parameters
        ----------
            prob_attend: This is the probability that the agent will go to the foodline. Based on tucker model, people
                without symptoms visit foodline once per day on 3 out of 4 days. On other occasions, food is brought to
                that individual by another individual without additional interactions.
                Hence, we gave it value of 3/4
        """

        # We assume that only individuals without symptoms attend food lines
        if self.is_showing_symptoms():
            return

        # going to foodline has some probability defined to it as mentioned above
        if random.random() > prob_attend:
            return

        self.route = Route.FOOD_LINE  # change agent's route
        # no need to change `pos` since `route` is FOOD_LINE hence current position is redundant for infection dynamics
        self.model.foodline_queue.append(self.unique_id)  # add agent to food line

    def visit_toilet(self, prob_visit=0.3, toilet_proximity=Camp.CAMP_SIZE*0.02) -> None:
        """
        Simulating agent's visit to toilet

        Parameters
        ----------
            prob_visit: This is the probability that agent will go to the toilet.
                Based on tucker model description of Moria camp, each person visits toilet 3 times/day. We try to
                convert this description to probability
            toilet_proximity: An agent will go to the toilet which is less than `toilet_proximity` units far from it

        """

        if self.route != Route.WANDERING and self.route != Route.HOUSEHOLD:
            # assumption: person will visit toilet if wandering or inside household
            # TODO: how to model toilet visit during lockdown or when quarantined?
            return

        # first, check if agent will/want to go to toilet
        # this is calculated based on probability to go to the toilet
        if random.random() > prob_visit:
            return

        # second, check if there is a toilet nearby
        nearest_toilet_id, nearest_toilet_distance = self._find_nearest(self.pos, self.model.toilets)

        # third, check if toilet is in desired proximity
        if nearest_toilet_distance > toilet_proximity:
            return  # toilet is too far away

        # now, the agent wants to visit toilet and there is a toilet nearby, now change the position
        self.route = Route.TOILET
        self.pos = self.model.toilets[nearest_toilet_id]

    def infection_dynamics(self, ph=0.5):

        # Infections can be transmitted from infectious to susceptible individuals in four ways:
        # within households, at toilets, in the food line, or as individuals move about the camp

        if self.route == Route.HOUSEHOLD:
            """
            On each day, each infectious individual in a household infects each susceptible individual in that 
            household with probability ph. Thus, if individual i shares its household with hcid infectious individuals 
            on day d, then
                𝑝𝑖𝑑ℎ = 1 − (1 − 𝑝ℎ)^ℎ𝑐𝑖𝑑. (2)
            We set 𝑝ℎ = 0.5 in our baseline model.
            """

            # Calculate if agent will be infected BY other infectious people
            # if agent is already infected, don't do anything
            # TODO: verify
            if self.disease_state != DiseaseStage.SUSCEPTIBLE:
                return

            # We need to calculate the number of infectious people with whom agent shares a household
            # First, get the total number of people currently in household who are infected
            people = self.model.get_filter_array()
            infectious_household_ids = self.model.filter_agents(people, skip_agent_id=self.unique_id,
                                                                route=Route.HOUSEHOLD, household_id=self.household_id,
                                                                is_infected=1, has_symptoms=-1)
            h_cid = infectious_household_ids.shape[0]  # number of infectious households
            p_ih = 1.0 - (1.0 - ph) ** h_cid  # probability value based on the formula (2)

            # change state from susceptible to exposed with probability `p_ih`
            self._change_state(p_ih, DiseaseStage.EXPOSED)
            return

        if self.route == Route.WANDERING:
            return

        if self.route == Route.FOOD_LINE:
            return

        if self.route == Route.TOILET:
            return

        if self.route == Route.QUARANTINED:
            return

    def disease_progression(self) -> None:

        # exposed to presymptomatic
        # a person becomes presymptomatic after half of the incubation period is completed
        if self.disease_state == DiseaseStage.EXPOSED and self.day_counter > self.incubation_period/2.0:
            self.disease_state = DiseaseStage.PRESYMPTOMATIC
            return

        # presymptomatic to 1st asymptomatic
        # After the incubation period, the individual enters one of two states: “symptomatic” or “1st asymptomatic.
        if self.disease_state == DiseaseStage.PRESYMPTOMATIC and \
                self.day_counter >= self.incubation_period and \
                self.is_asymptomatic:
            self.disease_state = DiseaseStage.ASYMPTOMATIC1
            self.day_counter = 0
            return

        # presymptomatic to symptomatic
        # a person who is not asymptomatic becomes symptomatic after the incubation period ends
        if self.disease_state == DiseaseStage.PRESYMPTOMATIC and \
                self.day_counter >= self.incubation_period and \
                not self.is_asymptomatic:
            self.disease_state = DiseaseStage.SYMPTOMATIC
            self.day_counter = 0
            return

        # After 5 days, individuals pass from the symptomatic to the “mild” or “severe” states, with age- and
        # condition-dependent probabilities following Verity and colleagues (2020) and Tuite and colleagues (preprint).
        # Verity (low-risk)
        asp = np.array([0, .000408, .0104, .0343, .0425, .0816, .118, .166, .184])
        # Tuite (high-risk)
        aspc = np.array([.0101, .0209, .0410, .0642, .0721, .2173, .2483, .6921, .6987])
        age_slot = int(self.age/10)

        # symptomatic to mild
        if self.disease_state == DiseaseStage.SYMPTOMATIC and \
                self.day_counter >= 6 and random.random() < asp[age_slot]:
            self.disease_state = DiseaseStage.MILD
            return

        # symptomatic to severe
        if self.disease_state == DiseaseStage.SYMPTOMATIC and \
                self.day_counter >= 6 and self.is_high_risk and \
                random.random() < aspc[age_slot]:
            self.disease_state = DiseaseStage.SEVERE
            return

        # mild to recovered
        # 2nd asymptomatic to recovered
        # On each day, individuals in the mild or 2nd asymptomatic state pass to the recovered state with
        # probability 0.37 (Lui et al. 2020)
        if self.disease_state in [DiseaseStage.MILD, DiseaseStage.ASYMPTOMATIC2] and random.random() <= 0.37:
            self.disease_state = DiseaseStage.RECOVERED
            return

        # severe to recovered
        # individuals in the severe state pass to the recovered state with probability 0.071 (Cai et al., preprint)
        if self.disease_state == DiseaseStage.SEVERE and random.random() <= 0.071:
            self.disease_state = DiseaseStage.RECOVERED
            return

        # severe to deceased
        # TODO: missing in tucker model

        # 1st asymptomatic to 2nd asymptomatic
        # After 5 days, All individuals in the 1st asymptomatic state pass to the “2nd asymptomatic” state
        if self.disease_state == DiseaseStage.ASYMPTOMATIC1 and self.day_counter >= 6:
            self.disease_state = DiseaseStage.ASYMPTOMATIC2
            return

    def _change_state(self, prob: float, new_state: DiseaseStage):
        # change state of the agent with some probability
        if random.random() < prob:
            self.disease_state = new_state

    def is_infectious(self) -> bool:
        # returns `True` if a person can infect others
        return self.disease_state in (
            DiseaseStage.PRESYMPTOMATIC,
            DiseaseStage.SYMPTOMATIC, DiseaseStage.MILD, DiseaseStage.SEVERE,
            DiseaseStage.ASYMPTOMATIC1, DiseaseStage.ASYMPTOMATIC2
        )

    def is_showing_symptoms(self) -> bool:
        # returns `True` if a person is showing infection symptoms
        return self.disease_state in (
            DiseaseStage.SYMPTOMATIC, DiseaseStage.MILD, DiseaseStage.SEVERE
        )