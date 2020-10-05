import random
import numpy as np
from mesa import Agent

from ai4good.models.abm.mesa.model import Camp
from ai4good.models.abm.mesa.common import DiseaseStage
from ai4good.models.abm.mesa.utils import get_incubation_period


class Person(Agent):
    """
    Modelling people of the camp
    """

    def __init__(self, unique_id, model: Camp):
        super().__init__(unique_id, model)
        self.model = model

        self.day_counter = 0
        self.disease_state = model.agents_disease_states[unique_id]  # initial disease state
        self.gender = model.agents_gender[unique_id]  # agent's gender
        self.age = model.agents_age[unique_id]  # agent's age
        self.pos = model.agents_pos[unique_id]  # agent's position (x, y) in the camp
        self.route = model.agents_route[unique_id]  # current route of the agent: household, food-line, toilet, wander
        self.household_id = model.agents_households[unique_id]  # household id of the person (fixed)
        self.home_range = model.agents_home_ranges[unique_id]  # radius of circle centered at household for movement

        # calculate if asymptomatic
        # All children under the age of 16 become asymptomatic (ref), and others become asymptomatic
        # with probability 0.178 (Mizumoto et al. 2020)
        self.is_asymptomatic = self.age < 16 or random.random() <= 0.178

        # calculate if high risk
        self.is_high_risk = self.age >= 80

        # number of days from exposure until symptoms appear
        self.incubation_period = get_incubation_period(self.model.people_count)

    def step(self):
        # simulate 1 day in the camp

        if self.disease_state not in [DiseaseStage.SUSCEPTIBLE, DiseaseStage.RECOVERED, DiseaseStage.DECEASED]:
            self.day_counter += 1

        self.disease_progression()
        self._update_model()

    def _update_model(self):
        # update the model data
        self.model.agents_disease_states[self.unique_id] = self.disease_state
        self.model.agents_pos[self.unique_id] = self.pos

    def move(self):
        pass

    def infection_dynamics(self):

        # Infections can be transmitted from infectious to susceptible individuals in four ways:
        # within households, at toilets, in the food line, or as individuals move about the camp

        # infection spread by infectious person in household (p_h)
        if self.disease_state == DiseaseStage.SUSCEPTIBLE:
            # get number of infectious people in same household
            # num_infectious_hh = sum([1 for agent in self.model.schedule.agent_buffer()
            #                          if agent.is_infectious() and agent.
            #                          ])
            pass

    def disease_progression(self) -> None:

        # susceptible to exposed
        # if self.disease_state == DiseaseStage.SUSCEPTIBLE and self. day_counter <= self.incubation_period/2.0:
        #     self.disease_state = DiseaseStage.EXPOSED

        # exposed to presymptomatic
        # a person becomes presymptomatic after half of the incubation period is completed
        if self.disease_state == DiseaseStage.EXPOSED and self.day_counter > self.incubation_period/2.0:
            self.disease_state = DiseaseStage.PRESYMPTOMATIC

        # presymptomatic to 1st asymptomatic
        # After the incubation period, the individual enters one of two states: “symptomatic” or “1st asymptomatic.
        if self.disease_state == DiseaseStage.PRESYMPTOMATIC and \
                self.day_counter >= self.incubation_period and \
                self.is_asymptomatic:
            self.disease_state = DiseaseStage.ASYMPTOMATIC1
            self.day_counter = 0

        # presymptomatic to symptomatic
        # a person who is not asymptomatic becomes symptomatic after the incubation period ends
        if self.disease_state == DiseaseStage.PRESYMPTOMATIC and \
                self.day_counter >= self.incubation_period and \
                not self.is_asymptomatic:
            self.disease_state = DiseaseStage.SYMPTOMATIC
            self.day_counter = 0

        # After 5 days, individuals pass from the symptomatic to the “mild” or “severe” states, with age- and
        # condition-dependent probabilities following Verity and colleagues (2020) and Tuite and colleagues (preprint).
        # Verity (low-risk)
        asp = np.array([0, .000408, .0104, .0343, .0425, .0816, .118, .166, .184])
        # Tuite (high-risk)
        aspc = np.array([.0101, .0209, .0410, .0642, .0721, .2173, .2483, .6921, .6987])
        age_slot = self.get_age_slot()

        # symptomatic to mild
        if self.disease_state == DiseaseStage.SYMPTOMATIC and \
                self.day_counter >= 6 and random.random() < asp[age_slot]:
            self.disease_state = DiseaseStage.MILD

        # symptomatic to severe
        if self.disease_state == DiseaseStage.SYMPTOMATIC and \
                self.day_counter >= 6 and self.is_high_risk and \
                random.random() < aspc[age_slot]:
            self.disease_state = DiseaseStage.SEVERE

        # mild to recovered
        # 2nd asymptomatic to recovered
        # On each day, individuals in the mild or 2nd asymptomatic state pass to the recovered state with
        # probability 0.37 (Lui et al. 2020)
        if self.disease_state in [DiseaseStage.MILD, DiseaseStage.ASYMPTOMATIC2] and random.random() <= 0.37:
            self.disease_state = DiseaseStage.RECOVERED

        # severe to recovered
        # individuals in the severe state pass to the recovered state with probability 0.071 (Cai et al., preprint)
        if self.disease_state == DiseaseStage.SEVERE and random.random() <= 0.071:
            self.disease_state = DiseaseStage.RECOVERED

        # severe to deceased
        # TODO: missing in tucker model

        # 1st asymptomatic to 2nd asymptomatic
        # After 5 days, All individuals in the 1st asymptomatic state pass to the “2nd asymptomatic” state
        if self.disease_state == DiseaseStage.ASYMPTOMATIC1 and self.day_counter >= 6:
            self.disease_state = DiseaseStage.ASYMPTOMATIC2

    def _infection_spread_movement(self):
        # Returns the infection spread probability while person is wandering

        pass

    def is_infectious(self) -> bool:
        # returns `True` if a person can infect others
        return self.disease_state in (
            DiseaseStage.PRESYMPTOMATIC,
            DiseaseStage.SYMPTOMATIC, DiseaseStage.MILD, DiseaseStage.SEVERE,
            DiseaseStage.ASYMPTOMATIC1, DiseaseStage.ASYMPTOMATIC2
        )

    def get_age_slot(self) -> int:
        # return age slot. 0: 0-9, 1: 10-19, 2: 20-20, ...
        return int(self.age/10)
