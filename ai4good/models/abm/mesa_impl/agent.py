import random
import numpy as np
from mesa import Agent

from ai4good.models.abm.mesa_impl.common import *
from ai4good.models.abm.mesa_impl.helper import PersonHelper


class Person(Agent, PersonHelper):
    """
    Modelling a person living in the camp
    """
    
    def do_infection_dynamics_afterwards(func):
        # use this wrapper to always execute `infection_dynamics` after function `func` call
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.infection_dynamics()
        return wrapper

    def __init__(self, unique_id: int, model):
        super().__init__(unique_id, model)
        self.model = model

        self.day_counter = model.agents_day_counter[unique_id]  # initial day counter value
        self.disease_state = model.agents_disease_states[unique_id]  # initial disease state
        self.gender = model.agents_gender[unique_id]  # agent's gender
        self.age = model.agents_age[unique_id]  # agent's age
        self.pos = model.agents_pos[unique_id]  # agent's position (x, y) in the camp
        self.route = model.agents_route[unique_id]  # current route of the agent: household, food-line, toilet, wander
        self.ethnic_group = model.agents_ethnic_groups[unique_id]  # ethnic group to which agent belongs

        # each person is allocated a household initially (iso-box or tent). This does not change during model simulation
        self.household_id = model.agents_households[unique_id]  # household id of the person (fixed)

        # for movement, household center and home range are required
        self.household_center = self.model.households[self.household_id, 2:]  # center co-ordinate of household
        self.home_range = model.agents_home_ranges[unique_id]  # radius of circle centered at household for movement

        self.toilet_id = -1  # if agent is standing in queue for toilet, store the id of the toilet

        # id of the foodline to visit (nearest one)
        self.foodline_id = PersonHelper.find_nearest(self.household_center, self.model.foodlines)

        # calculate if asymptomatic
        # All children under the age of 16 become asymptomatic (ref), and others become asymptomatic
        # with probability 0.178 (`permanently_asymptomatic_cases`) (Mizumoto et al. 2020)
        self.is_asymptomatic = self.age < 16 or random.random() <= self.model.params.permanently_asymptomatic_cases

        # calculate if high risk
        self.is_high_risk = self.age >= 80

        # number of days from exposure until symptoms appear
        self.incubation_period = self.model.agents_incubation_periods[unique_id]

    def step(self) -> None:
        # An agent in the camp can do following activities at any given time based on their route

        # if a person is quarantined, they don't do any activities (toilet visits and food line visits during quarantine
        # are not modelled)
        if self.route == QUARANTINED:
            # a susceptible person in quarantine can still be infected by other people in his/her household
            self.infection_dynamics()
            return

        # `activity` variable is used to denote how much activities the agent does in a day
        # More the number of activities an agent performs, more is their movement
        # TODO: parameterize it? or some better alternative?
        activity = 2
        
        [self.move() for _ in range(activity)]
        self.visit_toilet()
        [self.move() for _ in range(activity)]
        self.visit_food_line(prob_visit=self.model.params.pct_food_visit)
        [self.move() for _ in range(activity)]
        self.visit_toilet()
        [self.move() for _ in range(activity)]
        self.visit_food_line(prob_visit=self.model.params.pct_food_visit)
        [self.move() for _ in range(activity)]
        self.visit_toilet()
        [self.move() for _ in range(activity)]
        self.visit_food_line(prob_visit=self.model.params.pct_food_visit)
        [self.move() for _ in range(activity)]
        self.visit_toilet()
        [self.move() for _ in range(activity)]
        self.goto_household(0.9)  # at the end of the day, agent goes back to his/her household with high probability

        # update the model data
        self.model.agents_day_counter[self.unique_id] = self.day_counter

    def advance(self) -> None:
        # things to do at the end of the day

        # 1. update disease state
        self.disease_progression()

        # 2. check if agent needs to be quarantined
        # An agent in the camp who is showing symptoms can be quarantined with some probability
        # The detected agent will be removed along with its household
        if self.route != QUARANTINED and self.is_showing_symptoms() and random.random() <= self.model.P_detect:
            # when detected, quarantine the complete household
            self.model.isolate_household(self.household_id)

        # 3. check if person in quarantine can come back to the camp
        # We assume that individuals are returned to the camp 7 days after they have recovered, or if they do not
        # become infected, 7 days after the last infected person in their household has recovered
        # TODO: Right now to remove agents from quarantine, we check if agent has shown no symptoms for some days.
        # TODO: Should the logic be instead: check if agent's state is SUSCEPTIBLE OR RECOVERED? This way, asymptomatic
        # TODO: and exposed agents will not be sent back to the camp
        if self.route == QUARANTINED and not self.is_showing_symptoms() and self.day_counter >= self.model.P_n:
            self.model.check_remove_from_isolation(self.household_id)

        # increase day counter to track number of days in a disease state
        if self.disease_state not in [SUSCEPTIBLE, RECOVERED, DECEASED]:
            self.day_counter += 1

    @do_infection_dynamics_afterwards
    def goto_household(self, prob):
        # The agent goes back to the household for rest or other activities which may involve interaction with other
        # people in the household.
        # Personal assumption: Going to the household has some probability linked to it. At the end of the day, the
        # agent will definitely go back to the household (i.e. `prob` ~ 1). At other instances in the day, one goes to
        # the household with the probability of 0.5
        if random.random() <= prob:
            self.set_route(HOUSEHOLD)
            self.set_pos(self.household_center)

    @do_infection_dynamics_afterwards
    def move(self) -> None:
        # We assume that each individual occupies a circular home range centred on its household, and uses all
        # parts of its home range equally

        # We assume that only individuals without symptoms interact in their home ranges
        # personal assumption: quarantined individuals will not wander
        if self.is_showing_symptoms() or self.route == QUARANTINED:
            return

        # set current route
        self.set_route(WANDERING)

        # set new position
        # this internal calls the helper class where the numba optimized code is executed
        self.set_pos(self._move(self.household_center, self.home_range))

    @do_infection_dynamics_afterwards
    def visit_food_line(self, prob_visit) -> None:
        """
        Simulating agent's visit to food line. This is not probabilistic/stochastic, but fixed i.e. 3 times per day

        Parameters
        ----------
            prob_visit: This is the probability that the agent will go to the food line. Based on tucker model, people
                without symptoms visit food line once per day on 3 out of 4 days. On other occasions, food is brought to
                that individual by another individual without additional interactions. Therefore, default value = 3/4
        """

        # We assume that only individuals without symptoms attend food lines
        # personal assumption: quarantined individuals will not go to food line
        if self.is_showing_symptoms() or self.route == QUARANTINED:
            return

        # going to food line has some probability defined to it as mentioned above
        if random.random() > prob_visit:
            return

        self.set_route(FOOD_LINE)  # change agent's route
        # no need to update `pos` since `route` is FOOD_LINE hence current position is redundant for infection dynamics

        # add the agent to the end of food line queue and store the position
        if self.foodline_id in self.model.foodline_queue:
            # if already people in food line, go to end of line
            self.model.foodline_queue[self.foodline_id].append(self.unique_id)
        else:
            # if line not formed yet, start the line
            self.model.foodline_queue[self.foodline_id] = [self.unique_id]

    @do_infection_dynamics_afterwards
    def visit_toilet(self, prob_visit=0.3, toilet_proximity=CAMP_SIZE*0.02) -> None:
        """
        Simulating agent's visit to toilet

        Parameters
        ----------
            prob_visit: This is the probability that agent will go to the toilet.
                Based on tucker model description of Moria camp, each person visits toilet 3 times/day. We try to
                convert this description to probability
            toilet_proximity: An agent will go to the toilet which is less than `toilet_proximity` units far from it

        """

        if self.route not in (WANDERING, HOUSEHOLD):
            # personal assumption: person will visit toilet if wandering or inside household
            # i.e. person will not visit toilet whilst in food line or when quarantined
            # TODO: how to model toilet visit during lockdown or when quarantined?
            # Possible answer: Agents still visit toilets in isolation but quarantine infection spread is not modelled
            return

        # First, check if agent will/want to go to toilet
        # This is calculated based on probability to go to the toilet
        if random.random() > prob_visit:
            return

        # Second, check if there is a toilet nearby
        nearest_toilet_id, nearest_toilet_distance = self.find_nearest(self.pos, self.model.toilets)

        # third, check if toilet is in desired proximity
        if nearest_toilet_distance > toilet_proximity:
            return  # toilet is too far away

        # Now, the agent wants to visit toilet and there is a toilet nearby, so change the position of the agent
        # Set current route as TOILET (used for infection dynamics)
        self.set_route(TOILET)

        # update toilet queue
        if nearest_toilet_id in self.model.toilets_queue:
            self.model.toilets_queue[nearest_toilet_id].append(self.unique_id)
        else:
            self.model.toilets_queue[nearest_toilet_id] = [self.unique_id]
        self.toilet_id = nearest_toilet_id  # update the current toilet id

    def infection_dynamics(self) -> None:
        # Infections can be transmitted from infectious to susceptible individuals in four ways:
        # within households, at toilets, in the food line, or as individuals move about the camp
        # This method will make susceptible people infectious

        # On each day, each infectious individual in a household infects each susceptible individual in that
        # household with probability ph
        # TODO: in csv file, ph=0.33 in baseline model. But tucker model says baseline value is 0.5. Verify this.
        ph = self.model.params.probability_infecting_person_in_household_per_day

        # if agent is already infected, don't do anything. Let `disease_progression` run its course.
        # TODO: verify
        if self.disease_state != SUSCEPTIBLE:
            return

        if self.route == HOUSEHOLD or self.route == QUARANTINED:
            """
            On each day, each infectious individual in a household infects each susceptible individual in that 
            household with probability ph. Thus, if individual i shares its household with hcid infectious individuals 
            on day d, then
                𝑝𝑖𝑑ℎ = 1 − (1 − 𝑝ℎ)^ℎ𝑐𝑖𝑑. (2)
            We set 𝑝ℎ = 0.5 in our baseline model.
            """

            # Calculate if agent will be infected BY other infectious people

            # We need to calculate the number of infectious people with whom agent shares a household
            # First, get the total number of people currently in household who are infected
            # create a filter array of agents
            people = self.model.get_filter_array()
            # filter people who are also inside household same as of current agent and who are infected
            infectious_household_ids = self.model.filter_agents(people, skip_agent_id=self.unique_id,
                                                                route=self.route, household_id=self.household_id,
                                                                is_infected=1)
            h_cid = infectious_household_ids.shape[0]  # number of infectious households
            p_ih = 1.0 - (1.0 - ph) ** h_cid  # probability value based on the formula (2)

            # change state from susceptible to exposed with probability `p_ih`
            self._change_state(p_ih, EXPOSED)
            return

        if self.route == WANDERING:
            # TODO
            return

        if self.route == FOOD_LINE:
            """
            If an individual attends the food line, it interacts with the individual in front of and the individual
            behind it in the line.
            """

            # Thus, catching infection is dependent on if the people in front and back are infectious or not
            n = 0

            # Find the position of the agent in the food line
            try:
                foodline_queue_idx = self.model.foodline_queue[self.foodline_id].index(self.unique_id)
            except ValueError:
                # Code should never reach here, but if it does, do nothing.
                return

            # check infection for person in front of the agent
            if foodline_queue_idx > 0 and \
                    self._is_showing_symptoms(
                        self.model.agents_disease_states[
                            self.model.foodline_queue[self.foodline_id][foodline_queue_idx-1]
                        ]
                    ):
                n += 1

            # check infection for person behind the agent
            if foodline_queue_idx < (len(self.model.foodline_queue[self.foodline_id])-1) and \
                    self._is_showing_symptoms(
                        self.model.agents_disease_states[
                            self.model.foodline_queue[self.foodline_id][foodline_queue_idx+1]
                        ]
                    ):
                n += 1

            # find probability that infection will be spread
            p_if = 1.0 - (1.0 - self.model.Pa) ** n

            # change state from susceptible to exposed with probability `p_if`
            self._change_state(p_if, EXPOSED)

            # Finally, remove agent from the food line queue if he/she is at front of the queue
            if foodline_queue_idx == 0:  # check if agent is at the front of the queue
                self.set_route(WANDERING)  # send agent back to wandering
                self.model.foodline_queue[self.foodline_id].remove(self.unique_id)  # remove agent from the queue

            return

        if self.route == TOILET:
            """
            On each visit, it interacts with the individual in front of it and the individual behind it in the 
            toilet line
            """

            # Thus, catching infection is dependent on if the people in front and back are infectious or not
            n = 0

            # Get the position of the agent in the toilet queue
            try:
                toilet_queue_idx = self.model.toilets_queue[self.toilet_id].index(self.unique_id)
            except ValueError:
                # Code should never reach here. But if it does, do nothing
                return

            # check infection for person in front of the agent
            if toilet_queue_idx > 0 and \
                    self._is_showing_symptoms(
                        self.model.agents_disease_states[
                            self.model.toilets_queue[self.toilet_id][toilet_queue_idx - 1]
                        ]
                    ):
                n += 1

            # check infection for person behind the agent
            if toilet_queue_idx < (len(self.model.toilets_queue[self.toilet_id]) - 1) and \
                    self._is_showing_symptoms(
                        self.model.agents_disease_states[
                            self.model.toilets_queue[self.toilet_id][toilet_queue_idx + 1]
                        ]
                    ):
                n += 1

            # find probability that infection will be spread
            p_it = 1.0 - (1.0 - self.model.Pa) ** n

            # change state from susceptible to exposed with probability `p_it`
            self._change_state(p_it, EXPOSED)

            # Finally, remove agent from the toilet queue if he/she is at front of the queue
            if toilet_queue_idx == 0:  # check if agent is at the front of the queue
                self.set_route(WANDERING)  # send agent back to wandering
                self.model.toilets_queue[self.toilet_id].remove(self.unique_id)  # remove agent from the queue

            return

        if self.route == HOSPITALIZED:
            # TODO
            return

    def disease_progression(self) -> None:

        # exposed to presymptomatic
        # a person becomes presymptomatic after half of the incubation period is completed
        if self.disease_state == EXPOSED and self.day_counter > self.incubation_period/2.0:
            self.set_disease_state(PRESYMPTOMATIC)
            return

        # presymptomatic to 1st asymptomatic
        # After the incubation period, the individual enters one of two states: “symptomatic” or “1st asymptomatic.
        if self.disease_state == PRESYMPTOMATIC and \
                self.day_counter >= self.incubation_period and \
                self.is_asymptomatic:
            self.set_disease_state(ASYMPTOMATIC1)
            self.day_counter = 0
            return

        # presymptomatic to symptomatic
        # a person who is not asymptomatic becomes symptomatic after the incubation period ends
        if self.disease_state == PRESYMPTOMATIC and \
                self.day_counter >= self.incubation_period and \
                not self.is_asymptomatic:
            self.set_disease_state(SYMPTOMATIC)
            self.day_counter = 0
            return

        # After 5 days, individuals pass from the symptomatic to the “mild” or “severe” states, with age- and
        # condition-dependent probabilities following Verity and colleagues (2020) and Tuite and colleagues (preprint).
        # Verity (low-risk) : probability values for each age slot [0-10, 10-20, ...90+]
        asp = np.array([0, .000408, .0104, .0343, .0425, .0816, .118, .166, .184])
        # Tuite (high-risk) : probability values for each age slot [0-10, 10-20, ...90+]
        aspc = np.array([.0101, .0209, .0410, .0642, .0721, .2173, .2483, .6921, .6987])
        # calculate age slot from age
        age_slot = int(self.age/10)

        # symptomatic to mild
        if self.disease_state == SYMPTOMATIC and \
                self.day_counter >= 6 and not self.is_high_risk and random.random() < asp[age_slot]:
            self.set_disease_state(MILD)
            return

        # symptomatic to severe
        if self.disease_state == SYMPTOMATIC and \
                self.day_counter >= 6 and self.is_high_risk and random.random() < aspc[age_slot]:
            self.set_disease_state(SEVERE)

            # TODO: tucker model does not cover hospitalization of agents, should we keep it?
            # TODO: change step function for hospitalized agents
            self.set_route(HOSPITALIZED)  # hospitalize agent when condition gets severe
            return

        # mild to recovered
        # 2nd asymptomatic to recovered
        # On each day, individuals in the mild or 2nd asymptomatic state pass to the recovered state with
        # probability 0.37 (Lui et al. 2020)
        if self.disease_state in [MILD, ASYMPTOMATIC2] and random.random() <= 0.37:
            self.set_disease_state(RECOVERED)
            return

        # severe to recovered
        # individuals in the severe state pass to the recovered state with probability 0.071 (Cai et al., preprint)
        if self.disease_state == SEVERE and random.random() <= 0.071:
            self.set_disease_state(RECOVERED)
            return

        # severe to deceased
        # TODO: missing in tucker model

        # 1st asymptomatic to 2nd asymptomatic
        # After 5 days, All individuals in the 1st asymptomatic state pass to the “2nd asymptomatic” state
        if self.disease_state == ASYMPTOMATIC1 and self.day_counter >= 6:
            self.set_disease_state(ASYMPTOMATIC2)
            return

    def isolate(self):
        # Isolate the agent (detected to have infections or someone in his/her household has an infection)

        # change route and reset day counter
        self.set_route(QUARANTINED)
        self.day_counter = 0

    def _change_state(self, prob: float, new_state: int):
        # change state of the agent with some probability
        if random.random() < prob:
            self.set_disease_state(new_state)

    def is_showing_symptoms(self) -> bool:
        # returns `True` if a person is showing infection symptoms
        return self._is_showing_symptoms(self.disease_state)

    def set_disease_state(self, val):
        self.disease_state = val
        self.model.agents_disease_states[self.unique_id] = val

    def set_pos(self, val):
        self.pos = val
        self.model.agents_pos[self.unique_id, :] = val

    def set_route(self, val):
        self.route = val
        self.model.agents_route[self.unique_id] = val

    do_infection_dynamics_afterwards = staticmethod(do_infection_dynamics_afterwards)