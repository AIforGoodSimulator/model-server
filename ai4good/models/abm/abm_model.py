from typeguard import typechecked
from ai4good.models.model import Model, ModelResult
from ai4good.params.param_store import ParamStore
from ai4good.models.abm.initialise_parameters import Parameters
# from ai4good.webapp.cm_model_report_utils import *
import logging
from . import abm
import numpy as np
import pandas as pd
import math


@typechecked
class ABM(Model):

    ID = 'agent-based-model'

    def __init__(self, ps: ParamStore):
        Model.__init__(self, ps)

    def id(self) -> str:
        return self.ID

    def result_id(self, p: Parameters) -> str:
        return p.sha1_hash()

    def run(self, p: Parameters) -> ModelResult:

        # run multiple steps of the simulation
        for i in range(p.number_of_steps):
            self.step(p, i)

        # placeholders for the report
        standard_sol = [{'t': range(p.number_of_steps)}]
        perc = [0] * p.number_of_steps
        percentiles = [perc, perc, perc, perc, perc]
        config_dict = []
        [config_dict.append(
            dict(
                beta=0,
                latentRate=0,
                removalRate=0,
                hospRate=0,
                deathRateICU=0,
                deathRateNoIcu=0
            )) for _ in range(p.number_of_steps)]

        report_raw = [[0]]
        prevalence_age = pd.DataFrame([[0]])
        prevalence_all = pd.DataFrame([[0]])
        cumulative_all = pd.DataFrame([[0]])
        cumulative_age = pd.DataFrame([[0]])

        states = ['exposed_tl', 'presymptomatic_tl', 'symptomatic_tl', 'mild_tl', 'severe_tl', 'recovered_tl',
                  'qua_susceptible_tl', 'qua_exposed_tl', 'qua_presymptomatic_tl', 'qua_symptomatic_tl', 'qua_mild_tl',
                  'qua_severe_tl', 'qua_recovered_tl']
        # disease_state_tracker_plot = go.Figure()

        mr = ModelResult(self.result_id(p), {
            'standard_sol': standard_sol,
            'percentiles': percentiles,
            'config_dict': config_dict,
            'params': p,
            'report': report_raw,
            'track_states_df': p.track_states,
            'multiple_categories_to_plot': states,
            'prevalence_all': prevalence_all,
            'cumulative_all': cumulative_all,
            'cumulative_age': cumulative_age
        })

        # at end of simulation, return model result
        return mr

    @staticmethod
    def step(p: Parameters, t):
        """
        Execute abm simulation step

        Parameters
        ----------
            p: `Parameters` object for simulation
            t: Step number

        Returns
        -------
            None

        """
        p.track_states[t, :] = np.bincount(p.population[:, 1].astype(int), minlength=14)

        # finish simulation by checking states
        # TODO: currently, if all people are either susceptible/recovered then we finish simulation
        # TODO: should it not include qua_susceptible/qua_recovered states as well?
        if abm.epidemic_finish(np.concatenate((p.track_states[t, 1:6], p.track_states[t, 7:p.number_of_states])), t):
            return

        # The probability that a person's disease state will change from mild->recovered (0.37)
        # Liu et al 2020 The Lancet.
        p.mild_rec = np.random.uniform(0, 1, p.total_population) > math.exp(0.2 * math.log(0.1))

        # The probability that a person's disease state will change from severe->recovered (0.071)
        # Cai et al.
        p.sev_rec = np.random.uniform(0, 1, p.total_population) > math.exp(math.log(63 / 153) / 12)

        # Get random numbers to determine health states
        p.pick_sick = np.random.uniform(0, 1, p.total_population)

        if p.ACTIVATE_INTERVENTION and t != 0:
            p.iat1 = t
            p.ACTIVATE_INTERVENTION = False
            p.smaller_movement_radius = 0.001
            p.transmission_reduction = 0.25  # TODO: why is this here?
            p.foodpoints_location, p.foodpoints_numbers, p.foodpoints_sharing = abm.position_foodline(
                p.households_location, p.foodline_blocks[0], p.foodline_blocks[1])
            p.local_interaction_space = abm.interaction_neighbours_fast(p.households_location,
                                                                        p.smaller_movement_radius,
                                                                        p.larger_movement_radius,
                                                                        p.overlapping_rages_radius,
                                                                        p.ethnical_corellations)
            p.viol_rate = 0.05
            p.population[:, 8] = np.where(np.random.rand(p.total_population) < p.viol_rate, 1, 0)

        # increase day count for all non-susceptible people
        # TODO: should we exclude qua_susceptible (index=7) as well here along with susceptible (index=0) ?
        p.population[np.where(p.population[:, 1] > 0), 3] += 1

        # update disease states of people not in quarantine
        # also update the population matrix and total hospitalized count
        p.population, p.total_number_of_hospitalized = abm.disease_state_update(
            p.population,
            p.mild_rec,
            p.sev_rec,
            p.pick_sick,
            p.total_number_of_hospitalized
        )

        # update disease states of people in quarantine
        # also update the population matrix and total hospitalized count
        p.population, p.total_number_of_hospitalized = abm.disease_state_update(
            p.population,
            p.mild_rec,
            p.sev_rec,
            p.pick_sick,
            p.total_number_of_hospitalized,
            quarantined=True
        )

        p.population = abm.assign_new_infections(p.population,
                                                 p.toilets_sharing,
                                                 p.foodpoints_sharing,
                                                 p.num_toilet_visit,
                                                 p.num_toilet_contact,
                                                 p.num_food_visit,
                                                 p.num_food_contact,
                                                 p.pct_food_visit,
                                                 p.transmission_reduction,
                                                 p.local_interaction_space,
                                                 p.probability_infecting_person_in_household_per_day,
                                                 p.probability_infecting_person_in_foodline_per_day,
                                                 p.probability_infecting_person_in_toilet_per_day,
                                                 p.probability_infecting_person_in_moving_per_day)

        p.population = abm.move_hhl_quarantine(p.population, p.probability_spotting_symptoms_per_day)

        p.quarantine_back = np.logical_and(p.population[:, 1] == 13, p.population[:, 3] >= p.clearday)
        p.population[p.quarantine_back, 1] = 6
