#from ai4good.models.cm.initialise_parameters import params, control_data, categories, calculated_categories, change_in_categories

import logging
import statistics
import warnings
from math import ceil, floor

import dask
import numpy as np
from dask.diagnostics import ProgressBar
import sdeint

from ai4good.models.cm.initialise_parameters import Parameters

AGE_SEP = ': '  # separate compartment and age in column name

def timing_function(t,time_vector):
    for ii in range(ceil(len(time_vector)/2)):
        if t>=time_vector[2*ii] and t<time_vector[2*ii+1]:
            return True
    # if wasn't in any of these time interval
    return False

##
# -----------------------------------------------------------------------------------
##


class SEIRSDESolver:

    def __init__(self, params: Parameters):
        self.params = params
        population = self.params.population
        population_frame = self.params.population_frame
        control_dict = self.params.control_dict

        seirvars = np.zeros((self.params.number_compartments, 1));

        seirvars[self.params.categories['I']['index'], 0] = 1 / population  # sympt
        seirvars[self.params.categories['A']['index'], 0] = 1 / population  # asympt

        seirvars[self.params.categories['S']['index'], 0] = 1-seirvars.sum()


        # initial conditions
        #y0 = y1.T.reshape(self.params.number_compartments*age_categories)


        self.infection_matrix = self.params.infection_matrix
        self.age_categories = int(population_frame.shape[0])
        self.symptomatic_prob = np.asarray(population_frame.p_symptomatic)
        self.hospital_prob = np.asarray(population_frame.p_hospitalised)
        self.critical_prob = np.asarray(population_frame.p_critical)
        self.beta = self.params.beta_list[1]
        self.beta_sigma = self.params.beta_sigma_list[1]
        self.latent_rate = self.params.latent_rate
        self.latent_rate_sigma = self.params.latent_rate_sigma
        self.removal_rate = self.params.removal_rate
        self.removal_rate_sigma = self.params.removal_rate_sigma
        self.hosp_rate = self.params.hosp_rate
        self.hosp_rate_sigma = self.params.hosp_period_sigma
        self.death_rate_icu = self.params.death_rate_with_ICU
        self.death_rate_icu_sigma = self.params.death_rate_with_icu_sigma
        self.death_rate_no_icu = self.params.death_rate  # more params
        self.death_rate_no_icu_sigma = self.params.death_rate_sigma
        self.stoc_vars_num = 6
        self.index_beta = 0
        self.index_latent = 1
        self.index_removal = 2
        self.index_hosp = 3
        self.index_death_icu = 4
        self.index_death_no_icu = 5

        self.better_hygiene = control_dict['better_hygiene']
        self.remove_symptomatic = control_dict['remove_symptomatic']
        self.remove_high_risk = control_dict['remove_high_risk']
        self.icu_capacity = control_dict['ICU_capacity']

        population_vector = population_frame.Population_structure.to_numpy()
        y1 = np.dot(seirvars, population_vector.reshape(1, self.age_categories)/100)
        self.y0 = y1.T.reshape(self.params.number_compartments*self.age_categories)
        self.sigma = np.full((self.y0.shape[0], 1), 0.1)
        self.driftOnly = False;
        self.zero_diffusion = np.zeros((self.y0.shape[0], self.stoc_vars_num))
        # self.iteration=0


    def sde_drift(self, y, t):

        ##
        params = self.params

        y2d = y.reshape(self.age_categories, self.params.number_compartments).T
        dydt2d = np.zeros(y2d.shape)

        index_I = params.categories['I']['index']
        index_S = params.categories['S']['index']
        index_E = params.categories['E']['index']
        index_C = params.categories['C']['index']
        index_Q = params.categories['Q']['index']
        index_U = params.categories['U']['index']
        index_H = params.categories['H']['index']
        index_A = params.categories['A']['index']

        I_vec = y2d[index_I, :]
        H_vec = y2d[index_H, :]
        A_vec = y2d[index_A, :]
        C_vec = y2d[index_C, :]
        S_vec = y2d[index_S, :]

        E_latent = self.latent_rate * y2d[index_E, :]
        I_removed = self.removal_rate * I_vec
        Q_quarantined = params.quarant_rate * y2d[index_Q, :]

        total_I = sum(I_vec)
        total_H = sum(H_vec)

        # better hygiene
        if timing_function(t, self.better_hygiene['timing']):  # control in place
            control_factor = self.better_hygiene['value']
        else:
            control_factor = 1

        # removing symptomatic individuals
        if timing_function(t, self.remove_symptomatic['timing']):  # control in place
            remove_symptomatic_rate = min(total_I, self.remove_symptomatic['rate'])  # if total_I too small then can't take this many off site at once
        else:
            remove_symptomatic_rate = 0

        first_high_risk_category_n = self.age_categories - self.remove_high_risk['n_categories_removed']
        S_removal = sum(y2d[index_S, first_high_risk_category_n:])  # add all old people to remove
        # removing symptomatic individuals
        # these are put into Q ('quarantine');
        quarantine_sicks = (remove_symptomatic_rate / total_I) * I_vec  # no age bias in who is moved

        # removing susceptible high risk individuals
        # these are moved into O ('offsite')
        high_risk_people_removal_rates = np.zeros(self.age_categories);
        if timing_function(t, self.remove_high_risk['timing']):
            high_risk_people_removal_rates[first_high_risk_category_n:] = min(self.remove_high_risk['rate'],
                                      S_removal)  # only removing high risk (within time control window). Can't remove more than we have

        # ICU capacity
        if total_H > 0: # can't divide by 0
            hospitalized_on_icu = self.icu_capacity['value'] / total_H * H_vec
            # ICU beds allocated on a first come, first served basis based on the numbers in hospital
        else:
            hospitalized_on_icu = np.full(self.age_categories, self.icu_capacity['value'])

        # ODE system:
        # S
        infection_I = np.dot(self.infection_matrix, I_vec)
        infection_A = np.dot(self.infection_matrix, A_vec)
        infection_total = (infection_I + params.AsymptInfectiousFactor * infection_A)
        offsite = high_risk_people_removal_rates / S_removal * S_vec
        dydt2d[index_S, :] = (- control_factor * self.beta * S_vec * infection_total - offsite)

        # E
        dydt2d[index_E, :] = (control_factor * self.beta * S_vec * infection_total - E_latent)

        # I
        dydt2d[index_I, :] = ((1 - self.symptomatic_prob) * E_latent - I_removed- quarantine_sicks)

        # A
        A_removed = self.removal_rate * A_vec
        dydt2d[index_A, :] = (self.symptomatic_prob * E_latent- A_removed)

        # H
        dydt2d[index_H, :] = (self.hospital_prob * I_removed - self.hosp_rate * H_vec
                              + self.death_rate_icu * (1 - params.death_prob_with_ICU) *
                              np.minimum(C_vec, hospitalized_on_icu)  # recovered from ICU
                              + self.hospital_prob * Q_quarantined  # proportion of removed people who were hospitalised once returned
                              )

        # Critical care (ICU)
        deaths_on_icu = self.death_rate_icu * C_vec
        without_deaths_on_icu = C_vec - deaths_on_icu
        needing_care = self.hosp_rate * self.critical_prob * H_vec # number needing care

        # number who get icu care (these entered category C)
        icu_cared = np.minimum(needing_care, hospitalized_on_icu - without_deaths_on_icu)

        # amount entering is minimum of: amount of beds available**/number needing it
        # **including those that will be made available by new deaths
        # without ICU treatment
        dydt2d[index_C, :] = (icu_cared - deaths_on_icu)
        # self.iteration = self.iteration + 1
        # if np.max(dydt2d[index_C, :]) > 10e12:
        #     print("iteration:" + str(self.iteration) + " dydt2d[index_C, :]=" + str(dydt2d[index_C, :]))
        # Uncared - no ICU
        deaths_without_icu = self.death_rate_no_icu * y2d[index_U, :] # died without ICU treatment (all cases that don't get treatment die)
        dydt2d[index_U, :] = (needing_care - icu_cared - deaths_without_icu)  # without ICU treatment

        # R
        # proportion of removed people who recovered once returned
        dydt2d[params.categories['R']['index'], :] = (
                (1 - self.hospital_prob) * I_removed + A_removed + self.hosp_rate * (1 - self.critical_prob) * H_vec + (1 - self.hospital_prob) * Q_quarantined
        )

        # D
        dydt2d[params.categories['D']['index'], :] = (
                    deaths_without_icu + params.death_prob_with_ICU * deaths_on_icu # died despite attempted ICU treatment
        )
        # O
        dydt2d[params.categories['O']['index'], :] = offsite

        # Q
        dydt2d[index_Q, :] = quarantine_sicks - Q_quarantined

        return dydt2d.T.reshape(y.shape)

    def sde_diffusion(self, y, t):
        if self.driftOnly:
            return self.zero_diffusion;
        ##
        params = self.params

        y2d = y.reshape(self.age_categories, self.params.number_compartments).T
        dgdt3d = np.zeros((self.stoc_vars_num, self.params.number_compartments, self.age_categories))

        index_I = params.categories['I']['index']
        index_S = params.categories['S']['index']
        index_E = params.categories['E']['index']
        index_C = params.categories['C']['index']
        index_Q = params.categories['Q']['index']
        index_U = params.categories['U']['index']
        index_H = params.categories['H']['index']
        index_A = params.categories['A']['index']
        index_R = params.categories['R']['index']
        index_D = params.categories['D']['index']

        I_vec = y2d[index_I, :]
        H_vec = y2d[index_H, :]
        A_vec = y2d[index_A, :]
        C_vec = y2d[index_C, :]
        S_vec = y2d[index_S, :]

        E_latent = self.latent_rate_sigma * y2d[index_E, :]
        I_removed = self.removal_rate_sigma * I_vec

        total_I = sum(I_vec)
        total_H = sum(H_vec)

        # better hygiene
        if timing_function(t, self.better_hygiene['timing']):  # control in place
            control_factor = self.better_hygiene['value']
        else:
            control_factor = 1

        first_high_risk_category_n = self.age_categories - self.remove_high_risk['n_categories_removed']
        S_removal = sum(y2d[index_S, first_high_risk_category_n:])  # add all old people to remove
        # removing symptomatic individuals
        # these are put into Q ('quarantine');

        # removing susceptible high risk individuals
        # these are moved into O ('offsite')
        high_risk_people_removal_rates = np.zeros(self.age_categories);
        if timing_function(t, self.remove_high_risk['timing']):
            high_risk_people_removal_rates[first_high_risk_category_n:] = min(self.remove_high_risk['rate'],
                                      S_removal)  # only removing high risk (within time control window). Can't remove more than we have

        # ICU capacity
        if total_H > 0: # can't divide by 0
            hospitalized_on_icu = self.icu_capacity['value'] / total_H * H_vec
            # ICU beds allocated on a first come, first served basis based on the numbers in hospital
        else:
            hospitalized_on_icu = np.full(self.age_categories, self.icu_capacity['value'])

        # SDE system Wiener part:
        # S
        infection_I = np.dot(self.infection_matrix, I_vec)
        infection_A = np.dot(self.infection_matrix, A_vec)
        infection_total = (infection_I + params.AsymptInfectiousFactor * infection_A)

        dgdt3d[self.index_beta, index_S, :] = -control_factor * self.beta_sigma * S_vec * infection_total

        # E
        dgdt3d[self.index_beta, index_E, :] = control_factor * self.beta_sigma * S_vec * infection_total
        dgdt3d[self.index_latent, index_E, :] = -E_latent


        # I
        dgdt3d[self.index_latent, index_I, :] = (1 - self.symptomatic_prob) * E_latent
        dgdt3d[self.index_removal, index_I, :] = -I_removed

        # A
        A_removed = self.removal_rate_sigma * A_vec
        dgdt3d[self.index_latent, index_A, :] = self.symptomatic_prob * E_latent
        dgdt3d[self.index_removal, index_A, :] = -A_removed

        # H
        dgdt3d[self.index_removal, index_H, :] = self.hospital_prob * I_removed
        dgdt3d[self.index_hosp, index_H, :] = self.hosp_rate_sigma * H_vec
        dgdt3d[self.index_death_icu, index_H, :] = self.death_rate_icu_sigma * (1 - params.death_prob_with_ICU) * np.minimum(C_vec, hospitalized_on_icu)  # recovered from ICU

        # Critical care (ICU)
        deaths_on_icu = self.death_rate_icu_sigma * C_vec
        without_deaths_on_icu = C_vec - deaths_on_icu
        needing_care = self.hosp_rate_sigma * self.critical_prob * H_vec # number needing care

        needing_care_less_then_died_on_icu = needing_care < hospitalized_on_icu - without_deaths_on_icu
        dgdt3d[self.index_hosp, index_C, needing_care_less_then_died_on_icu] = needing_care[needing_care_less_then_died_on_icu]
        dgdt3d[self.index_death_icu, index_C, needing_care_less_then_died_on_icu] = -deaths_on_icu[needing_care_less_then_died_on_icu]
        dgdt3d[self.index_hosp, index_U, needing_care_less_then_died_on_icu] = needing_care[needing_care_less_then_died_on_icu]
        dgdt3d[self.index_death_icu, index_U, needing_care_less_then_died_on_icu] = -deaths_on_icu[needing_care_less_then_died_on_icu]

        # Uncared - no ICU
        deaths_without_icu = self.death_rate_no_icu_sigma * y2d[index_U, :] # died without ICU treatment (all cases that don't get treatment die)
        dgdt3d[self.index_death_no_icu, index_U, :] = - deaths_without_icu  # without ICU treatment

        # R
        # proportion of removed people who recovered once returned
        dgdt3d[self.index_removal, index_R, :] = (1 - self.hospital_prob) * I_removed + A_removed
        dgdt3d[self.index_hosp, index_R, :] = self.hosp_rate * (1 - self.critical_prob) * H_vec


        # D
        dgdt3d[self.index_death_no_icu, index_D, :] = deaths_without_icu
        dgdt3d[self.index_death_icu, index_D, :] = params.death_prob_with_ICU * deaths_on_icu # died despite attempted ICU treatment

        # O

        # Q

        dgdt2d = np.zeros((y.shape[0], self.stoc_vars_num))
        for i in range(self.stoc_vars_num):
            dgdt2d[:, i] = dgdt3d[i].T.reshape(y.shape)
        return dgdt2d

    def run_model(self, tspan, random_seed=None, driftOnly=False):
        self.driftOnly = driftOnly

        if random_seed:
            print("Set random seed: " + str(random_seed))
            np.random.seed(random_seed)
        warnings.simplefilter("ignore")
        result = sdeint.itoint(self.sde_drift, self.sde_diffusion, self.y0, tspan)

        y_plot = np.zeros((len(tspan), len(self.params.categories.keys()) ))
        for name in self.params.calculated_categories:
            y_plot[:,self.params.categories[name]['index']] = result[:,self.params.categories[name]['index']]
            for i in range(1, self.age_categories): # age_categories
                y_plot[:, self.params.categories[name]['index']] += result[:, self.params.categories[name]['index'] + i*self.params.number_compartments]

        for name in self.params.change_in_categories: # daily change in
            name_changed_var = name[-1] # name of the variable we want daily change of
            y_plot[:, self.params.categories[name]['index']] = np.concatenate([[0], np.diff(y_plot[:,self.params.categories[name_changed_var]['index']])])

        # finally, 
        E = y_plot[:, self.params.categories['CE']['index']]
        I = y_plot[:, self.params.categories['CI']['index']]
        A = y_plot[:, self.params.categories['CA']['index']]

        y_plot[:, self.params.categories['Ninf']['index']] = [E[i] + I[i] + A[i] for i in range(len(E))] # change in total number of people with active infection

        return {'y': result.T,'t': tspan, 'y_plot': y_plot.T}

#--------------------------------------------------------------------

    def simulate_over_parameter_range_parallel(self, numberOfIterations, t_stop, n_processes, random_seed=None):
        logging.info(f"Running parallel simulation with {n_processes} processes")
        lazy_sols = []
        sols_raw = {}
        tspan = np.linspace(0,t_stop, t_stop+1) # 1 time value per day
        for ii in range(numberOfIterations):
            lazy_result = dask.delayed(self.run_model)(tspan, random_seed)
            lazy_sols.append(lazy_result)

        #with dask.config.set(scheduler='processes', num_workers=n_processes): --Does not work with Dask Distributed
        with dask.config.set(scheduler='single-threaded', num_workers=1):
            with ProgressBar():
                sols = dask.compute(*lazy_sols)

        for ii in range(numberOfIterations):
            sols_raw[ii] = sols[ii]

        [y_U95, y_UQ, y_LQ, y_L95, y_median] = self.generate_percentiles(sols)
        # standard run
        StandardSol = [self.run_model(tspan, random_seed, True)]

        return sols_raw, StandardSol, [y_U95, y_UQ, y_LQ, y_L95, y_median], None

    def generate_percentiles(self, sols):
        n_time_points = len(sols[0]['t'])

        y_plot = np.zeros((len(self.params.categories.keys()), len(sols) , n_time_points ))

        for k, sol in enumerate(sols):
            sol['y'] = np.asarray(sol['y'])
            for name in self.params.categories.keys():
                y_plot[self.params.categories[name]['index'],k,:] = sol['y_plot'][self.params.categories[name]['index']]

        y_L95, y_U95, y_LQ, y_UQ, y_median = [np.zeros((len(self.params.categories.keys()),n_time_points)) for i in range(5)]

        for name in self.params.categories.keys():
            y_L95[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],2.5) for i in range(n_time_points) ])
            y_LQ[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],25) for i in range(n_time_points) ])
            y_UQ[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],75) for i in range(n_time_points) ])
            y_U95[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],97.5) for i in range(n_time_points) ])

            y_median[self.params.categories[name]['index'],:] = np.asarray([statistics.median(y_plot[self.params.categories[name]['index'],:,i]) for i in range(n_time_points) ])
        return [y_U95, y_UQ, y_LQ, y_L95, y_median]
