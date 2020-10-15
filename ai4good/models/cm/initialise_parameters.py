"""
This file sets up the parameters for SEIR models used in the cov_functions_AI.py
"""

import numpy as np
import pandas as pd
import json
import hashlib
from ai4good.params.param_store import ParamStore
from ai4good.utils import path_utils as pu

class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}):
        self.ps = ps
        self.camp = camp
        disease_params = ps.get_disease_params()
        camp_params = ps.get_camp_params(camp)
        # ------------------------------------------------------------
        # disease params
        parameter_csv = disease_params
        model_params = parameter_csv[parameter_csv['Type'] == 'Model Parameter']
        model_params = model_params.loc[:, ['Name', 'Value']]
        control_data = parameter_csv[parameter_csv['Type'] == 'Control']
        self.model_params = model_params

        # print()

        R_0_list = np.asarray(model_params[model_params['Name'] == 'R0'].Value)

        latent_rate = 1 / (np.float(model_params[model_params['Name'] == 'latent period'].Value))
        removal_rate = 1 / (np.float(model_params[model_params['Name'] == 'infectious period'].Value))
        hosp_rate = 1 / (np.float(model_params[model_params['Name'] == 'hosp period'].Value))
        death_rate = 1 / (np.float(model_params[model_params['Name'] == 'death period'].Value))
        death_rate_with_ICU = 1 / (np.float(model_params[model_params['Name'] == 'death period with ICU'].Value))

        quarant_rate = 1 / (np.float(model_params[model_params['Name'] == 'quarantine period'].Value))

        death_prob_with_ICU = np.float(model_params[model_params['Name'] == 'death prob with ICU'].Value)

        number_compartments = int(model_params[model_params['Name'] == 'number_compartments'].Value)

        beta_list = [R_0 * removal_rate for R_0 in R_0_list]  # R_0 mu/N, N=1

        shield_decrease = np.float(control_data[control_data['Name'] == 'Reduction in contact between groups'].Value)
        shield_increase = np.float(control_data[control_data['Name'] == 'Increase in contact within group'].Value)

        better_hygiene = np.float(control_data.Value[control_data.Name == 'Better hygiene'])

        AsymptInfectiousFactor = np.float(model_params[model_params['Name'] == 'infectiousness of asymptomatic'].Value)

        
        self.R_0_list = R_0_list
        self.beta_list = beta_list
        
        self.shield_increase = shield_increase
        self.shield_decrease = shield_decrease
        self.better_hygiene  = better_hygiene


        self.number_compartments = number_compartments

        self.AsymptInfectiousFactor         = AsymptInfectiousFactor

        self.latent_rate     = latent_rate
        self.removal_rate    = removal_rate
        self.hosp_rate       = hosp_rate        
        self.quarant_rate    = quarant_rate

        self.death_rate          = death_rate
        self.death_rate_with_ICU = death_rate_with_ICU
        
        self.death_prob_with_ICU = death_prob_with_ICU

        self.S_ind = 0
        self.E_ind = 1
        self.I_ind = 2
        self.A_ind = 3
        self.R_ind = 4
        self.H_ind = 5
        self.C_ind = 6
        self.D_ind = 7
        self.O_ind = 8
        self.Q_ind = 9
        self.U_ind = 10

        self.index = {
            'S': self.S_ind,
            'E': self.E_ind,
            'I': self.I_ind,
            'A': self.A_ind,
            'R': self.R_ind,
            'H': self.H_ind,
            'C': self.C_ind,
            'D': self.D_ind,
            'O': self.O_ind,
            'Q': self.Q_ind,
            'U': self.U_ind,
            'CS': self.number_compartments + self.S_ind,
            'CE': self.number_compartments + self.E_ind,
            'CI': self.number_compartments + self.I_ind,
            'CA': self.number_compartments + self.A_ind,
            'CR': self.number_compartments + self.R_ind,
            'CH': self.number_compartments + self.H_ind,
            'CC': self.number_compartments + self.C_ind,
            'CD': self.number_compartments + self.D_ind,
            'CO': self.number_compartments + self.O_ind,
            'CQ': self.number_compartments + self.Q_ind,
            'CU': self.number_compartments + self.U_ind,
            'Ninf': 2 * self.number_compartments
        }

        self.calculated_categories = ['S',
                'E',
                'I',
                'A',
                'R',
                'H',
                'C',
                'D',
                'O',
                'Q',
                'U'
                ]

        self.change_in_categories = ['C'+ii for ii in self.calculated_categories] # gives daily change for each category

        self.longname = {'S':  'Susceptible',
                    'E':  'Exposed',
                    'I':  'Infected (symptomatic)',
                    'A':  'Asymptomatically Infected',
                    'R':  'Recovered',
                    'H':  'Hospitalised',
                    'C':  'Critical',
                    'D':  'Deaths',
                    'O':  'Offsite',
                    'Q':  'Quarantined',
                    'U':  'No ICU Care',
                    'CS': 'Change in Susceptible',
                    'CE': 'Change in Exposed',
                    'CI': 'Change in Infected (symptomatic)',
                    'CA': 'Change in Asymptomatically Infected',
                    'CR': 'Change in Recovered',
                    'CH': 'Change in Hospitalised',
                    'CC': 'Change in Critical',
                    'CD': 'Change in Deaths',
                    'CO': 'Change in Offsite',
                    'CQ': 'Change in Quarantined',
                    'CU': 'Change in No ICU Care',
                    'Ninf': 'Change in total active infections', # sum of E, I, A
        }

        self.shortname = {'S':  'Sus.',
                     'E':  'Exp.',
                     'I':  'Inf. (symp.)',
                     'A':  'Asym.',
                     'R':  'Rec.',
                     'H':  'Hosp.',
                     'C':  'Crit.',
                     'D':  'Deaths',
                     'O':  'Offsite',
                     'Q':  'Quar.',
                     'U':  'No ICU',
                     'CS': 'Change in Sus.',
                     'CE': 'Change in Exp.',
                     'CI': 'Change in Inf. (symp.)',
                     'CA': 'Change in Asym.',
                     'CR': 'Change in Rec.',
                     'CH': 'Change in Hosp.',
                     'CC': 'Change in Crit.',
                     'CD': 'Change in Deaths',
                     'CO': 'Change in Offsite',
                     'CQ': 'Change in Quar.',
                     'CU':  'Change in No ICU',
                     'Ninf': 'New Infected', # newly exposed to the disease = - change in susceptibles
        }

        self.colour = {'S':  'rgb(0,0,255)', #'blue',
                  'E':  'rgb(255,150,255)', #'pink',
                  'I':  'rgb(255,150,50)', #'orange',
                  'A':  'rgb(255,50,50)', #'dunno',
                  'R':  'rgb(0,255,0)', #'green',
                  'H':  'rgb(255,0,0)', #'red',
                  'C':  'rgb(50,50,50)', #'black',
                  'D':  'rgb(130,0,255)', #'purple',
                  'O':  'rgb(130,100,150)', #'dunno',
                  'Q':  'rgb(150,130,100)', #'dunno',
                  'U':  'rgb(150,100,150)', #'dunno',
                  'CS': 'rgb(0,0,255)', #'blue',
                  'CE': 'rgb(255,150,255)', #'pink',
                  'CI': 'rgb(255,150,50)', #'orange',
                  'CA': 'rgb(255,50,50)', #'dunno',
                  'CR': 'rgb(0,255,0)', #'green',
                  'CH': 'rgb(255,0,0)', #'red',
                  'CC': 'rgb(50,50,50)', #'black',
                  'CD': 'rgb(130,0,255)', #'purple',
                  'CO': 'rgb(130,100,150)', #'dunno',
                  'CQ': 'rgb(150,130,100)', #'dunno',
                  'CU':  'rgb(150,100,150)', #'dunno',
                  'Ninf': 'rgb(255,125,100)', #
                }

        self.categories = {}
        for key in self.longname.keys():
            self.categories[key] = dict(
                longname = self.longname[key],
                shortname = self.shortname[key],
                colour = self.colour[key],
                fill_colour = 'rgba' + self.colour[key][3:-1] + ',0.1)' ,
                index = self.index[key]
            )

        self.population_frame, self.population = self.prepare_population_frame(camp_params)

        self.control_dict, self.icu_count = self.load_control_dict(profile, profile_override_dict)

        self.infection_matrix, self.im_beta_list, self.largest_eigenvalue = self.generate_infection_matrix()
        self.generated_disease_vectors = self.ps.get_generated_disease_param_vectors()

    def sha1_hash(self) -> str:
        hash_params = [
            {i: self.control_dict[i] for i in self.control_dict if i != 'nProcesses'},
            self.infection_matrix.tolist(),
            self.im_beta_list.tolist(),
            self.largest_eigenvalue,
            self.generated_disease_vectors.to_dict('records'),
            self.population_frame.to_dict('records'),
            self.population,
            self.camp,
            self.calculated_categories,
            self.model_params.to_dict('records')
        ]
        serialized_params = json.dumps(hash_params, sort_keys=True)
        hash_object = hashlib.sha1(serialized_params.encode('UTF-8'))
        _hash = hash_object.hexdigest()
        return _hash

    def load_control_dict(self, profile, profile_override_dict):

        def get_values(df, p_name):
            val_rows = df[df['Parameter'] == p_name]
            assert len(val_rows) == 1
            val_row = val_rows.iloc[0]
            st = val_row['Start Time']
            et = val_row['End Time']
            _v = val_row['Value']
            if (st is None) or (et is None) or (st == '<no_edit>') or (et == '<no_edit>'):
                _t = None
            else:
                _t = [int(st), int(et)]
            return p_name, _v, _t

        def add_int_scalar(df, p_name, dct):
            p, _v, _ = get_values(df, p_name)
            dct[p] = int(_v)

        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")

        cd = profile.copy()
        dct = {}
        p, v, t = get_values(cd, 'better_hygiene')
        dct[p] = {
            'timing': t,
            'value': self.better_hygiene if v == '<default>' else float(v)
        }

        p, v, _ = get_values(cd, 'ICU_capacity')
        icu_capacity = int(v)
        dct[p] = {'value': icu_capacity / self.population}

        p, v, t = get_values(cd, 'remove_symptomatic')
        dct[p] = {
            'timing': t,
            'rate': float(v) / self.population
        }

        p, v, t = get_values(cd, 'shielding')
        dct[p] = {'used': str2bool(v)}

        p, v, t = get_values(cd, 'remove_high_risk')
        _, v2, _ = get_values(cd, 'remove_high_risk_categories')
        dct[p] = {
            'timing': t,
            'rate': float(v) / self.population,
            'n_categories_removed': int(v2)
        }

        add_int_scalar(cd, 't_sim', dct)
        add_int_scalar(cd, 'numberOfIterations', dct)
        add_int_scalar(cd, 'nProcesses', dct)

        for k, d in dct.items():
            if k in profile_override_dict.keys():
                dct[k] = profile_override_dict[k]

        return dct, icu_capacity

    @staticmethod
    def prepare_population_frame(population_frame):
        population_frame = population_frame.loc[:, 'Age':'Total_population']

        population_frame = population_frame.assign(p_hospitalised=lambda x: (x.Hosp_given_symptomatic / 100),
                                                   # *frac_symptomatic,
                                                   p_critical=lambda x: (x.Critical_given_hospitalised / 100))

        # make sure population frame.value sum to 100
        # population_frame.loc[:,'Population'] = population_frame.Population/sum(population_frame.Population)

        population_size = np.float(population_frame.reset_index()['Total_population'][0])

        return population_frame, population_size

    def generate_infection_matrix(self):
        infection_matrix = np.asarray(self.ps.get_contact_matrix_params(self.camp))
        infection_matrix = infection_matrix[:, 2:].astype(np.double)
        assert infection_matrix.shape[0] == infection_matrix.shape[1]
        next_generation_matrix = np.matmul(0.01 * np.diag(self.population_frame.Population_structure), infection_matrix)
        largest_eigenvalue = max(np.linalg.eig(next_generation_matrix)[0])  # max eigenvalue

        beta_list = np.linspace(self.beta_list[0], self.beta_list[2], 20)
        beta_list = np.real((1 / largest_eigenvalue) * beta_list)  # in case eigenvalue imaginary

        if self.control_dict['shielding']['used']:  # increase contact within group and decrease between groups
            divider = -1  # determines which groups separated. -1 means only oldest group separated from the rest

            infection_matrix[:divider, :divider] = self.shield_increase * infection_matrix[:divider, :divider]
            infection_matrix[:divider, divider:] = self.shield_decrease * infection_matrix[:divider, divider:]
            infection_matrix[divider:, :divider] = self.shield_decrease * infection_matrix[divider:, :divider]
            infection_matrix[divider:, divider] = self.shield_increase * infection_matrix[divider:, divider:]

        return infection_matrix, beta_list, largest_eigenvalue


def generate_contact_matrix(camp_name: str, age_limits: np.array, contact_matrix: np.ndarray):
    camp_params_df = pd.read_csv(pu.params_path('camp_params.csv'))
    population_array = camp_params_df[camp_params_df['Camp'] == camp_name]['Population_structure'].to_numpy()

    n_categories = len(age_limits) - 1
    ind_limits = np.array(age_limits / 5, dtype=int)

    p = np.zeros(16)
    for i in range(n_categories):
        p[ind_limits[i]: ind_limits[i + 1]] = population_array[i] / (ind_limits[i + 1] - ind_limits[i])

    transformed_matrix = np.zeros((n_categories, n_categories))

    for i in range(n_categories):
        for j in range(n_categories):
            sump = sum(p[ind_limits[i]: ind_limits[i + 1]])
            b = contact_matrix[ind_limits[i]: ind_limits[i + 1], ind_limits[j]: ind_limits[j + 1]] * np.array(
                p[ind_limits[i]: ind_limits[i + 1]]).transpose()
            v1 = b.sum() / sump
            transformed_matrix[i, j] = v1

    return transformed_matrix
