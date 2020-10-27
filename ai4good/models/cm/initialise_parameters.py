import numpy as np
import pandas as pd
import json
import hashlib
from ai4good.params.param_store import ParamStore
from ai4good.utils import path_utils as pu
from ai4good.models.cm import longname, shortname, colour, index, fill_colour
from ai4good.params.disease_params import covid_specific_parameters
from ai4good.params.model_control_params import model_config_cm


class Parameters:
    def __init__(self, ps: ParamStore, camp: str, profile: pd.DataFrame, profile_override_dict={}):
        self.ps = ps
        self.camp = camp
        self.age_limits = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=int)
        # disease_params = ps.get_disease_params()
        disease_params = covid_specific_parameters
        self.camp_params = ps.get_camp_params(camp)
        # ------------------------------------------------------------
        # disease params
        # parameter_csv = disease_params
        # model_params = parameter_csv[parameter_csv['Type'] == 'Model Parameter']
        # model_params = model_params.loc[:, ['Name', 'Value']]
        # control_data = parameter_csv[parameter_csv['Type'] == 'Control']
        # self.model_params = model_params

        # TODO: get all the disease parameters injected to the right place
        # TODO: seperate the profile default parameters into a model config or something

        self.R_0_list = np.asarray([covid_specific_parameters["R0_low"],covid_specific_parameters["R0_medium"],covid_specific_parameters["R0_high"]])
        self.latent_rate = 1 / (np.float(covid_specific_parameters["Latent_period"]))
        self.removal_rate = 1 / (np.float(covid_specific_parameters["Infectious_period"]))
        self.hosp_rate = 1 / (np.float(covid_specific_parameters["Hosp_period"]))
        self.death_rate = 1 / (np.float(covid_specific_parameters["Death_period"]))
        self.death_rate_with_ICU = 1 / (np.float(covid_specific_parameters["Death_period_withICU"]))
        self.death_prob_with_ICU= np.float(covid_specific_parameters["Death_prob_withICU"])
        self.number_compartments = 11 # S,E,I,A,R,H,C,D,O,Q,U refer to model write up for more details
        self.beta_list = [R_0 * self.removal_rate for R_0 in self.R_0_list]  # R_0 mu/N, N=1
        self.AsymptInfectiousFactor = np.float(covid_specific_parameters["Infectiousness_asymptomatic"])

        # These are unique model control params
        self.shield_decrease = np.float(model_config_cm["shiedling_reduction_between_groups"])
        self.shield_increase = np.float(model_config_cm["shielding_increase_within_group"])
        self.better_hygiene = np.float(model_config_cm["better_hygiene_infection_scale"])

        #we will get this from the UI default to 14 for now
        self.quarant_rate = 1 / (np.float(model_config_cm["default_quarantine_period"]))

        self.calculated_categories = ['S','E','I','A','R','H','C','D','O','Q','U']
        self.change_in_categories = ['C'+category for category in self.calculated_categories]
        categories_dicts = [longname,shortname,colour,index,fill_colour]
        categories_df = pd.DataFrame(categories_dicts, index =['longname', 'shortname','colour','index','fill_colour'])
        self.categories: dict = categories_df.to_dict()

        self.population_frame, self.population = self.prepare_population_frame(self.camp_params)

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
            self.country,
            self.calculated_categories,
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

        profile_copy = profile.copy()
        dct = {}
        p, v, t = get_values(profile_copy, 'better_hygiene')
        dct[p] = {
            'timing': t,
            'value': self.better_hygiene if v == '<default>' else float(v)
        }

        p, v, _ = get_values(profile_copy, 'ICU_capacity')
        icu_capacity = int(v)
        dct[p] = {'value': icu_capacity / self.population}

        p, v, t = get_values(profile_copy, 'remove_symptomatic')
        dct[p] = {
            'timing': t,
            'rate': float(v) / self.population
        }

        p, v, t = get_values(profile_copy, 'shielding')
        dct[p] = {'used': str2bool(v)}

        p, v, t = get_values(profile_copy, 'remove_high_risk')
        _, v2, _ = get_values(profile_copy, 'remove_high_risk_categories')
        dct[p] = {
            'timing': t,
            'rate': float(v) / self.population,
            'n_categories_removed': int(v2)
        }

        add_int_scalar(profile_copy, 't_sim', dct)
        add_int_scalar(profile_copy, 'numberOfIterations', dct)
        add_int_scalar(profile_copy, 'nProcesses', dct)

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
        infection_matrix = self.generate_contact_matrix(self.age_limits)
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

    def generate_contact_matrix(self, age_limits: np.array):
        supported_countries = self.ps.get_supported_countries()
        self.country = self.camp_params['Country'].tolist()[0]
        contact_matrix_path = pu.params_path(f'contact_matrices/{self.country}.csv')
        contact_matrix = pd.read_csv(contact_matrix_path).to_numpy()
        population_array = self.camp_params['Population_structure'].to_numpy()
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
