from typeguard import typechecked
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import pandas as pd
import copy
import ai4good.utils.path_utils as pu


@typechecked
class ParamStore(ABC):

    @abstractmethod
    def get_models(self) -> List[str]:
        """
        Returns list of available models
        """
        pass

    @abstractmethod
    def get_camps(self) -> List[str]:
        """
        Returns list of available camps
        """
        pass

    @abstractmethod
    def get_profiles(self, model: str) -> List[str]:
        """
        Returns list of available profiles for given model
        """
        pass

    @abstractmethod
    def get_params(self, model: str, profile: str) -> Dict[str, Any]:
        """
        Given model name and profile return model specific parameter dictionary
        """
        pass

    @abstractmethod
    def get_camp_params(self, camp: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_contact_matrix_params(self, camp: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_disease_params(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_generated_disease_param_vectors(self) -> pd.DataFrame:
        pass


@typechecked
class SimpleParamStore(ParamStore):

    profiles = {
        'compartmental-model': {  # Each of this needs to be separate document
            'baseline': {
                # reduces transmission rate by disease_params.better_hygiene or by explicy value
                'better_hygiene': {
                    'timing': [0, 0]
                },
                'ICU_capacity': {
                    'value': 6
                },
                # move symptomatic cases off site
                'remove_symptomatic': {
                    'rate': 10, # people per day
                    'timing': [0, 0]
                },
                # partially separate low and high risk
                # (for now) assumed that if do this, do for entire course of epidemic
                'shielding': {
                    'used': False
                },
                # move uninfected high risk people off site
                'remove_high_risk': {
                    'rate': 20,  # people per day
                    'n_categories_removed': 2,  # remove oldest n categories
                    'timing': [0, 0]
                },
                't_sim': 200,  # simulation runtime,
                'numberOfIterations': 1000,  # suggest 800-1000 for real thing
                'nProcesses': 4 # parallelization
            },
            'better_hygiene': {
                'better_hygiene': {
                    'timing': [0, 30]
                },
                'ICU_capacity': {
                    'value': 6
                },
                'remove_symptomatic': {
                    'rate': 10,
                    'timing': [0, 0]
                },
                'shielding': {
                    'used': False
                },
                'remove_high_risk': {
                    'rate': 20,
                    'n_categories_removed': 2,
                    'timing': [0, 0]
                },
                't_sim': 200,
                'numberOfIterations': 1000,
                'nProcesses': 4
            },
            'custom': {
                'better_hygiene': {
                    'value': 0.7,
                    'timing': [0, 200]
                },
                'ICU_capacity': {
                    'value': 6
                },
                'remove_symptomatic': {
                    'rate': 50,
                    'timing': [30, 90]
                },
                'shielding': {
                    'used': False
                },
                'remove_high_risk': {
                    'rate': 50,
                    'n_categories_removed': 2,
                    'timing': [0, 12]
                },
                't_sim': 200,
                'numberOfIterations': 1000,
                'nProcesses': 4
            },
            'remove_highrisk': {
                'better_hygiene': {
                    'timing': [0, 200]
                },
                'ICU_capacity': {
                    'value': 6
                },
                'remove_symptomatic': {
                    'rate': 10,
                    'timing': [0, 0]
                },
                'shielding': {
                    'used': False
                },
                'remove_high_risk': {
                    'rate': 100,
                    'n_categories_removed': 2,
                    'timing': [0, 6]
                },
                't_sim': 200,
                'numberOfIterations': 1000,
                'nProcesses': 4
            },
            'remove_symptomatic': {
                'better_hygiene': {
                    'timing': [0, 0]
                },
                'ICU_capacity': {
                    'value': 6
                },
                'remove_symptomatic': {
                    'rate': 20,
                    'timing': [0, 200]
                },
                'shielding': {
                    'used': False
                },
                'remove_high_risk': {
                    'rate': 20,
                    'n_categories_removed': 2,
                    'timing': [0, 0]
                },
                't_sim': 200,
                'numberOfIterations': 1000,
                'nProcesses': 4
            },
            'shielding': {
                'better_hygiene': {
                    'timing': [0, 0]
                },
                'ICU_capacity': {
                    'value': 6
                },
                'remove_symptomatic': {
                    'rate': 10,
                    'timing': [0, 0]
                },
                'shielding': {
                    'used': True
                },
                'remove_high_risk': {
                    'rate': 20,
                    'n_categories_removed': 2,
                    'timing': [0, 0]
                },
                't_sim': 200,
                'numberOfIterations': 1000,
                'nProcesses': 4
            }
        }
    }

    def __init__(self):
        assert len(self.get_models()) == len(self.profiles.keys())

    def get_models(self) -> List[str]:
        return ['compartmental-model']

    def get_profiles(self, model: str) -> List[str]:
        return list(self.profiles[model].keys())

    def get_params(self, model: str, profile: str) -> Dict[str, Any]:
        return copy.deepcopy(self.profiles[model][profile])

    def get_camps(self) -> List[str]:
        df = self._read_csv("camp_params.csv")
        return df.Camp.dropna().sort_values().unique().tolist()

    def get_camp_params(self, camp: str) -> pd.DataFrame:
        df = self._read_csv("camp_params.csv")
        return df[df.Camp == camp]

    def get_contact_matrix_params(self, camp: str) -> pd.DataFrame:
        df = self._read_csv("contact_matrix_params.csv")
        return df[df.Camp == camp].copy()

    def get_disease_params(self) -> pd.DataFrame:
        return self._read_csv("disease_params.csv")

    def get_generated_disease_param_vectors(self) -> pd.DataFrame:
        return self._read_csv("generated_params.csv")

    @staticmethod
    def _read_csv(name: str) -> pd.DataFrame:
        return pd.read_csv(pu.params_path(name))
