import unittest
import pandas as pd
import pandas.testing as pdt
from ai4good.models.cm.cm_model import CompartmentalModel
from ai4good.models.abm.abm_model import ABM
import ai4good.runner.console_runner as cr
import ai4good.utils.path_utils as pu


class TestEndToEndRuns(unittest.TestCase):

    def test_custom_run(self):
        mr = cr.run_model(
            _model=CompartmentalModel.ID,
            _profile='custom',
            camp='Moria',
            load_from_cache=False,
            save_to_cache=False,
            is_save_plots=False,
            is_show_plots=False,
            is_save_report=False,
            overrides='{"numberOfIterations": 4, "nProcesses": 1}'
        )
        actual_report_df = mr.get('report')
        expected_df = pd.read_csv(pu._path('../runner/tests', 'expected_report.csv'))
        pdt.assert_frame_equal(expected_df, actual_report_df, check_exact=False, check_less_precise=1)

        # check(mr.get('prevalence_age'), 'expected_page.csv')
        # check(mr.get('prevalence_all'), 'expected_pall.csv')
        # check(mr.get('cumulative_all'), 'expected_call.csv')
        # check(mr.get('cumulative_age'), 'expected_cage.csv')

    # def test_custom_run_ABM(self):
    #     mr = cr.run_model(
    #         _model=ABM.ID,
    #         _profile='small',
    #         camp='Moria',
    #         load_from_cache=False,
    #         save_to_cache=False,
    #         is_save_plots=False,
    #         is_show_plots=False,
    #         is_save_report=False,
    #         overrides='{"numberOfIterations": 1, "nProcesses": 1}'
    #     )
    #     actual_report_df = mr.get('report')
    #     expected_df = pd.read_csv(pu._path('../runner/tests', 'expected_report.csv'))
    #     pdt.assert_frame_equal(expected_df, actual_report_df, check_exact=False, check_less_precise=1)

        # check(mr.get('prevalence_age'), 'expected_page.csv')
        # check(mr.get('prevalence_all'), 'expected_pall.csv')
        # check(mr.get('cumulative_all'), 'expected_call.csv')
        # check(mr.get('cumulative_age'), 'expected_cage.csv')

def check(actual, expected):
    actual = actual.reset_index()
    # actual.to_csv(expected, index=False)
    expected_df = pd.read_csv(pu._path('../runner/tests', expected))
    pdt.assert_frame_equal(expected_df, actual)



