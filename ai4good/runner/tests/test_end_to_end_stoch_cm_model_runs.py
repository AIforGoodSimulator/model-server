# import unittest
# import numpy as np
# import pandas as pd
# import pandas.testing as pdt
# from ai4good.models.cm.cm_model import CompartmentalModelStochastic
# import ai4good.runner.console_runner as cr
# import ai4good.utils.path_utils as pu
#
#
# class TestEndToEndRuns(unittest.TestCase):
#
#     def test_custom_run(self):
#         mr = cr.run_model(
#             _model=CompartmentalModelStochastic.ID,
#             _profile='custom',
#             camp='Moria',
#             load_from_cache=False,
#             save_to_cache=False,
#             is_save_plots=False,
#             is_show_plots=False,
#             is_save_report=False,
#             overrides='{"numberOfIterations": 4, "nProcesses": 1, "random_seed": 7}'
#         )
#         actual_report_df = mr.get('report')
#         # actual_report_df.to_csv(pu._path('../runner/tests', 'stoch_cm_expected_report.csv'), index=False) # for making new snapshot
#         expected_df = pd.read_csv(pu._path('../runner/tests', 'stoch_cm_expected_report.csv'))
#         pdt.assert_frame_equal(expected_df, actual_report_df, check_exact=False, check_less_precise=0)
#
#
#
#
