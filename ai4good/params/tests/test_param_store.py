import unittest
from ai4good.params.param_store import SimpleParamStore


class TestParamStore(unittest.TestCase):

    def test_simple_store_profiles(self):
        ps = SimpleParamStore()
        self.assertTrue(len(ps.get_models()) > 0)
        for m in ps.get_models():
            profiles = ps.get_profiles(m)
            self.assertTrue(len(profiles) > 0)
            for p in profiles:
                self.assertTrue(len(ps.get_params(m, p)) > 0)

    def test_simple_params(self):
        ps = SimpleParamStore()
        self.assertTrue(len(ps.get_camps()) > 0)

