# Standard
import os
import pytest

# Third Party
import numpy as np
import matplotlib
matplotlib.use('Agg') # specifically for Travis CI to avoid backend errors

# Local
from gs2_correlation.middle_box_analysis import MiddleBox

class TestClass(object):
    
    def setup_class(self):
        os.system('tar -zxf test/test_run.tar.gz -C test/.')

    def teardown_class(self):
        os.system('rm -rf test/test_run')

    @pytest.fixture(scope='function')
    def run(self):
        sim = MiddleBox('test/test_run/config.ini')
        return sim

    def test_perp_analysis(self, run):
        run.perp_analysis()
        assert run.perp_fit_params.shape == (5,4)
        assert ('perp_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('time_avg_correlation.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_corr_fit.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_comparison.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_params_vs_time_slice.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_summary.txt' in os.listdir('test/test_run/v/id_1/analysis/perp'))

