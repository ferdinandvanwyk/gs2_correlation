# Standard
import os
import pytest

# Third Party

# Local
from gs2_correlation.simulation import Simulation
from gs2_correlation.configuration import Configuration

class TestClass(object):
    
    def setup_class(self):
        os.system('tar -zxf test/test_run.tar.gz -C test/.')

    def teardown_class(self):
        os.system('rm -rf test/test_run')

    @pytest.fixture(scope='function')
    def conf(self):
        config = Configuration('test/test_run/config.ini')
        config.read_config()
        return config

    @pytest.fixture(scope='function')
    def run(self, conf):
        sim = Simulation()
        sim.read_netcdf(conf)
        return sim

    def test_read_netcdf(self, run):
        assert ('field' in vars(run)) == True
        assert ('kx' in vars(run)) == True
        assert ('ky' in vars(run)) == True
        assert ('t' in vars(run)) == True
         

















