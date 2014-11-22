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
        field_shape = run.field.shape
        arr_shapes = (len(run.t), len(run.kx), len(run.ky), 2)
        assert field_shape == arr_shapes
    
    def test_interpolate(self, run, conf):
        # Need to think of a good test here
        run.interpolate(conf)
        field_shape = run.field.shape
        arr_shapes = (len(run.t), len(run.kx), len(run.ky), 2)
        assert field_shape == arr_shapes
        
    def test_zero_bes_scales(self, run, conf):
        conf.zero_bes_scales = True
        run.zero_bes_scales(conf)
        assert (run.field[:, 1, 1, :] == 0).all()


    def test_zf_bes(self, run, conf):
        run.zero_zf_scales(conf)
        assert (run.field[:, :, 0, :] == 0).all()

         

















