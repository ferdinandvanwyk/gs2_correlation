# Standard
import os
import pytest

# Third Party
import numpy as np

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
        sim = Simulation(conf)
        return sim

    def test_read_netcdf(self, run):
        field_shape = run.field.shape
        arr_shapes = (len(run.t), len(run.kx), len(run.ky))
        assert field_shape == arr_shapes
    
    def test_interpolate(self, run, conf):
        field_shape = run.field.shape
        arr_shapes = (len(run.t), len(run.kx), len(run.ky))
        assert field_shape == arr_shapes
        
    def test_zero_bes_scales(self, run, conf):
        assert (run.field[:, 1, 1] == 0).all()


    def test_zf_bes(self, run, conf):
        assert (run.field[:, :, 0] == 0).all()

    def test_to_complex(self, run):
        assert np.iscomplexobj(run.field) == True 

     

















