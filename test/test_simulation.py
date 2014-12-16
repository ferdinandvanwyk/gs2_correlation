# Standard
import os
import pytest

# Third Party
import numpy as np
import matplotlib
matplotlib.use('Agg') # specifically for Travis CI to avoid backend errors

# Local
from gs2_correlation.simulation import Simulation

class TestClass(object):
    
    def setup_class(self):
        os.system('tar -zxf test/test_run.tar.gz -C test/.')

    def teardown_class(self):
        os.system('rm -rf test/test_run')

    @pytest.fixture(scope='function')
    def run(self):
        sim = Simulation('test/test_run/config.ini')
        return sim

    def test_init(self, run):
        assert type(run.config_file) == str

    def test_read(self, run):
        run.read_config()
        assert type(run.in_file) == str
        assert type(run.file_ext) == str
        assert type(run.in_field) == str
        assert type(run.analysis) == str
        assert type(run.out_dir) == str
        assert type(run.time_slice) == int
        assert type(run.perp_fit_length) == int
        assert type(run.perp_guess) == list
        assert type(run.interpolate_bool) == bool
        assert type(run.zero_bes_scales_bool) == bool
        assert type(run.zero_zf_scales_bool) == bool
        assert (type(run.spec_idx) == int or type(run.spec_idx) == type(None))
        assert (type(run.theta_idx) == int or type(run.theta_idx) == 
                                               type(None))
        assert type(run.amin) == float
        assert type(run.vth) == float
        assert type(run.rhoref) == float
        assert type(run.pitch_angle) == float

    def test_read_netcdf(self, run):
        field_shape = run.field.shape
        arr_shapes = (run.nt, run.nkx, run.nky)
        assert field_shape == arr_shapes
    
    def test_interpolate(self, run):
        field_shape = run.field.shape
        arr_shapes = (run.nt, run.nkx, run.nky)
        assert field_shape == arr_shapes
        
    def test_zero_bes_scales(self, run):
        assert (run.field[:, 1, 1] == 0).all()


    def test_zf_bes(self, run):
        assert (run.field[:, :, 0] == 0).all()

    def test_to_complex(self, run):
        assert np.iscomplexobj(run.field) == True 

    def test_wk_2d(self, run):
        run.wk_2d()
        assert run.perp_corr.shape == (run.nt, run.nkx, run.ny-1)
    
    def test_perp_analysis(self, run):
        run.perp_analysis()
        assert run.perp_fit_params.shape == (5,4)

    def test_perp_plots(self, run):
        run.perp_analysis()
        assert ('perp_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis'))
        assert ('time_avg_correlation.pdf' in os.listdir('test/test_run/v/id_1/analysis'))
        assert ('perp_corr_fit.pdf' in os.listdir('test/test_run/v/id_1/analysis'))
        assert ('perp_fit_comparison.pdf' in os.listdir('test/test_run/v/id_1/analysis'))

    def test_perp_plots(self, run):
        run.perp_analysis()
        assert ('perp_fit_params_vs_time_slice.pdf' in os.listdir('test/test_run/v/id_1/analysis'))
        assert ('perp_fit_summary.txt' in os.listdir('test/test_run/v/id_1/analysis'))



     

















