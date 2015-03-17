# Standard
import os
import pytest

# Third Party
import numpy as np
import matplotlib
matplotlib.use('Agg') # specifically for Travis CI to avoid backend errors
from PIL import Image

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
        assert type(run.domain) == str
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
        assert type(run.npeaks_fit) == int
        assert type(run.time_guess) == int
        assert type(run.box_size) == list
        assert type(run.time_range) == list

        assert type(run.amin) == float
        assert type(run.vth) == float
        assert type(run.rhoref) == float
        assert type(run.pitch_angle) == float
        assert type(run.rmaj) == float
        assert type(run.nref) == float
        assert type(run.tref) == float

        assert type(run.seaborn_context) == str
        assert type(run.film_fps) == int
        assert type(run.film_contours) == int
        assert (type(run.film_lim) == list or type(run.film_lim) == 
                                               type(None))
        assert type(run.write_field_interp_x) == bool

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

    def test_field_to_complex(self, run):
        assert np.iscomplexobj(run.field) == True 

    def test_field_to_real_space(self, run):
        run.field_to_real_space()
        assert run.field_real_space.shape == (run.nt, run.nx, run.ny)

    def test_domain_reduce(self, run):
        run.box_size = [0.1, 0.1]
        original_max_x = run.x[-1]
        original_max_y = run.y[-1]
        run.domain_reduce()
        assert run.x[-1] <= original_max_x
        assert run.y[-1] <= original_max_y

    def test_perp_analysis(self, run):
        run.perp_analysis()
        assert run.perp_fit_params.shape == (5,4)
        assert ('perp_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('time_avg_correlation.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_corr_fit.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_comparison.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_params_vs_time_slice.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_summary.txt' in os.listdir('test/test_run/v/id_1/analysis/perp'))

    def test_perp_analysis_3(self, run):
        run.perp_guess = [1,1,1]
        run.perp_analysis()
        assert run.perp_fit_params.shape == (5,4)
        assert ('perp_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('time_avg_correlation.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_corr_fit.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_comparison.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_params_vs_time_slice.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_summary.txt' in os.listdir('test/test_run/v/id_1/analysis/perp'))

    def test_calculate_perp_corr(self, run):
        run.calculate_perp_corr()
        assert run.perp_corr.shape == (run.nt, 2*run.nx-1, 2*run.ny-1)
    
    def test_time_analysis(self, run):
        run.time_analysis()
        assert ('corr_time.csv' in os.listdir('test/test_run/v/id_1/analysis/time'))
        assert ('corr_time.pdf' in os.listdir('test/test_run/v/id_1/analysis/time'))
        assert run.field_real_space.shape == (run.nt, run.nx, run.ny)
        assert run.time_corr.shape == (run.nt_slices, 2*run.time_slice-1,
                                       run.nx, 2*run.ny-1)
        assert ('corr_fns' in os.listdir('test/test_run/v/id_1/analysis/time'))
        assert ('time_fit_it_0_ix_0.pdf' in 
                os.listdir('test/test_run/v/id_1/analysis/time/corr_fns'))

    def test_write_field(self, run):
        run.write_field()
        assert ('ntot_igomega_by_mode.cdf' in os.listdir('test/test_run/v/id_1/analysis/write_field'))

    def test_make_film(self, run):
        run.make_film()
        assert ('ntot_igomega_by_mode_spec_0_0000.png' in os.listdir('test/test_run/v/id_1/analysis/film/film_frames'))
        assert ('ntot_igomega_by_mode_spec_0.mp4' in os.listdir('test/test_run/v/id_1/analysis/film/'))
        
        im = Image.open('test/test_run/v/id_1/analysis/film/film_frames/ntot_igomega_by_mode_spec_0_0000.png')
        size = im.size
        assert size[0] % 2 == 0
        assert size[1] % 2 == 0


















