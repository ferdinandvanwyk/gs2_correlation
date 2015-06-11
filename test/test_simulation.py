# Standard
import os
import pytest

# Third Party
import numpy as np
import matplotlib
matplotlib.use('Agg') # specifically for Travis CI to avoid backend errors
from PIL import Image
import f90nml as nml

# Local
from gs2_correlation.simulation import Simulation

class TestClass(object):
    
    def setup_class(self):
        os.system('tar -zxf test/test_run.tar.gz -C test/.')

    def teardown_class(self):
        os.system('rm -rf test/test_run')

    @pytest.fixture(scope='function')
    def run(self):
        sim = Simulation('test/test_config.ini')
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
        assert type(run.time_interpolate_bool) == bool
        assert type(run.time_interp_fac) == int
        assert type(run.zero_bes_scales_bool) == bool
        assert type(run.zero_zf_scales_bool) == bool
        assert type(run.lab_frame) == bool
        assert (type(run.spec_idx) == int or type(run.spec_idx) == type(None))
        assert type(run.npeaks_fit) == int
        assert type(run.time_guess) == float
        assert type(run.box_size) == list
        assert type(run.time_range) == list
        assert type(run.par_guess) == list

        assert type(run.amin) == float
        assert type(run.vth) == float
        assert type(run.rho_ref) == float
        assert type(run.bref) == float
        assert type(run.nref) == float
        assert type(run.tref) == float
        assert type(run.omega) == float
        assert type(run.dpsi_da) == float

        assert type(run.seaborn_context) == str
        assert type(run.film_fps) == int
        assert type(run.film_contours) == int
        assert (type(run.film_lim) == list or type(run.film_lim) == 
                                               type(None))
        assert type(run.write_field_interp_x) == bool

    def test_read_netcdf(self, run):
        field_shape = run.field.shape
        arr_shapes = (run.nt, run.nkx, run.nky, run.ntheta)
        assert field_shape == arr_shapes

    def test_read_netcdf_theta(self, run):
        run.theta_idx = -1
        field_shape = run.field.shape
        arr_shapes = (run.nt, run.nkx, run.nky, run.ntheta)
        assert field_shape == arr_shapes

    def test_read_geometry_file(self, run):
        run.read_geometry_file()
        assert run.geometry.shape[1] > 6
    
    def test_read_input_file(self, run):
        run.read_input_file()
        assert type(run.input_file) == nml.namelist.NmlDict
    
    def test_time_interpolate(self, run):
        field_shape = run.field.shape
        arr_shapes = (run.nt, run.nkx, run.nky, run.ntheta)
        assert field_shape == arr_shapes
        
    def test_time_interpolate_4(self, run):
        run.time_interp_fac = 4
        arr_shapes = (run.time_interp_fac*run.nt, run.nkx, run.nky, run.ntheta)
        run.time_interpolate()
        field_shape = run.field.shape
        assert field_shape == arr_shapes

    def test_zero_bes_scales(self, run):
        assert (run.field[:, 1, 1, 0] == 0).all()

    def test_zero_zf_scales(self, run):
        assert (run.field[:, :, 0, :] == 0).all()

    def test_to_lab_frame(self, run):
        run.field = np.ones([51,5,6,9])
        run.to_lab_frame()
        assert np.abs(run.field[5,0,3,0] - np.real(np.exp(1j*3*10*run.omega*run.t[5]))) < 1e-5

    def test_field_to_complex(self, run):
        assert np.iscomplexobj(run.field) == True 

    def test_fourier_correction(self, run):
        run.field = np.ones([51, 5, 5, 9])
        run.fourier_correction()
        assert ((run.field[:,:,1:,:] - 0.5) < 1e-5).all()

    def test_field_to_real_space(self, run):
        assert run.field_real_space.shape == (run.nt, run.nx, run.ny)
        
    def test_domain_reduce(self, run):
        run.box_size = [0.1, 0.1]
        original_max_x = run.x[-1]
        original_max_y = run.y[-1]
        run.field_real_space = run.field_real_space[:,:,:,np.newaxis]
        run.domain_reduce()
        assert run.x[-1] <= original_max_x
        assert run.y[-1] <= original_max_y

    def test_field_odd_pts(self, run):
        run.field_real_space = np.ones([51,5,6,9])
        run.x = np.ones([5])
        run.nx = 5
        run.y = np.ones([6])
        run.ny = 6
        run.field_odd_pts()
        #print(run.field_real_space.shape)
        assert (np.array(run.field_real_space.shape)%2 == [1,1,1,1]).all()
        assert len(run.t)%2 == 1
        assert len(run.x)%2 == 1
        assert len(run.y)%2 == 1
        assert len(run.dx)%2 == 1
        assert len(run.dy)%2 == 1
        assert len(run.fit_dx)%2 == 1
        assert len(run.fit_dy)%2 == 1

    def test_perp_analysis_3(self, run):
        run.perp_guess = [5,1,0.1]
        run.perp_analysis()
        assert run.perp_fit_params.shape == (5,4)
        assert ('perp_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis/perp_ky_fixed'))
        assert ('time_avg_correlation.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp_ky_fixed'))
        assert ('perp_corr_fit.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp_ky_fixed'))
        assert ('perp_fit_comparison.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp_ky_fixed'))
        assert ('perp_fit_params_vs_time_slice.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp_ky_fixed'))
        assert ('perp_fit_summary.dat' in os.listdir('test/test_run/v/id_1/analysis/perp_ky_fixed'))

    def test_perp_analysis(self, run):
        run.perp_analysis()
        assert run.perp_fit_params.shape == (5,4)
        assert ('perp_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('time_avg_correlation.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_corr_fit.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_comparison.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_params_vs_time_slice.pdf' in os.listdir('test/test_run/v/id_1/analysis/perp'))
        assert ('perp_fit_summary.dat' in os.listdir('test/test_run/v/id_1/analysis/perp'))

    def test_field_normalize_perp(self, run):
        run.field_normalize_perp()
        assert run.field_real_space_norm.shape == (run.nt, run.nx, run.ny)

    def test_perp_norm_mask(self, run):
        run.perp_corr = np.ones([51,5,5])
        run.perp_norm_mask()
        assert np.abs(run.perp_corr[0,2,2] - 1./25.) < 1e-5

    def test_calculate_perp_corr(self, run):
        run.field_normalize_perp()
        run.calculate_perp_corr()
        assert run.perp_corr.shape == (run.nt, run.nx, run.ny)

    def test_fluctuation_level(self, run):
        run.field_real_space = np.random.randint(0,10,size=[5,5,5])
        run.perp_dir = 'perp'
        run.fluctuation_levels()
        assert np.abs(run.fluc_level - np.mean(np.sqrt(np.mean(run.field_real_space**2, axis=0)))) < 1e-5
        assert np.abs(run.fluc_level_std - np.std(np.sqrt(np.mean(run.field_real_space**2, axis=0)))) < 1e-5
        assert ('fluctuation_summary.dat' in os.listdir('test/test_run/v/id_1/analysis/perp'))
    
    def test_time_analysis(self, run):
        run.lab_frame = False
        run.time_analysis()
        assert ('corr_time.csv' in os.listdir('test/test_run/v/id_1/analysis/time'))
        assert ('corr_time.pdf' in os.listdir('test/test_run/v/id_1/analysis/time'))
        assert run.field_real_space.shape == (run.nt, run.nx, run.ny)
        assert run.time_corr.shape == (run.nt_slices, run.time_slice,
                                       run.nx, run.ny)
        assert ('corr_fns' in os.listdir('test/test_run/v/id_1/analysis/time'))
        assert ('time_fit_it_0_ix_0.pdf' in 
                os.listdir('test/test_run/v/id_1/analysis/time/corr_fns'))

    def test_time_analysis_lab_frame(self, run):
        run.lab_frame = True
        run.time_analysis()
        assert ('corr_time.csv' in os.listdir('test/test_run/v/id_1/analysis/time_lab_frame'))
        assert ('corr_time.pdf' in os.listdir('test/test_run/v/id_1/analysis/time_lab_frame'))
        assert run.field_real_space.shape == (run.nt, run.nx, run.ny)
        assert run.time_corr.shape == (run.nt_slices, run.time_slice,
                                       run.nx, run.ny)
        assert ('corr_fns' in os.listdir('test/test_run/v/id_1/analysis/time_lab_frame'))
        assert ('time_fit_it_0_ix_0.pdf' in 
                os.listdir('test/test_run/v/id_1/analysis/time_lab_frame/corr_fns'))

    def test_field_normalize_time(self, run):
        run.field_normalize_time()
        assert run.field_real_space_norm.shape == (run.nt, run.nx, run.ny)

    def test_time_norm_mask(self, run):
        run.time_corr = np.ones([5, 9, 5, 5])
        run.time_norm_mask(0)
        assert np.abs(run.time_corr[0,4,0,2] - 1./45.) < 1e-5

    def test_par_analysis(self, run):
        run.field_real_space = np.random.randint(0,10,size=[51,5,5,9])
        run.ntheta = 9
        run.par_analysis()
        assert run.par_corr.shape == (51,5,5,9)
        assert ('par_fit_params.csv' in os.listdir('test/test_run/v/id_1/analysis/parallel'))
        assert ('par_fit_summary.dat' in os.listdir('test/test_run/v/id_1/analysis/parallel'))
        assert ('par_fit_length_vs_time_slice.pdf' in 
                os.listdir('test/test_run/v/id_1/analysis/parallel'))
        assert ('par_fit_wavenumber_vs_time_slice.pdf' in 
                os.listdir('test/test_run/v/id_1/analysis/parallel'))
        assert ('par_fit_it_0.pdf' in 
                os.listdir('test/test_run/v/id_1/analysis/parallel/corr_fns'))

    def test_calculate_l_par(self, run):
        run.calculate_l_par()
        assert run.l_par.shape[0] == len(run.theta)

    def test_calculate_par_corr(self, run):
        run.field_real_space = np.ones([51,5,5,9])
        run.ntheta = 9
        run.calculate_l_par()
        run.calculate_par_corr()
        assert np.abs(np.abs(run.l_par[1] - run.l_par[0]) - 
                np.abs(run.l_par[-1]/(run.ntheta-1))) < 1e-5
        assert run.par_corr.shape == (51,5,5,9)

    def test_write_field(self, run):
        run.write_field()
        assert ('ntot_t.cdf' in os.listdir('test/test_run/v/id_1/analysis/write_field'))

    def test_write_field_lab_frame(self, run):
        run.lab_frame = True
        run.write_field()
        assert ('ntot_t_lab_frame.cdf' in os.listdir('test/test_run/v/id_1/analysis/write_field'))

    def test_make_film(self, run):
        run.film_lim = [1,1]
        run.make_film()
        assert ('ntot_t_spec_0_0000.png' in os.listdir('test/test_run/v/id_1/analysis/film/film_frames'))
        assert ('ntot_t_spec_0.mp4' in os.listdir('test/test_run/v/id_1/analysis/film/'))
        
        im = Image.open('test/test_run/v/id_1/analysis/film/film_frames/ntot_t_spec_0_0000.png')
        size = im.size
        assert size[0] % 2 == 0
        assert size[1] % 2 == 0

    def test_make_film_lab_frame(self, run):
        run.film_lim = [1,1]
        run.lab_frame = True
        run.make_film()
        assert ('ntot_t_spec_0_0000.png' in os.listdir('test/test_run/v/id_1/analysis/film_lab_frame/film_frames'))
        assert ('ntot_t_spec_0.mp4' in os.listdir('test/test_run/v/id_1/analysis/film_lab_frame/'))
        
        im = Image.open('test/test_run/v/id_1/analysis/film_lab_frame/film_frames/ntot_t_spec_0_0000.png')
        size = im.size
        assert size[0] % 2 == 0
        assert size[1] % 2 == 0


















