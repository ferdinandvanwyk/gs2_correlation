# Standard
import os
import pytest

# Third Party

# Local
from gs2_correlation.configuration import Configuration

class TestClass(object):

    def setup_class(self):
        os.system('tar -zxf test/test_run.tar.gz -C test/.')

    def teardown_class(self):
        os.system('rm -rf test/test_run')

    @pytest.fixture(scope='function')
    def conf(self):
        return Configuration('test/test_run/config.ini')

    def test_init(self, conf):
        assert type(conf.config_file) == str

    def test_read(self, conf):
        conf.read_config()
        assert type(conf.in_file) == str
        assert type(conf.file_ext) == str
        assert type(conf.in_field) == str
        assert type(conf.analysis) == str
        assert type(conf.out_dir) == str
        assert type(conf.interpolate) == bool
        assert type(conf.zero_bes_scales) == bool
        assert (type(conf.spec_idx) == int or type(conf.spec_idx) == type(None))
        assert (type(conf.theta_idx) == int or type(conf.theta_idx) == 
                                               type(None))
        assert type(conf.amin) == float
        assert type(conf.vth) == float
        assert type(conf.rhoref) == float
        assert type(conf.pitch_angle) == float












