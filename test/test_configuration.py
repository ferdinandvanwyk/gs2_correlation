from gs2_correlation.configuration import Configuration
import pytest

class TestClass(object):

    @pytest.fixture(scope='function')
    def conf(self):
        return Configuration('config.ini')

    def test_init(self, conf):
        assert type(conf.config_file) == str

    def test_read(self, conf):
        conf.read_config()
        print(os.listdir())
        assert type(conf.interpolate) == bool

