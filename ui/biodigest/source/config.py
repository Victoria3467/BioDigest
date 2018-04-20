class Config(object):
    SECRET_KEY = 'key'


class ProductionConfig(Config):
    DEBUG = False


class DebugConfig(Config):
    DEBUG = True
