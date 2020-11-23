import os

# class for basic setup and secrets management
class BaseConfig:
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ['SECRET_KEY']
    MAX_CONCURRENT_MODELS = 30
    HISTORY_SIZE = 100
    INPUT_PARAMETER_TIMEOUT = 60*30 # in seconds
    CACHE_TIMEOUT = 60*60*2  # in seconds (not used yet)
