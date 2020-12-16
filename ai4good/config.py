import os
import datetime

# class for secrets management
class FlaskConfig:
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URI')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ['SECRET_KEY']
    REMEMBER_COOKIE_DURATION = datetime.timedelta(minutes=30)

# class for app setup
class ModelConfig:
    MAX_CONCURRENT_MODELS = 30
    HISTORY_SIZE = 100
    INPUT_PARAMETER_TIMEOUT = 60*30 # in seconds
    CACHE_TIMEOUT = 60*60*2  # in seconds
