from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db_sqlalchemy = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
