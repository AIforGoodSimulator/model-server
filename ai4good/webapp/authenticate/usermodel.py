import secrets
from flask_login import UserMixin
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
from ai4good.webapp.apps import db_sqlalchemy
from ai4good.webapp.apps import login


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class User(UserMixin, db_sqlalchemy.Model):
    id = db_sqlalchemy.Column(db_sqlalchemy.Integer, primary_key=True)
    username = db_sqlalchemy.Column(db_sqlalchemy.String(64), index=True, unique=True)
    password_hash = db_sqlalchemy.Column(db_sqlalchemy.String(128))
    sid = db_sqlalchemy.Column(db_sqlalchemy.String(64))
    #simulation = db_sqlalchemy.relationship('Simulation', backref='user')
    
    def set_sid(self):
        self._sid = secrets.token_urlsafe(64)  # simulation session id

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)

'''    
class Simulation(user_id, sid=None, db_sqlalchemy.Model):
    id = db_sqlalchemy.Column(db_sqlalchemy.Integer, primary_key=True)
    user_id = Column(db_sqlalchemy.Integer, db_sqlalchemy.ForeignKey('user.id'), nullable=False)
    sid = db_sqlalchemy.Column(db_sqlalchemy.String(64), nullable=False)
    simulation_output = db_sqlalchemy.Column(db_sqlalchemy.LargeBinary)

    def __init__(self):
        user = User.query.filter_by(id=user_id).first()
        if not sid:
            self.sid = user.sid # simulation session id
        
    def set_simulation_output(self, simulation_output):
        self.simulation_output = simulation_output

       
    def __repr__(self):
        return '<Simulation {}>'.format(self.sid)

class Output(db_sqlalchemy.Model):
'''