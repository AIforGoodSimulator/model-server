import json
import hashlib
import secrets
import datetime
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
    simulation = db_sqlalchemy.relationship('Simulation', backref='user')
    
    def set_sid(self):
        self._sid = secrets.token_urlsafe(64)  # user current session id

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)


class Simulation(db_sqlalchemy.Model):
    id = db_sqlalchemy.Column(db_sqlalchemy.Integer, primary_key=True)
    parent_id = db_sqlalchemy.Column(db_sqlalchemy.Integer, db_sqlalchemy.ForeignKey('user.id'))
    user_sid = db_sqlalchemy.Column(db_sqlalchemy.String(64), index=True, nullable=False) # user session id
    datetime_start = db_sqlalchemy.Column(db_sqlalchemy.DateTime(timezone=True), default=datetime.datetime.utcnow())
    datetime_end = db_sqlalchemy.Column(db_sqlalchemy.DateTime(timezone=True), default=0)
    sim_id = db_sqlalchemy.Column(db_sqlalchemy.String(64), index=True) # hash of simulation input
    sim_cfg_blob = db_sqlalchemy.Column(db_sqlalchemy.LargeBinary) # simulation config blob
    sim_inp_blob = db_sqlalchemy.Column(db_sqlalchemy.LargeBinary) # simulation input blob
    sim_out_blob = db_sqlalchemy.Column(db_sqlalchemy.LargeBinary) # simulation output blob
    
    user = db_sqlalchemy.relationship('User', backref=db_sqlalchemy.backref('simulations', lazy=True))

    def __init__(self):
        user_sid = user.sid
        assert datetime_start
        if sim_id:
            assert get_config

    def _hash(self, serialised_json) -> str:
        hash_object = hashlib.sha3_512(serialised_json.encode('UTF-8'))
        return hash_object.hexdigest()
        
    def check_input(sim_input):
        hash_input = _hash(json.dumps(sim_input))
        simulation_done = Simulation.query.filter_by(sim_id=hash_input).first()
        if simulation_done:
            return simulation_done.id

    def get_config(self):
        return json.loads(self.sim_cfg_blob)
        
    def get_input(self):
        assert get_config
        return json.loads(self.sim_inp_blob)
            
    def get_output(self):
        assert get_config
        assert get_input
        assert sim_id
        return json.loads(self.sim_out_blob)

    def set_config(self, sim_config):
        self.sim_inp_blob = json.dumps(sim_config)
    
    def set_input(self, sim_input):
        self.sim_inp_blob = json.dumps(sim_input)
        self.sim_id = _hash(self.sim_inp_blob)

    def set_output(self, sim_output):
        self.sim_out_blob = json.dumps(sim_output)
        self.datetime_end = datetime.datetime.utcnow()
       
    def __repr__(self):
        return '<Simulation {}>'.format(self.sim_id)
