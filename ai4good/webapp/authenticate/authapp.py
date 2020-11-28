from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.urls import url_parse
from urllib.parse import urlparse, urljoin, urlencode

from ai4good.utils.logger_util import get_logger
from ai4good.webapp.apps import db_sqlalchemy
from ai4good.webapp.authenticate.usermodel import User

logger = get_logger(__file__, 'DEBUG')

server_bp = Blueprint('main', __name__)

def is_safe_url(target):
    # without validation, system may be vulnerable to open redirects
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc


@server_bp.route('/')
def index():
    return redirect('/sim/')


@server_bp.route('/login/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    # obtain form data from dash page
    # note: dash is unsusceptible to CSRF attacks (https://github.com/plotly/dash/issues/141)
    form_data = request.form
    form_email = form_data.get('email')
    form_password = form_data.get('password')
    form_remember_me = form_data.get('remember_me')
    if (not form_email) | (not form_password):
        return redirect('/auth/')
    else:
        user = User.query.filter_by(username=form_email).first()
        if user is None or not user.check_password(form_password):
            error = 'Invalid username or password'
            logger.warn('Login error: {} for {}'.format(error, form_email))
            return redirect(url_for('/auth/', error=error))
        else:
            login_user(user, remember=form_remember_me)
            user.set_sid()
            #db_sqlalchemy.session.commit()
            next_page = request.args.get('next')
            if not next_page or url_parse(next_page).netloc != '':
                next_page = url_for('main.index')
            elif not is_safe_url(next_page):
                logger.warn('Attempt to redirect user to an external or/and unsafe site: {}'.format(next_page))
                next_page = url_for('main.index')
            return redirect(next_page)


@server_bp.route('/logout/')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))


@server_bp.route('/register/', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form_data = request.form
    form_email = form_data.get('email')
    form_password = form_data.get('password')
    if (not form_email) | (not form_password):
        return redirect('/auth/register/')
    else:
        user = User.query.filter_by(username=form_email).first()
        if user is None:
            user = User(username=form_email)
            user.set_password(form_password)
            db_sqlalchemy.session.add(user)
            db_sqlalchemy.session.commit()
            
            next_page = request.args.get('next')
            if not next_page or url_parse(next_page).netloc != '':
                next_page = url_for('main.login')
            elif not is_safe_url(next_page):
                logger.warn('Attempt to redirect user to an external or/and unsafe site: {}'.format(next_page))
                next_page = url_for('main.login')
            return redirect(next_page)
        else:
            error = f'Email address {form_email} has already been registered'
            logger.warn('Register error: {}'.format(error))
            if (not error):
                return redirect('/auth/register/')
            else:
                return redirect('/auth/register/' + '?' + urlencode({'error': error}))
