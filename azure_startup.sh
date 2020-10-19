apt-get install -y gcc
apt-get install -y build-essential libssl-dev libffi-dev python3.6-dev
apt-get install -y python-numba python3-dask
pip install -r requirements.txt
gunicorn --bind=0.0.0.0 --timeout 600 --chdir ai4good/webapp server:flask_app
