apt-get install -y gcc
pip install -r requirements.txt
gunicorn --bind=0.0.0.0 --timeout 600 --chdir ai4good/webapp server:flask_app
