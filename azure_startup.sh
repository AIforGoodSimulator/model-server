apt-get install -y gcc
apt-get install -y build-essential libssl-dev libffi-dev 
apt-get install -y python3-blosc xterm zip
python -m pip install --upgrade pip
pip install -r requirements.txt
gunicorn --bind=0.0.0.0 --timeout 600 --chdir ai4good/webapp server:flask_app