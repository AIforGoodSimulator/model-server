apt-get install -y gcc
apt-get install -y build-essential libssl-dev libffi-dev 
apt-get install -y python3-blosc xterm zip
python -m pip install --upgrade pip
pip install -r requirements.txt

# Upload the dev code to all kubernetes dash distributed pods
# Note if ALL the pods are not in a STATUS of 'Running' this may cause problems 
# as you will have different versions of code running across pods.

./upload.sh
gunicorn --bind=0.0.0.0 --timeout 600 --chdir ai4good/webapp server:flask_app