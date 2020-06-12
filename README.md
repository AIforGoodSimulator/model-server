### Install 

To start install required packages with 

    pip install -r requirements.txt
    
or first create and activate virtual environment

* On macOS and Linux:

        python3 -m venv env
        source env/bin/activate
        pip install -r requirements.txt

* On Windows:

        py -m venv env
        .\env\Scripts\activate
        pip install -r requirements.txt


    

Go to root of the repo and run (or configure corresponding env. vars)

Windows: 
    
    set PYTHONPATH=%PYTHONPATH%;.
    
Linux:

    export PYTHONPATH="${PYTHONPATH}:."
    
### Command line execution 
    
To get commandline help:
    
    python ai4good/runner/console_runner.py -h
    
    
Example execution (default camp is used):

    python ai4good/runner/console_runner.py --profile custom --save_plots --save_report
    
CSV Report is  saved in fs/reports, plots are in fs/figs, model result cache in fs/model_results.

Parameters are in fs/params + profile configuration in code right now, but to move to some kind of database in the future.


### Webapp

Webapp can be started from PyCharm by running server.py main method or from terminal:

    waitress-serve --port 8050 --host 0.0.0.0 ai4good.webapp.server:flask_app 