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
    
### Azure deployment

First add azure remote

    git remote add azure https://ai4good-sim2.scm.azurewebsites.net:443/ai4good-sim2.git
    
 Note down deployment credentials from Deployment Center/Deployment Credentials on Azure portal for AI4Good-Sim2 app service.
 
 Now just do 
    
    git push azure master

enter credentials when prompted.
    
### Docker

Build:

    docker build -t model-server .

Test:

    docker run model-server python -m unittest discover -s ai4good/ -p "test_*.py"

Run Example:

    docker run model-server python ai4good/runner/console_runner.py --profile custom --save_plots --save_report

Container Command Line:

    docker run -it model-server /bin/bash

### Design overview

Model-server consists of following top level packages:

* models - various COVID-19 models and model registry
* params - parameter storage and retrieval
* runner - console runner
* webapp - web application runner / viewer

 #### models
 Every model needs to implement ai4good.models.model.Model abstract base class and basically just needs
 to implement run(params) method, where params object can be chosen by the model itself and usually 
 contains general parameters, camp specific parameters and model profile parameters. Model profiles
 are there to investigate and compare various regimes of the model and to help user to select best
 intervention scenario that mitigates COVID-19 spread.
 
 Model also responsible to provide hash of it's own parameter object so that model results can cached. 
 This functionality of saving/retrieving model result is provided by ModelResultStore and currently 
 stored on filesystem.  
 
 Model result is represented by ModelResult object which effectively just a free-from dictionary and
 can include some precomputed aggregations to speed up result rending later.
 
 
 
#### params
Provides abstract interface to parameter storage which is at the moment is based on csv files 
stored on local file system.  
 
    
#### runner

Contains console runner that can run a model for single profile or all profiles/camps in a batch.
Also contains console_utils to list currently cached models with human readable names.


#### webapp

Web application is built with dash/plotly and runs on Azure via gunicorn with multiple workers. There is also Redis
instance used to store some shared state, such as models currently executing and also hosting some page cache. There is
also cache on local disk that is used to store larger amounts of data. Webapp has model runner page and report pages.
Report page is model specific and allows to compare various intervention scenarios.  


### Tests
use run_tests cmd/sh to execute all tests

### Other instances

* Address: http://207.154.208.109:8050/sim/run_model

* Python Server - Waitress Python

Waitress is meant to be a production-quality pure-Python WSGI server with very acceptable performance. It has no dependencies except ones which live in the Python standard library. It runs on CPython on Unix and Windows under Python 2.7+ and Python 3.5+. It is also known to run on PyPy 1.6.0 on UNIX. It supports HTTP/1.0 and HTTP/1.1.

* How to run: 

        update app
        install python
        install pip3
        git clone
        install waitress
        run waitress
        
