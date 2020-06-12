
pip install -r requirements.txt

Go to root of the repo, then run

Windows: 
    
    set PYTHONPATH=%PYTHONPATH%;.
    
Linux:

    export PYTHONPATH="${PYTHONPATH}:."
    
    
To get commandline help:
    
    python ai4good/runner/console_runner.py -h
    
    
Example execution (default camp is used):

    python ai4good/runner/console_runner.py --profile custom --save_plots --save_report
    
CSV Report is  saved in fs/reports, plots are in fs/figs, model result cache in fs/model_results.

Parameters are in fs/params + profile configuration in code right now, but to move to some kind of database in the future.