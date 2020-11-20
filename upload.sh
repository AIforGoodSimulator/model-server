#!/bin/sh


# Script that will be run on each dask worker to populate the ai4Good code remotely

rm upload.txt;
zip -r upload.txt .env ai4good fs -x '*.pkl'
python upload.py 
