FROM python:3.8.5
MAINTAINER AI4Good "https://twitter.com/AI4Good"

RUN apt-get update -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# need to add waitress in requirements.txt?
RUN pip install --no-cache-dir waitress
RUN pip install --no-cache-dir gunicorn

WORKDIR /usr/bin/

# install dependencies for orca
RUN apt-get install -y --no-install-recommends \
    wget \
    xvfb \
    xauth \
    libgtk2.0-0 \
    libxtst6 \
    libxss1 \
    libgconf-2-4 \
    libnss3 \
    libasound2

# download the orca appimage binary and make executable for extraction
RUN wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O orca.AppImage
RUN chmod +x orca.AppImage

# extract the AppImage and delete AppImage
RUN orca.AppImage --appimage-extract
RUN rm orca.AppImage


WORKDIR /model-server

# create script to invoke the orca executable and set environment
RUN printf '#!/bin/bash \nxvfb-run --auto-servernum --server-args "-screen 0 640x480x24" /usr/bin/squashfs-root/app/orca "$@" \n' > orca
RUN chmod +x orca

ENV PYTHONPATH="$PYTHONPATH:/model-server"
ENV PATH="$PATH:/model-server"
COPY . .

EXPOSE 8000 8000

# run webserver background process, do we want to run this as a separate container?
# CMD nohup waitress-serve --port 8050 --host 0.0.0.0 ai4good.webapp.server:flask_app

# executes console_runner.py
# ENTRYPOINT ["python", "./ai4good/runner/console_runner.py"]

# default  command line params go here
# CMD ["--profile", "custom", "--save_plots", "--save_report"]
