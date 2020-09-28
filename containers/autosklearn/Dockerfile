FROM autoautoml-core:latest
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    swig && rm -rf /var/lib/apt/lists/*
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
ADD requirements.txt /tmp/core/autosklearn-requirements.txt
RUN python -m pip install -r autosklearn-requirements.txt
ADD container_settings.json /tmp/container_settings.json
ADD project.py /tmp/core
