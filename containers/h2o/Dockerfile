FROM autoautoml-core:latest
RUN apt-get update && apt-get install -y openjdk-11-jdk
#--no-install-recommends openjdk-11-jdk
# && rm -rf /var/lib/apt/lists/*
ADD requirements.txt /tmp/core/h2o-requirements.txt
RUN python -m pip install -r h2o-requirements.txt
ADD container_settings.json /tmp/container_settings.json
ADD project.py /tmp/core/
