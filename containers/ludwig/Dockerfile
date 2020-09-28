FROM autoautoml-core:latest
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1-dev && rm -rf /var/lib/apt/lists/*
ADD requirements.txt /tmp/core/ludwig-requirements.txt
RUN python -m pip install -r ludwig-requirements.txt
ADD container_settings.json /tmp/container_settings.json
ADD project.py /tmp/core/
