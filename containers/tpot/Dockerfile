FROM autoautoml-core:latest
ADD requirements.txt /tmp/core/tpot-requirements.txt
RUN python -m pip install -r tpot-requirements.txt
ADD container_settings.json /tmp/container_settings.json
ADD project.py /tmp/core/
