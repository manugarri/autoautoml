FROM autoautoml-core:latest
ADD requirements.txt /tmp/core/pycaret-requirements.txt
RUN python -m pip install -r pycaret-requirements.txt
ADD container_settings.json /tmp/container_settings.json
ADD project.py /tmp/core/
