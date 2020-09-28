FROM python:3.7
ENV PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    python-pip python-dev libsnappy-dev && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
ADD requirements.txt /tmp/core/requirements.txt
WORKDIR /tmp/core
RUN python -m pip install  -r requirements.txt
ADD tests /tests
ADD autoautoml /tmp/core/autoautoml
ADD run.py /tmp/core/
ADD project.py /tmp/core
ADD container_settings.json /tmp/container_settings.json
ENTRYPOINT ["python","run.py"]
