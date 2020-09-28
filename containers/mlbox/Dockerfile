FROM autoautoml-core:latest
RUN apt-get update && apt-get install -y --no-install-recommends gcc gfortran python-dev libopenblas-dev liblapack-dev cython && rm -rf /var/lib/apt/lists/*
ADD deps/MLBox MLBox
RUN cd MLBox && python setup.py install
ADD container_settings.json /tmp/container_settings.json
ADD project.py /tmp/core/
