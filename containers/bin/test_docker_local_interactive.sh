#! /bin/bash
docker run --env-file experiment.env -it --entrypoint /bin/bash  $container
