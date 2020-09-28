CONTAINER=autokaggle-core make build
docker run --env-file .env $CONTAINER
