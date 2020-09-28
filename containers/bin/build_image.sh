#! /bin/bash
cd $CONTAINER && docker build -f Dockerfile -t $CONTAINER:latest .
