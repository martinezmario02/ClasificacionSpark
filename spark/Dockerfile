FROM docker.io/bitnami/spark:3.3.0

USER root
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN pip3 install numpy

USER 1001