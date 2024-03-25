# basically you need to add a linux to put in
# a BLAS package in the background
# i don't think you can put in a BLAS package through 
# python only
FROM ubuntu:20.04 as base

# Install Python 3.9 and set it as default
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \ 
    apt-get install -yf python3.9 python3.9-dev python3-pip 
    # && \
    # ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    # python3 --version

RUN pip3 install cobra==0.29.0
RUN pip3 install arviz==0.13.0
RUN pip3 install scipy==1.9.3
RUN pip3 install pandas==1.5.3 

RUN pip3 install pymc==4.3.0
RUN pip3 install aesara==2.8.7

# RUN pip3 install cloudpickle==2.2.0
RUN pip3 install pickle5

RUN pip3 install numpy==1.24

RUN apt-get update -qqy && apt-get install -qqy libopenblas-dev gfortran
# RUN apk add --no-cache --update-cache gfortran build-base wget libpng-dev openblas-dev

RUN mkdir -p /putidaBMCA
WORKDIR /putidaBMCA
COPY putidabmca ./putidabmca/
#COPY data ./data/
#COPY emll ./emll/

ENV PYTHONPATH /putidaBMCA

ENTRYPOINT ["python3", "/putidaBMCA/putidabmca/main.py"]