FROM debian:buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Pipenv
RUN apt-get update
RUN apt-get install -y python3 python3.7 python3-pip
RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3
RUN pip3 install pipenv

# Python build requirements
RUN apt-get install -y python3-dev build-essential libffi-dev

# Build requirements for HFST + Omorfi
RUN apk --no-cache add git autoconf automake libtool file

# Build requirements for HFST
RUN apk --no-cache add flex bison glib glib-dev

# Java stuff for Scorer + IMS
RUN apt-get install -y curl unzip zip openjdk-10-jdk
RUN curl -s "https://get.sdkman.io" | bash
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && sdk install gradle 4.9'

# Evaluation framework setup
COPY . /app
WORKDIR /app

RUN set -ex && pipenv install --deploy --system
RUN bash ./get_scorer.sh

# WSD system setup
RUN python ukb.py fetch
RUN python ims.py fetch
RUN python ctx2vec.py fetch
