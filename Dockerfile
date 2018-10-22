FROM debian:buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Apt-get requirements
RUN apt-get update && apt-get install -y \
    # Pipenv
        python3 python3.6 python3-pip \
    # UKB
        libboost-all-dev libboost-program-options-dev \
    # Python build requirements
        python3-dev python3.6-dev build-essential libffi-dev \
    # Build requirements for HFST + Omorfi
        git autoconf automake libtool file \
    # Build requirements for HFST
        flex bison libglib2.0-0 libglib2.0-dev \
    # Java stuff for Scorer + IMS
        curl wget unzip zip openjdk-10-jdk


# Pipenv
RUN pip3 install pipenv

# Java stuff for Scorer + IMS
RUN curl -s "https://get.sdkman.io" | bash
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && sdk install maven'

# Script driven stuff begins
WORKDIR /app

# Scorer
COPY ./compile_scorer.sh /app/
COPY ./support/scorer /app/support/scorer
RUN bash ./compile_scorer.sh

# Pipenv requirements
COPY ./Pipfile* /app/
RUN set -ex && pipenv install --deploy --system

# NLTK resources
RUN python -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"

RUN mkdir /app/fetchers

# UKB
COPY ./fetchers/ukb.py /app/fetchers/
COPY ./support/ukb /app/support/ukb
RUN python fetchers/ukb.py fetch

# SupWSD
COPY ./fetchers/supwsd.py /app/fetchers/
COPY ./support/supWSD /app/support/supWSD
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && python fetchers/supwsd.py'

# Context2Vec
COPY ./fetchers/ctx2vec.py /app/fetchers/
COPY ./support/context2vec /app/support/context2vec
RUN python fetchers/ctx2vec.py

# Evaluation framework setup
COPY . /app
