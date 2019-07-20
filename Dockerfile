FROM registry.gitlab.com/frankier/finntk/full-deb:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Apt-get requirements
RUN apt-get update && apt-get install -y \
    # Pipenv
        python3 python3.7 python3-pip \
    # UKB
        libboost-all-dev libboost-program-options-dev libxml-libxml-perl \
    # Python build requirements
        python3-dev python3.7-dev build-essential libffi-dev \
    # Build requirements for HFST + Omorfi
        git autoconf automake libtool file \
    # Build requirements for HFST
        flex bison libglib2.0-0 libglib2.0-dev \
    # Java stuff for Scorer + IMS
        curl wget unzip zip openjdk-11-jdk \
    # STIFF/opencc
        libopencc-dev


# Poetry + Pipenv
RUN set -ex && curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3
RUN ~/.poetry/bin/poetry config settings.virtualenvs.create false
RUN pip3 install pipenv

# Java stuff for Scorer + IMS
RUN curl -s "https://get.sdkman.io" | bash
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && \
        sdk install maven && \
        sdk flush archives && \
        sdk flush temp'

# Fixup Python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python

# Script driven stuff begins
WORKDIR /app

# Scorer
COPY ./compile_scorer.sh /app/
COPY ./support/scorer /app/support/scorer
RUN bash ./compile_scorer.sh

# Pipenv requirements
COPY ./pyproject.toml /app/
COPY ./poetry.lock /app/
RUN pip3 install --upgrade pip==19.0.3
RUN ~/.poetry/bin/poetry install --no-interaction

# NLTK resources
RUN python -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"

# FinnTK resources
RUN python -m finntk.scripts.bootstrap_all

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

# SIF
COPY ./fetchers/sif.py /app/fetchers/
RUN python fetchers/sif.py

# STIFF post_install
RUN python -m stiff.scripts.post_install

# Evaluation framework setup
COPY . /app

# Set up Python path
RUN echo "/app/" > "/usr/local/lib/python3.7/dist-packages/wsdeval.pth"
