FROM debian:buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Apt-get requirements
RUN apt-get update && apt-get install -y \
    # Pipenv
        python3 python3.7 python3-pip \
    # UKB
        libboost-all-dev libboost-program-options-dev \
    # Python build requirements
        python3-dev python3.7-dev build-essential libffi-dev \
    # Build requirements for HFST + Omorfi
        git autoconf automake libtool file \
    # Build requirements for HFST
        flex bison libglib2.0-0 libglib2.0-dev \
    # Java stuff for Scorer + IMS
        curl wget unzip zip openjdk-10-jdk


# Pipenv
RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3
RUN pip3 install pipenv

# Java stuff for Scorer + IMS
RUN curl -s "https://get.sdkman.io" | bash
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && sdk install gradle 4.9'
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && sdk install maven'

# Re-fixup Python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python

# Evaluation framework setup
COPY . /app
WORKDIR /app

RUN set -ex && pipenv install --deploy --system
RUN bash ./compile_scorer.sh

# NLTK resources
RUN python -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"

# WSD system setup
RUN python ukb.py fetch
RUN bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && python supwsd.py fetch'
RUN python ctx2vec.py fetch
