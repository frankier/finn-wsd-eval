FROM debian:buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update
RUN apt-get install -y python3 python3.7 python3-pip
RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3
RUN pip3 install pipenv

RUN apt-get install -y curl unzip zip openjdk-10-jdk
RUN curl -s "https://get.sdkman.io" | bash
RUN sdk install gradle 4.9

COPY . /app
WORKDIR /app

RUN set -ex && pipenv install --deploy --system
RUN bash ./get_scorer.sh

RUN python ukb.py fetch
RUN python ims.py fetch
RUN python ctx2vec.py fetch
