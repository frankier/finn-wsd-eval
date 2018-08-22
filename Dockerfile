FROM debian:buster-slim

RUN apt-get update
RUN apt-get install -y python3 python3.7 python3-pip openjdk-10-jdk
RUN pip3 install pipenv
RUN pipenv install
RUN curl -s "https://get.sdkman.io" | bash
RUN sdk install gradle 4.9

RUN bash ./get_scorer.sh
RUN pipenv run python ukb.py fetch
RUN pipenv run python ims.py fetch
RUN pipenv run python ctx2vec.py fetch
