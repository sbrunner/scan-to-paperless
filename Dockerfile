FROM ubuntu:20.04 as base-dist

ENV DEBIAN_FRONTEND=noninteractive
RUN \
    apt update && \
    apt install --assume-yes --no-install-recommends \
    graphicsmagick pdftk-java \
    tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-eng \
    libimage-exiftool-perl software-properties-common python3-pip && \
    apt clean && \
    rm --recursive --force /var/lib/apt/lists/* /root/.cache /var/cache/*

COPY requirements-install.txt /tmp/
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --requirement=/tmp/requirements-install.txt && \
    rm --recursive --force /tmp/*

COPY Pipfile Pipfile.lock /tmp/
RUN cd /tmp && pipenv sync --system --clear && \
    rm --recursive --force /usr/local/lib/python3.*/dist-packages/tests/ /root/.cache/*

VOLUME /source \
    /destination

ENV LANG=C.UTF-8

WORKDIR /opt

FROM base-dist as tests-dist

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install --assume-yes --no-install-recommends \
    poppler-utils ghostscript graphviz && \
    apt-get clean && \
    rm --recursive --force /var/lib/apt/lists/* /root/.cache /var/cache/* && \
    cd /tmp && pipenv sync --system --clear --dev


FROM base-dist as base

COPY scan_to_paperless scan_to_paperless/
COPY setup.py README.md ./
RUN python3 -m pip install --no-cache-dir --editable .

CMD ["scan-process"]


FROM tests-dist as tests

COPY . ./
RUN python3 -m pip install --no-cache-dir --editable .


FROM base as all

RUN \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install --assume-yes --no-install-recommends \
    tesseract-ocr-all && \
    apt-get clean && \
    rm --recursive --force /var/lib/apt/lists/* /var/cache/*
