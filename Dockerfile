FROM ubuntu:20.04 as base-dist

ENV DEBIAN_FRONTEND=noninteractive
RUN \
    apt update && \
    apt install --assume-yes --no-install-recommends \
    graphicsmagick pdftk-java \
    tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-eng \
    libimage-exiftool-perl software-properties-common python3-pip ghostscript && \
    apt clean && \
    rm --recursive --force /var/lib/apt/lists/* /root/.cache /var/cache/*

COPY requirements.txt /tmp/
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --requirement=/tmp/requirements.txt && \
    rm --recursive --force /tmp/*

VOLUME /source \
    /destination

ENV LANG=C.UTF-8

WORKDIR /opt

FROM base-dist as tests-dist

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install --assume-yes --no-install-recommends \
    poppler-utils ghostscript graphviz && \
    apt-get clean && \
    rm --recursive --force /var/lib/apt/lists/* /root/.cache /var/cache/*

COPY requirements-dev.txt /tmp/
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --requirement=/tmp/requirements-dev.txt && \
    rm --recursive --force /tmp/*

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
