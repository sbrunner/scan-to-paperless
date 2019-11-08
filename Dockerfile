FROM ubuntu:cosmic as base-dist

ENV DEBIAN_FRONTEND=noninteractive
RUN \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends \
      python3 graphicsmagick pdftk-java vim \
      tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-eng \
      libimage-exiftool-perl software-properties-common \
      python3-pip python3-setuptools && \
  python3 -m pip install PyYaml numpy scipy scikit-image opencv-python-headless deskew && \
  apt-get auto-remove --assume-yes python3-pip python3-setuptools && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/* /root/.cache /var/cache/*

CMD ["/opt/process"]

VOLUME /source \
    /destination

ENV LANG=C.UTF-8


FROM base-dist as tests-dist

RUN \
  . /etc/os-release && \
  apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
    python3-wheel python3-pip python3-setuptools poppler-utils ghostscript graphviz
RUN python3 -m pip install 'pytest<4.0.0' pylint pyflakes bandit mypy codespell coverage pytest-profiling

WORKDIR /opt


FROM base-dist as base

COPY process /opt/


FROM tests-dist as tests

COPY process /opt/
COPY .pylintrc mypy.ini setup.cfg /opt/
RUN touch __init__.py


FROM base as all

RUN \
  . /etc/os-release && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
      tesseract-ocr-all && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/* /var/cache/*
