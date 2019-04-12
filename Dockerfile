FROM ubuntu:cosmic as builder

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
    python3-dev python3-wheel python3-pip python3-setuptools \
    curl unzip

RUN pip3 install PyYaml numpy scipy scikit-image opencv-python-headless

RUN curl http://galfar.vevb.net/store/deskew-125.zip > /tmp/deskew-125.zip && \
  unzip /tmp/deskew-125.zip -d /opt && \
  chmod +x /opt/Deskew/Bin/deskew &&\
  rm /tmp/deskew-125.zip


FROM ubuntu:cosmic as base

RUN \
  . /etc/os-release && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
      python3 graphicsmagick pdftk-java \
      tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-eng \
      libimage-exiftool-perl software-properties-common && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages/
COPY --from=builder /opt/Deskew /opt/Deskew

CMD ["/opt/process"]

VOLUME /source \
    /destination

ENV LANG=C.UTF-8

COPY process /opt/


FROM base as tests

RUN \
  . /etc/os-release && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
    python3-wheel python3-pip python3-setuptools poppler-utils ghostscript graphviz
RUN pip3 install 'pytest<4.0.0' pylint pyflakes bandit mypy codespell coverage pytest-profiling

WORKDIR /opt
COPY .pylintrc mypy.ini setup.cfg /opt/
RUN touch __init__.py


FROM base as all

RUN \
  . /etc/os-release && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
      tesseract-ocr-all && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/*


FROM all as experimental

RUN \
  . /etc/os-release && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends unpaper && \
  add-apt-repository ppa:stephane-brunner/cosmic && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends scantailor && \
  (apt-get install --assume-yes --no-install-recommends scantailor-advanced || true) && \
  (apt-get install --assume-yes --no-install-recommends scantailor-universal || true) && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/*
