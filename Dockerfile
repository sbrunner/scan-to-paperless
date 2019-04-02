FROM ubuntu:cosmic

COPY requirements.txt /tmp/

RUN \
  . /etc/os-release && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends python3-pip python3-setuptools graphicsmagick pdftk-java \
      tesseract-ocr tesseract-ocr-fra libimage-exiftool-perl unpaper unzip curl \
      software-properties-common && \
  add-apt-repository ppa:stephane-brunner/cosmic && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends scantailor scantailor-advanced && \
  apt-get clean && \
  pip3 install --requirement=/tmp/requirements.txt && \
  rm --recursive --force /var/lib/apt/lists/* && \
  curl http://galfar.vevb.net/store/deskew-125.zip > /tmp/deskew-125.zip && \
  unzip /tmp/deskew-125.zip -d /opt && \
  chmod +x /opt/Deskew/Bin/deskew &&\
  rm /tmp/deskew-125.zip

CMD ["/opt/process"]

VOLUME /source \
    /destination

ENV LANG=C.UTF-8

COPY process /opt/
