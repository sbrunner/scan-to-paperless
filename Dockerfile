FROM ubuntu:cosmic

RUN \
  . /etc/os-release && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends python3-yaml graphicsmagick scantailor pdftk-java tesseract-ocr tesseract-ocr-fra libimage-exiftool-perl unpaper unzip curl && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/*

RUN \
  curl http://galfar.vevb.net/store/deskew-125.zip > /tmp/deskew-125.zip && \
  unzip /tmp/deskew-125.zip -d /opt && \
  chmod +x /opt/Deskew/Bin/deskew &&\
  rm /tmp/deskew-125.zip

CMD ["/opt/postprocess"]

VOLUME /source \
    /destination

COPY postprocess /opt/
