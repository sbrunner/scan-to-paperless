FROM ubuntu:cosmic

RUN \
  . /etc/os-release && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends python3-yaml graphicsmagick scantailor pdftk-java tesseract-ocr-fra libimage-exiftool-perl && \
  apt-get clean && \
  rm --recursive --force /var/lib/apt/lists/*

CMD ["/opt/scan"]

VOLUME /source \
    /destination

COPY scan /opt/
