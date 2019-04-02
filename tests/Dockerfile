FROM sbrunner/scan-to-paperless

RUN \
  . /etc/os-release && \
  apt-get update && \
  apt-get install --assume-yes --no-install-recommends poppler-utils ghostscript && \
  apt-get clean && \
  pip3 install pytest pylint pyflakes bandit mypy codespell coverage

WORKDIR /opt
COPY .pylintrc /opt
RUN touch __init__.py
COPY *.png /tests/
COPY test_* /tests/
