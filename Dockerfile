FROM ubuntu:22.04 as base-all
SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update \
    && apt-get upgrade --yes

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends python3-pip

FROM base-all as poetry

WORKDIR /tmp
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --requirement=requirements.txt \
    && rm requirements.txt

COPY poetry.lock pyproject.toml ./
RUN poetry export --output=requirements.txt \
    && poetry export --dev --output=requirements-dev.txt

FROM base-all as base-dist

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends \
    graphicsmagick pdftk-java \
    tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-eng \
    libimage-exiftool-perl software-properties-common ghostscript optipng

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,from=poetry,source=/tmp,target=/tmp \
    python3 -m pip install --disable-pip-version-check --no-deps --requirement=/tmp/requirements.txt

VOLUME /source \
    /destination

ENV LANG=C.UTF-8

WORKDIR /opt

FROM base-dist as tests-dist

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends poppler-utils ghostscript graphviz

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,from=poetry,source=/tmp,target=/tmp \
    python3 -m pip install --disable-pip-version-check --no-deps --requirement=/tmp/requirements-dev.txt

FROM base-dist as base

COPY scan_to_paperless scan_to_paperless/
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --no-deps --editable . \
    && pip freeze --all > /requirements.txt

CMD ["scan-process"]

FROM tests-dist as tests

COPY . ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --no-deps --editable .

FROM base as all

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends tesseract-ocr-all
