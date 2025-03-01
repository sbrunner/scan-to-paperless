FROM ubuntu:25.04 AS upstream

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get upgrade --yes \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install --assume-yes --no-install-recommends tzdata

FROM upstream AS base-all
SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends fonts-dejavu-core gnupg python-is-python3 python3-pip python3-venv \
    && python3 -m venv /venv

ENV PATH=/venv/bin:$PATH

FROM base-all AS poetry

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=0.0.0.dev
WORKDIR /tmp
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --requirement=requirements.txt \
    && rm requirements.txt

COPY poetry.lock pyproject.toml ./
RUN poetry export --extras=process --output=requirements.txt \
    && poetry export --with=dev --output=requirements-dev.txt

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends curl

FROM base-all AS base-dist

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends \
        graphicsmagick pdftk-java \
        tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-eng \
        libimage-exiftool-perl software-properties-common ghostscript optipng pngquant libzbar0

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,from=poetry,source=/tmp,target=/tmp \
    python3 -m pip install --disable-pip-version-check --no-deps --requirement=/tmp/requirements.txt \
    && python3 -m pip freeze > /requirements.txt \
    && mkdir -p /source /destination /scan-codes

VOLUME /source \
    /destination \
    /scan-codes

WORKDIR /opt

ARG VERSION
ENV VERSION=$VERSION

FROM base-dist AS tests-dist

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends poppler-utils ghostscript graphviz

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,from=poetry,source=/tmp,target=/tmp \
    python3 -m pip install --disable-pip-version-check --no-deps --requirement=/tmp/requirements-dev.txt

FROM base-dist AS base

COPY scan_to_paperless scan_to_paperless/
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --no-deps --editable . \
    && pip freeze --all > /requirements.txt

ENV SCHEMA_BRANCH=master
CMD ["scan-process"]

FROM upstream AS tests-node-modules

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

WORKDIR /src

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends npm curl ca-certificates

FROM tests-dist AS tests

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

COPY . ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --no-deps --editable .

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends software-properties-common \
    && add-apt-repository ppa:savoury1/pipewire \
    && add-apt-repository ppa:savoury1/chromium \
    && apt-get update \
    && apt-get install --assume-yes --no-install-recommends chromium-browser npm

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

FROM base AS all

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends tesseract-ocr-all
