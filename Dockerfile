FROM ubuntu:22.04 AS upstream

FROM upstream AS base-all
SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get upgrade --yes \
    && apt-get install --assume-yes --no-install-recommends python3-pip gnupg fonts-dejavu-core

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

CMD ["scan-process"]

FROM upstream AS tests-node-modules

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

WORKDIR /src

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends apt-transport-https gnupg curl ca-certificates
RUN . /etc/os-release \
    && echo "deb https://deb.nodesource.com/node_18.x ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/nodesource.list \
    && curl --silent https://deb.nodesource.com/gpgkey/nodesource.gpg.key > /etc/apt/sources.list.d/nodesource.gpg \
    && apt-key add /etc/apt/sources.list.d/nodesource.gpg
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends nodejs

COPY package.json package-lock.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm install

FROM tests-dist AS tests

SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

COPY . ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --no-deps --editable .

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends gnupg

COPY --from=tests-node-modules /etc/apt/sources.list.d/nodesource* /etc/apt/sources.list.d/

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-key add /etc/apt/sources.list.d/nodesource.gpg \
    && apt-get update \
    && apt-get install --assume-yes --no-install-recommends nodejs \
    && echo "For Chrome installed by Pupetter" \
    && DEBIAN_FRONTEND=noninteractive apt-get install --assume-yes --no-install-recommends \
        libx11-6 libx11-xcb1 libxcomposite1 libxcursor1 \
        libxdamage1 libxext6 libxi6 libxtst6 libnss3 libcups2 libxss1 libxrandr2 libasound2 libatk1.0-0 \
        libatk-bridge2.0-0 libpangocairo-1.0-0 libgtk-3.0 libxcb-dri3-0 libgbm1 libxshmfence1

COPY package.json package-lock.json ./
COPY --from=tests-node-modules /src/node_modules ./node_modules
COPY --from=tests-node-modules /root/.cache/puppeteer /root/.cache/puppeteer
COPY tests/screenshot.js ./

FROM base as all

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get install --assume-yes --no-install-recommends tesseract-ocr-all
