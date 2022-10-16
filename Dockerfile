FROM ubuntu:22.04 as base-all
SHELL ["/bin/bash", "-o", "pipefail", "-cux"]

RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update \
    && apt-get upgrade --yes

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends python3-pip gnupg fonts-dejavu-core

FROM base-all as poetry

ENV POETRY_DYNAMIC_VERSIONING_BYPASS=True
WORKDIR /tmp
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache \
    python3 -m pip install --disable-pip-version-check --requirement=requirements.txt \
    && rm requirements.txt

COPY poetry.lock pyproject.toml ./
RUN poetry export --output=requirements.txt \
    && poetry export --with=dev --output=requirements-dev.txt

RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get install --assume-yes --no-install-recommends curl \
    && curl https://notesalexp.org/debian/alexp_key.asc > alexp_key.asc

FROM base-all as base-dist

RUN --mount=type=bind,from=poetry,source=/tmp,target=/tmp2 \
    . /etc/os-release \
    && echo "deb https://notesalexp.org/tesseract-ocr5/${VERSION_CODENAME}/ ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/notesalexp.list \
    && cp /tmp2/alexp_key.asc /tmp/alexp_key.asc \
    && apt-key add /tmp/alexp_key.asc

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

ENV LANG=C.UTF-8

WORKDIR /opt

ARG VERSION
ENV VERSION=$VERSION

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
