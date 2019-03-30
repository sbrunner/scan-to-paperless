FROM sbrunner/scan-to-paperless

RUN pip3 install pytest pylint pyflakes bandit mypy codespell coverage

WORKDIR /opt
COPY .pylintrc /opt
RUN touch __init__.py
COPY *.png /tests/
COPY test_* /tests/
