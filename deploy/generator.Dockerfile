# The world generator's web interface: `worldgen serve`, behind a password.
#
#     docker build -t generator -f deploy/generator.Dockerfile .
#     docker run -p 8000:8000 -e WORLDGEN_PASSWORD='something long' generator
#
# The build context is the repository root, where `pyproject.toml` is. The root
# `.dockerignore` belongs to the campaign image and keeps `worldgen/` out, so this image has
# its own beside it, `generator.Dockerfile.dockerignore`, which BuildKit reads in place of
# the root one.
#
# No volume: a generated world lives in the process's memory until it is downloaded, and a
# restart losing the last few is the cost of not running storage for them.

FROM python:3.12-slim

# Unprivileged. Nothing here writes anywhere but /tmp.
RUN useradd --create-home --uid 1000 app

WORKDIR /src

# The manifest first, so a source change does not reinstall numpy and scipy. The package is
# then installed over that with --no-deps, which needs nothing new.
COPY pyproject.toml ./
RUN mkdir worldgen && touch worldgen/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf worldgen build

COPY worldgen/ ./worldgen/
RUN pip install --no-cache-dir --no-deps . && rm -rf /src

USER app
WORKDIR /home/app

ENV PYTHONUNBUFFERED=1 \
    # matplotlib wants a writable cache directory and warns on every start without one.
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

EXPOSE 8000

# `/health` answers without the password, which is what makes it usable here.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"

# 0.0.0.0, because inside a container the loopback is unreachable from outside. `serve`
# refuses to bind it without WORLDGEN_PASSWORD, so a deploy that forgot the secret fails to
# start rather than serving the generator to anyone.
CMD ["sh", "-c", "exec worldgen serve --host 0.0.0.0 --port \"$PORT\" --no-open"]
