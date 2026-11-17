FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
# No model SDKs, credentials, Docker client or external research frameworks.
RUN groupadd --gid 10001 researcher && useradd --uid 10001 --gid 10001 --no-create-home researcher
COPY http_fetch.py /opt/ctf/http_fetch.py
USER 10001:10001
WORKDIR /work
CMD ["python3", "--version"]
