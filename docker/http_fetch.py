"""Trusted one-request HTTP broker. Never imports or executes target data."""
import http.client
import ipaddress
import json
import sys

host, port, path, tls, timeout, limit = sys.argv[1:]
host = str(ipaddress.ip_address(host))
assert path.startswith('/') and not path.startswith('//')
assert '\r' not in path and '\n' not in path
connection_type = http.client.HTTPSConnection if tls == 'true' else http.client.HTTPConnection
connection = connection_type(host, int(port), timeout=float(timeout))
try:
    connection.request('GET', path, headers={'User-Agent': 'authorized-ctf-runner/0.1', 'Connection': 'close'})
    response = connection.getresponse()
    print(json.dumps({'http_status': response.status, 'headers': response.getheaders()}), flush=True)
    # No redirects, DNS, proxy settings, decompression, or further requests.
    remaining = int(limit)
    while remaining:
        chunk = response.read(min(4096, remaining))
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        remaining -= len(chunk)
finally:
    connection.close()
