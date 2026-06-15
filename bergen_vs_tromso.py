#!/usr/bin/env python3
"""Decide whether it is currently warmer in Bergen than in Tromso, using the
met.no compact endpoint, executed on a remote PySpell ESP32 device.

Device quirks discovered while testing the live API:

  * It serves ONE TCP connection at a time and each fetch takes several
    seconds, so we retry transient failures with a short sleep.
  * Its request buffer is tiny (~342 bytes) and anything beyond that is
    silently truncated, producing garbage results. We therefore keep every
    program small and split the work across several POST requests.
  * Its minimal HTTP server only reliably parses a request delivered as a
    single clean write, so we talk to it over a raw socket rather than via
    urllib (which got the request mis-parsed and returned the index page /
    "empty program").

Strategy:
  1. fetch Bergen air_temperature (one tiny device program)
  2. fetch Tromso air_temperature (one tiny device program)
  3. compare locally, then show() the verdict string on the device screen

The function returns True if Bergen is warmer than Tromso.
"""

import socket
import time

HOST = "100.65.240.107"
PORT = 80

BERGEN = (60.39, 5.32)
TROMSO = (69.65, 18.96)

COMPACT = ("https://api.met.no/weatherapi/locationforecast/2.0/compact"
           "?lat={lat}&lon={lon}")
TEMP_PATH = "properties.timeseries.0.data.instant.details.air_temperature"


def _post_once(code, lang, timeout):
    """Send one raw-socket POST /run and return the response body string."""
    body = code.encode("utf-8")
    request = (
        "POST /run?lang={lang}&timeout={t} HTTP/1.1\r\n"
        "Host: {host}\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: {clen}\r\n"
        "Connection: close\r\n\r\n"
    ).format(lang=lang, t=timeout, host=HOST, clen=len(body)).encode("ascii")
    request += body

    sock = socket.create_connection((HOST, PORT), timeout=60)
    try:
        sock.sendall(request)
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()

    _headers, _, payload = data.partition(b"\r\n\r\n")
    return payload.decode("utf-8", "replace").strip()


def run_program(code, lang="py", timeout=50):
    """POST a PySpell program to the device, retrying transient failures."""
    last = ""
    for _ in range(5):
        try:
            last = _post_once(code, lang, timeout)
        except Exception as exc:  # noqa: BLE001 - the link is flaky on purpose
            last = "error: {}".format(exc)
        transient = (not last) or any(
            s in last.lower()
            for s in ("connect failed", "network error", "field not found",
                      "empty program", "timeout", "timed out"))
        if not transient:
            return last
        time.sleep(3)
    return last


def fetch_temperature(lat, lon):
    """Ask the device to fetch+parse the current air temperature for a point."""
    url = COMPACT.format(lat=lat, lon=lon)
    code = 'fetch_json("{url}", "{path}")'.format(url=url, path=TEMP_PATH)
    last = ""
    for _ in range(5):
        last = run_program(code)
        try:
            return float(last)
        except ValueError:
            time.sleep(3)
    raise RuntimeError(
        "device did not return a number for ({}, {}): {!r}".format(
            lat, lon, last))


def show(text):
    """Draw text on the device screen; returns the device's raw response."""
    return run_program('show("{}")'.format(text))


def bergen_warmer_than_tromso():
    """Return True if Bergen is currently warmer than Tromso, and show the
    verdict on the device screen."""
    bergen = fetch_temperature(*BERGEN)
    tromso = fetch_temperature(*TROMSO)
    warmer = bergen > tromso
    verdict = "Bergen varmest" if warmer else "Tromso varmest"
    raw = show(verdict)
    print("Bergen: {} C".format(bergen))
    print("Tromso: {} C".format(tromso))
    print("Device show() raw response: {!r}".format(raw))
    return warmer


if __name__ == "__main__":
    result = bergen_warmer_than_tromso()
    print("Bergen warmer than Tromso? {}".format(result))
