"""Web-mode server: controls must work while the video stream is open."""

import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
import play_remote as pr


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    pr.state.current_frame = np.zeros((480, 640, 3), np.uint8)
    threading.Thread(target=pr.run_web_server, args=(port, "127.0.0.1"), daemon=True).start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"


def test_page_is_complete(server):
    body = urllib.request.urlopen(server + "/").read()
    assert body == pr.HTML_PAGE.encode("utf-8")
    assert body.rstrip().endswith(b"</html>")


def test_action_served_while_streaming(server):
    host, port = server.replace("http://", "").split(":")
    stream = socket.create_connection((host, int(port)))
    stream.sendall(b"GET /stream.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
    assert stream.recv(64)
    try:
        assert urllib.request.urlopen(server + "/action?act=2", timeout=3).read() == b"OK"
        assert pr.state.human_action == 2
    finally:
        stream.close()


@pytest.mark.parametrize("bad", ["7", "-1", "abc"])
def test_invalid_action_rejected(server, bad):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(server + f"/action?act={bad}", timeout=3)
    assert exc.value.code == 400
