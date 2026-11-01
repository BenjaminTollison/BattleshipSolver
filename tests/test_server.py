"""JSON boundary and lightweight real HTTP integration tests (stdlib only)."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import engine as e
import server


def test_board_from_payload_round_trip():
    payload = {'size': 4, 'grid': [0] * 16, 'fleet': [['A', 2]],
               'sunk': [], 'no_touch': True}
    payload['grid'][3] = e.MISS
    b = server.board_from(payload)
    assert b.size == 4 and b.no_touch
    assert b.get(0, 3) == e.MISS
    assert b.remaining_ships() == [('A', 2)]


def test_analyze_json_schema_and_probabilities():
    payload = {'size': 4, 'fleet': [['Patrol', 2]], 'samples': 250,
               'grid': [e.MISS] * 16, 'parity': True}
    payload['grid'][5] = e.HIT
    payload['grid'][6] = e.UNKNOWN
    response = server.analyze(payload)
    assert response['consistent']
    assert len(response['probs']) == len(response['fits']) == 16
    assert response['samples'] == 250
    assert response['remaining'] == [('Patrol', 2)]
    assert response['rec']['best_name'] == 'B3'
    assert response['probs'][6] == 1.0
    assert response['probs'][5] == 0.0


def test_analyze_clamps_sample_count():
    response = server.analyze({'size': 3, 'fleet': [['A', 1]], 'samples': 1})
    assert response['samples'] == 200


def test_sink_json_endpoint_converts_hit_line():
    grid = [0] * 16
    grid[5] = grid[6] = e.HIT
    response = server.sink({'size': 4, 'grid': grid, 'fleet': [['Patrol', 2]],
                            'sunk': [], 'ship': 'Patrol', 'last': 6})
    assert response['converted'] == [5, 6]
    assert response['grid'][5] == response['grid'][6] == e.SUNK
    assert response['sunk'] == ['Patrol']


def test_invalid_grid_length_rejected():
    with pytest.raises(ValueError, match='grid length'):
        server.board_from({'size': 4, 'grid': [0] * 15})


@pytest.fixture
def http_server():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}'
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def post_json(url, payload):
    request = Request(url, data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=8) as response:
        return response.status, json.load(response)


def test_http_dashboard_serves_html_and_no_cache(http_server):
    with urlopen(http_server + '/', timeout=8) as response:
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert response.headers['Cache-Control'] == 'no-store'
        assert 'text/html' in response.headers['Content-Type']
        assert '/api/analyze' in html and '/api/sink' in html


def test_http_analyze_and_sink_round_trip(http_server):
    payload = {'size': 4, 'fleet': [['Patrol', 2]], 'samples': 200,
               'grid': [e.MISS] * 16}
    payload['grid'][5] = e.HIT
    payload['grid'][6] = e.UNKNOWN
    status, data = post_json(http_server + '/api/analyze', payload)
    assert status == 200 and data['rec']['best_name'] == 'B3'
    payload['grid'][6] = e.HIT
    payload.update(ship='Patrol', last=6)
    status, data = post_json(http_server + '/api/sink', payload)
    assert status == 200 and data['converted'] == [5, 6]


def test_http_returns_400_for_bad_payload_and_404_for_unknown_route(http_server):
    with pytest.raises(HTTPError) as exc:
        post_json(http_server + '/api/analyze', {'size': 4, 'grid': [0]})
    assert exc.value.code == 400
    assert 'error' in json.load(exc.value)
    with pytest.raises(HTTPError) as exc:
        post_json(http_server + '/api/unknown', {})
    assert exc.value.code == 404


def test_api_rejects_unknown_grid_state_numbers():
    grid = [0] * 16
    grid[0] = 42
    with pytest.raises(ValueError):
        server.board_from({'size': 4, 'grid': grid})


def test_api_rejects_nonpositive_ship_length():
    with pytest.raises(ValueError):
        server.board_from({'size': 4, 'fleet': [['Broken', 0]]})


def test_http_rejects_out_of_range_ship_and_unknown_grid_state(http_server):
    bad_grid = [0] * 16
    bad_grid[2] = 99
    for payload in (
        {'size': 4, 'fleet': [['Patrol', 0]]},
        {'size': 4, 'grid': bad_grid, 'fleet': [['Patrol', 2]]},
    ):
        with pytest.raises(HTTPError) as error:
            post_json(http_server + '/api/analyze', payload)
        assert error.value.code == 400
        assert 'error' in json.load(error.value)


def test_analyze_reports_sampler_and_accepts_method():
    payload = {'size': 4, 'fleet': [['A', 2], ['B', 3]], 'samples': 300,
               'grid': [e.UNKNOWN] * 16, 'method': 'weighted'}
    payload['grid'][0] = e.HIT
    response = server.analyze(payload)
    assert response['method'] == 'weighted'
    assert 0 < response['ess'] <= 300
    assert response['exact'] is False and response['total_layouts'] > 0
    default = server.analyze({**payload, 'method': 'auto'})
    assert default['method'] in ('rejection', 'exact', 'weighted')


def test_analyze_rejects_unknown_method():
    with pytest.raises(ValueError):
        server.analyze({'size': 4, 'fleet': [['A', 2]], 'grid': [0] * 16, 'method': 'magic'})


def test_analyze_proves_impossible_board():
    grid = [e.UNKNOWN] * 16
    grid[0] = grid[15] = e.HIT
    response = server.analyze({'size': 4, 'fleet': [['A', 2]], 'grid': grid, 'samples': 200})
    assert not response['consistent'] and response['proven']
    assert 'checked exhaustively' in response['warnings'][0]
