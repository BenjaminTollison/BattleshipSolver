"""Static dashboard smoke checks. Browser behavior requires a browser-based suite."""
from pathlib import Path

import pytest

HTML = (Path(__file__).resolve().parents[1] / 'dashboard.html').read_text(encoding='utf-8')


@pytest.mark.parametrize('element_id', [
    'board', 'parity', 'notouch', 'samples', 'rec-miss', 'rec-hit', 'rerun',
    'cmd', 'cmd-go', 'newgame', 'fleet', 'fleet-apply', 'undo', 'showpct',
])
def test_dashboard_exposes_required_controls(element_id):
    assert f'id="{element_id}"' in HTML


def test_dashboard_persists_state_and_calls_real_api():
    assert 'localStorage.setItem' in HTML
    assert 'localStorage.getItem' in HTML
    assert 'fetch(path' in HTML
    assert '/api/analyze' in HTML
    assert '/api/sink' in HTML
    assert 'function undo()' in HTML
    assert 'function runCommands()' in HTML
