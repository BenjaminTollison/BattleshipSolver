"""Local opt-in game archive; partially observed losses are never completed.

The caller validates the Board with server.board_from before saving. No saved
record can turn an UNKNOWN square into a MISS by inference. No cloud services.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import uuid
from datetime import datetime, timezone

import engine

_LOCK = threading.RLock()
MAX_GAMES = 10000
VALID_OUTCOMES = {"won", "lost", "unfinished"}


def store_path():
    return Path(os.environ.get("BATTLESHIP_GAME_HISTORY",
                               str(Path(__file__).resolve().parent / "data" / "games.json")))


def normalize_opponent(value):
    if not isinstance(value, str):
        raise ValueError("opponent must be text")
    name = value.strip()
    if not 1 <= len(name) <= 64 or any(ord(x) < 32 for x in name):
        raise ValueError("opponent name must be 1–64 printable characters")
    return name


def _read():
    path = store_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("game history file is not a list")
    return data


def _atomic_write(games):
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(games, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save(payload, board):
    opponent = normalize_opponent(payload.get("opponent", ""))
    outcome = payload.get("outcome")
    if outcome not in VALID_OUTCOMES:
        raise ValueError("outcome must be won, lost, or unfinished")
    # Never complete or alter the submitted observation grid.
    grid = list(board.grid)
    known = sum(x != engine.UNKNOWN for x in grid)
    if known == 0:
        raise ValueError("cannot save an empty game")
    session_id = payload.get("session_id")
    if session_id is not None:
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 80 or any(ord(c) < 33 for c in session_id):
            raise ValueError("invalid game session id")
    record = {"id": uuid.uuid4().hex, "session_id": session_id, "saved_at": datetime.now(timezone.utc).isoformat(),
              "opponent": opponent, "outcome": outcome, "size": board.size,
              "fleet": [list(x) for x in board.fleet], "no_touch": bool(board.no_touch),
              "grid": grid, "sunk": list(board.sunk_ships),
              "observed": known, "unknown": len(grid) - known,
              "complete_layout": bool(outcome == "won" and len(board.sunk_ships) == len(board.fleet)
                                      and all(v != engine.HIT for v in grid))}
    with _LOCK:
        games = _read()
        existing = next((j for j, g in enumerate(games)
                         if session_id and g.get("session_id") == session_id), None)
        if existing is not None:
            previous = games[existing]
            if previous["opponent"] != opponent or previous["size"] != board.size or \
                    previous["fleet"] != record["fleet"] or previous["no_touch"] != record["no_touch"]:
                raise ValueError("saved game identity/rules changed; start a new game session")
            record["id"] = previous["id"]
            games[existing] = record  # update the snapshot, never double-count one game
        else:
            if len(games) >= MAX_GAMES:
                raise ValueError("game archive limit reached; export or back up existing games")
            games.append(record)
        _atomic_write(games)
    return {k: record[k] for k in ("id", "saved_at", "opponent", "outcome", "observed",
                                    "unknown", "complete_layout")}


def list_games(opponent=None):
    with _LOCK:
        games = _read()
    if opponent:
        opponent = normalize_opponent(opponent)
        games = [g for g in games if g["opponent"] == opponent]
    return [{k: g[k] for k in ("id", "saved_at", "opponent", "outcome", "observed",
                                 "unknown", "complete_layout")} for g in reversed(games)]


def all_games():
    with _LOCK:
        return _read()


def get_game(game_id):
    if not isinstance(game_id, str) or len(game_id) != 32 or any(c not in "0123456789abcdef" for c in game_id):
        return None
    with _LOCK:
        return next((g for g in _read() if g["id"] == game_id), None)
