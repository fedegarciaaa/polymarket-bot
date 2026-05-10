"""Tests for the Telegram command listener — handler dispatch + env writes.

These tests don't make network calls; they exercise the dispatch logic
and the env-file writer in isolation.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strategies.telegram_commands import TelegramCommandHandler


def _make_handler(env_path: str, **kw) -> TelegramCommandHandler:
    # Stub env vars so .enabled is True without needing a real bot
    with patch.dict(os.environ, {"TELEGRAM_TOKEN": "fake", "TELEGRAM_CHAT_ID": "12345"}):
        h = TelegramCommandHandler(env_path=env_path, **kw)
    h.token = "fake"
    h.chat_id = "12345"
    h.enabled = True
    return h


def _msg(text: str, chat_id: str = "12345", message_id: int = 1) -> dict:
    return {
        "update_id": 1,
        "message": {
            "message_id": message_id,
            "chat": {"id": int(chat_id)},
            "text": text,
        },
    }


class EnvWriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.env = Path(self._tmp.name) / ".env"

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_to_empty_creates_file(self):
        h = _make_handler(env_path=str(self.env))
        ok = h._write_env_var("FOO", "bar")
        self.assertTrue(ok)
        self.assertEqual(self.env.read_text().strip(), "FOO=bar")

    def test_replace_existing_key_preserves_others(self):
        self.env.write_text("EXISTING=keepme\nFOO=oldvalue\nOTHER=keepme2\n")
        h = _make_handler(env_path=str(self.env))
        ok = h._write_env_var("FOO", "newvalue")
        self.assertTrue(ok)
        content = self.env.read_text()
        self.assertIn("FOO=newvalue", content)
        self.assertIn("EXISTING=keepme", content)
        self.assertIn("OTHER=keepme2", content)
        self.assertNotIn("oldvalue", content)

    def test_append_when_key_not_present(self):
        self.env.write_text("EXISTING=keepme\n")
        h = _make_handler(env_path=str(self.env))
        h._write_env_var("NEW_KEY", "added")
        text = self.env.read_text()
        self.assertIn("EXISTING=keepme", text)
        self.assertIn("NEW_KEY=added", text)

    def test_chmod_600_after_write(self):
        h = _make_handler(env_path=str(self.env))
        h._write_env_var("FOO", "bar")
        # On Windows os.chmod has limited effect; only assert on POSIX.
        if os.name == "posix":
            mode = self.env.stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)


class CommandDispatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.env = Path(self._tmp.name) / ".env"
        self.halt = Path(self._tmp.name) / "halt_live"
        self.handler = _make_handler(
            env_path=str(self.env), halt_file_path=str(self.halt),
        )
        self._sent = []
        self.handler._send = lambda text: self._sent.append(text)
        self.handler._delete_message = lambda chat_id, message_id: True

    def tearDown(self):
        self._tmp.cleanup()

    def test_unauthorized_chat_is_ignored(self):
        self.handler._handle_update(_msg("/setpk abcd", chat_id="99999"))
        self.assertEqual(len(self._sent), 0)
        self.assertFalse(self.env.exists())

    def test_setpk_with_valid_64_hex_writes_to_env(self):
        pk = "ab" * 32
        self.handler._handle_update(_msg(f"/setpk {pk}"))
        self.assertEqual(len(self._sent), 1)
        self.assertIn("✅", self._sent[0])
        self.assertIn(f"POLYMARKET_PRIVATE_KEY={pk}", self.env.read_text())

    def test_setpk_strips_0x_prefix(self):
        pk = "cd" * 32
        self.handler._handle_update(_msg(f"/setpk 0x{pk}"))
        self.assertIn(f"POLYMARKET_PRIVATE_KEY={pk}", self.env.read_text())
        self.assertNotIn("0x" + pk, self.env.read_text())

    def test_setpk_rejects_short_input(self):
        self.handler._handle_update(_msg("/setpk abcd"))
        self.assertEqual(len(self._sent), 1)
        self.assertIn("❌", self._sent[0])
        self.assertFalse(self.env.exists())

    def test_setpk_rejects_non_hex(self):
        self.handler._handle_update(_msg("/setpk " + "z" * 64))
        self.assertIn("❌", self._sent[0])
        self.assertFalse(self.env.exists())

    def test_setfunder_valid_address(self):
        addr = "0x" + "ab" * 20
        self.handler._handle_update(_msg(f"/setfunder {addr}"))
        self.assertIn("✅", self._sent[0])
        self.assertIn(f"POLYMARKET_FUNDER_ADDRESS={addr}", self.env.read_text())

    def test_setfunder_rejects_short_address(self):
        self.handler._handle_update(_msg("/setfunder 0xshort"))
        self.assertIn("❌", self._sent[0])
        self.assertFalse(self.env.exists())

    def test_setfunder_rejects_missing_0x(self):
        self.handler._handle_update(_msg("/setfunder " + "ab" * 20))
        self.assertIn("❌", self._sent[0])

    def test_halt_live_creates_file(self):
        self.handler._handle_update(_msg("/halt_live"))
        self.assertTrue(self.halt.exists())
        self.assertIn("HALT", self._sent[0])

    def test_unhalt_live_removes_file_when_present(self):
        self.halt.parent.mkdir(parents=True, exist_ok=True)
        self.halt.write_text("manual")
        self.handler._handle_update(_msg("/unhalt_live"))
        self.assertFalse(self.halt.exists())
        self.assertIn("✅", self._sent[0])

    def test_unhalt_live_noop_when_no_halt(self):
        self.handler._handle_update(_msg("/unhalt_live"))
        self.assertFalse(self.halt.exists())
        self.assertIn("ℹ️", self._sent[0])

    def test_status_uses_provider(self):
        self.handler.status_provider = lambda: {
            "live_enabled": True, "halt_active": False,
            "bankroll_usdc": 100.0,
        }
        self.handler._handle_update(_msg("/clag_status"))
        out = self._sent[0]
        self.assertIn("100.0", out)
        self.assertIn("live_enabled", out)

    def test_unknown_command_is_silent(self):
        self.handler._handle_update(_msg("/something_random"))
        self.assertEqual(len(self._sent), 0)

    def test_help_listed_commands(self):
        self.handler._handle_update(_msg("/help"))
        self.assertEqual(len(self._sent), 1)
        for cmd in ("/setpk", "/setfunder", "/restart_bot", "/halt_live"):
            self.assertIn(cmd, self._sent[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
