"""Telegram command listener — operator commands for the bot service.

Runs in a daemon thread alongside the main bot. Listens for slash
commands from the authorized chat (same `TELEGRAM_CHAT_ID` the
notifier sends to) and dispatches to local handlers. Replies with
confirmation but never echoes secret payloads.

Commands:
  /setpk <hex>            — write POLYMARKET_PRIVATE_KEY to .env
                            (auto-deletes the source message)
  /setfunder 0x...        — write POLYMARKET_FUNDER_ADDRESS to .env
  /restart_bot            — exit(0); systemd auto-restarts → loads .env
  /halt_live              — touch data/halt_live (instant kill switch)
  /unhalt_live            — remove data/halt_live (next restart re-arms)
  /clag_status            — quick status reply (bankroll / halt state)

Security model:
  - Hard-coded chat_id whitelist (= TELEGRAM_CHAT_ID env var).
  - Messages from any other chat are logged and IGNORED.
  - /setpk validates 64-hex format; rejects malformed input.
  - /setpk attempts to delete the original message via deleteMessage
    (works only if bot is admin in that chat; for 1:1 chats it works).
  - Replies confirm action without echoing the secret value.

Notes on operational use:
  - The bot's running offset is persisted to `data/.tg_cmd_offset` so
    a restart doesn't replay old commands.
  - `/restart_bot` relies on `Restart=always` in the systemd unit. The
    service WILL come back automatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("polymarket_bot.telegram_commands")


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ADDR = re.compile(r"^0x[0-9a-fA-F]{40}$")


class TelegramCommandHandler:
    """Background thread that long-polls Telegram for operator commands."""

    GET_UPDATES_URL = "https://api.telegram.org/bot{}/getUpdates"
    SEND_MESSAGE_URL = "https://api.telegram.org/bot{}/sendMessage"
    DELETE_MESSAGE_URL = "https://api.telegram.org/bot{}/deleteMessage"

    def __init__(
        self,
        env_path: str = ".env",
        offset_path: str = "data/.tg_cmd_offset",
        halt_file_path: str = "data/halt_live",
        long_poll_seconds: int = 30,
        status_provider=None,
    ):
        self.token = os.getenv("TELEGRAM_TOKEN", "").strip()
        self.chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "").strip())
        self.enabled = bool(self.token and self.chat_id)
        self.env_path = Path(env_path)
        self.offset_path = Path(offset_path)
        self.halt_file_path = Path(halt_file_path)
        self.long_poll_seconds = int(long_poll_seconds)
        self.status_provider = status_provider  # callable returning dict
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─── lifecycle ─────────────────────────────────────────────
    def start(self) -> None:
        if not self.enabled:
            logger.info("telegram commands disabled (no token/chat_id)")
            return
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="tg-commands"
        )
        self._thread.start()
        logger.info(f"telegram command listener started (chat_id={self.chat_id})")

    def stop(self) -> None:
        self._stop.set()

    # ─── poll loop ─────────────────────────────────────────────
    def _poll_loop(self) -> None:
        offset = self._load_offset()
        while not self._stop.is_set():
            try:
                resp = requests.get(
                    self.GET_UPDATES_URL.format(self.token),
                    params={
                        "offset": offset + 1,
                        "timeout": self.long_poll_seconds,
                        "allowed_updates": json.dumps(["message"]),
                    },
                    timeout=self.long_poll_seconds + 5,
                )
                if resp.status_code != 200:
                    time.sleep(5)
                    continue
                data = resp.json()
                if not data.get("ok"):
                    time.sleep(5)
                    continue
                for update in data.get("result", []):
                    offset = max(offset, int(update.get("update_id", 0)))
                    try:
                        self._handle_update(update)
                    except Exception as exc:
                        logger.warning(f"tg cmd: handler raised {exc}")
                self._save_offset(offset)
            except requests.RequestException as exc:
                logger.debug(f"tg cmd: network error {exc}")
                time.sleep(5)
            except Exception as exc:
                logger.warning(f"tg cmd: poll loop error {exc}")
                time.sleep(5)

    def _load_offset(self) -> int:
        try:
            return int(self.offset_path.read_text().strip())
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        try:
            self.offset_path.parent.mkdir(parents=True, exist_ok=True)
            self.offset_path.write_text(str(offset))
        except Exception:
            pass

    # ─── handlers ──────────────────────────────────────────────
    def _handle_update(self, update: dict) -> None:
        msg = update.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if chat_id != self.chat_id:
            logger.warning(
                f"tg cmd: rejected message from unauthorized chat_id={chat_id}"
            )
            return
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]
        arg = parts[1] if len(parts) > 1 else ""
        handlers = {
            "/setpk": self._cmd_setpk,
            "/setfunder": self._cmd_setfunder,
            "/setbuilderkey": self._cmd_setbuilderkey,
            "/setbuildersecret": self._cmd_setbuildersecret,
            "/setbuilderpass": self._cmd_setbuilderpass,
            "/restart_bot": self._cmd_restart,
            "/halt_live": self._cmd_halt,
            "/unhalt_live": self._cmd_unhalt,
            "/clag_status": self._cmd_status,
            "/help": self._cmd_help,
            "/start": self._cmd_help,
        }
        h = handlers.get(cmd)
        if h is None:
            return
        try:
            reply = h(arg, msg)
        except Exception as exc:
            logger.exception(f"tg cmd {cmd}: handler error")
            reply = f"❌ Error en {cmd}: {exc}"
        if reply:
            self._send(reply)

    def _cmd_help(self, _arg, _msg) -> str:
        return (
            "<b>Comandos disponibles</b>\n"
            "<code>/setpk &lt;hex&gt;</code> — guarda POLYMARKET_PRIVATE_KEY (auto-borra mensaje)\n"
            "<code>/setfunder 0x…</code> — guarda POLYMARKET_FUNDER_ADDRESS\n"
            "<code>/setbuilderkey &lt;key&gt;</code> — guarda BUILDER_KEY (auto-borra)\n"
            "<code>/setbuildersecret &lt;sec&gt;</code> — guarda BUILDER_SECRET (auto-borra)\n"
            "<code>/setbuilderpass &lt;pass&gt;</code> — guarda BUILDER_PASSPHRASE (auto-borra)\n"
            "<code>/restart_bot</code> — reinicia (systemd recarga .env)\n"
            "<code>/halt_live</code> — kill switch instantáneo (cancela órdenes LIVE)\n"
            "<code>/unhalt_live</code> — quita el kill switch (requiere /restart_bot después)\n"
            "<code>/clag_status</code> — resumen rápido"
        )

    def _cmd_setpk(self, arg: str, msg: dict) -> str:
        if not arg:
            return "Uso: <code>/setpk &lt;hex_64_chars&gt;</code>"
        clean = arg.strip()
        if clean.startswith("0x") or clean.startswith("0X"):
            clean = clean[2:]
        # Strip any whitespace (user might have line breaks pasted in)
        clean = re.sub(r"\s+", "", clean)
        if not _HEX64.match(clean):
            return "❌ Formato inválido (esperado: 64 caracteres hex)"
        ok = self._write_env_var("POLYMARKET_PRIVATE_KEY", clean)
        # Try to delete the source message so the secret isn't lingering.
        deleted = self._delete_message(
            chat_id=msg.get("chat", {}).get("id"),
            message_id=msg.get("message_id"),
        )
        suffix = "✅ borrado del chat" if deleted else "⚠️ <b>borrá el mensaje original a mano</b>"
        if ok:
            return f"✅ Private key guardada en .env (chmod 600). Mensaje: {suffix}"
        return "❌ Error escribiendo .env (revisar permisos)"

    def _cmd_setfunder(self, arg: str, _msg: dict) -> str:
        if not arg:
            return "Uso: <code>/setfunder 0x…</code>"
        clean = arg.strip().split()[0]
        if not _ADDR.match(clean):
            return "❌ Formato inválido (esperado: 0x + 40 hex chars)"
        ok = self._write_env_var("POLYMARKET_FUNDER_ADDRESS", clean)
        if ok:
            return f"✅ Funder address guardada: <code>{clean[:6]}…{clean[-4:]}</code>"
        return "❌ Error escribiendo .env"

    def _cmd_setbuilderkey(self, arg: str, msg: dict) -> str:
        if not arg:
            return "Uso: <code>/setbuilderkey &lt;key&gt;</code>"
        clean = arg.strip().split()[0]
        if len(clean) < 8:
            return "❌ Builder key parece muy corta"
        ok = self._write_env_var("POLYMARKET_BUILDER_KEY", clean)
        self._delete_message(msg.get("chat", {}).get("id"), msg.get("message_id"))
        return f"✅ Builder key guardada (<code>{clean[:6]}…</code>). Mensaje borrado."

    def _cmd_setbuildersecret(self, arg: str, msg: dict) -> str:
        if not arg:
            return "Uso: <code>/setbuildersecret &lt;secret&gt;</code>"
        clean = arg.strip().split()[0]
        if len(clean) < 8:
            return "❌ Builder secret parece muy corto"
        ok = self._write_env_var("POLYMARKET_BUILDER_SECRET", clean)
        self._delete_message(msg.get("chat", {}).get("id"), msg.get("message_id"))
        return f"✅ Builder secret guardado. Mensaje borrado."

    def _cmd_setbuilderpass(self, arg: str, msg: dict) -> str:
        if not arg:
            return "Uso: <code>/setbuilderpass &lt;passphrase&gt;</code>"
        clean = arg.strip().split()[0]
        if len(clean) < 4:
            return "❌ Builder passphrase parece muy corta"
        ok = self._write_env_var("POLYMARKET_BUILDER_PASSPHRASE", clean)
        self._delete_message(msg.get("chat", {}).get("id"), msg.get("message_id"))
        return f"✅ Builder passphrase guardada. Mensaje borrado."

    def _cmd_restart(self, _arg, _msg) -> str:
        # Reply BEFORE exiting so the user sees confirmation.
        self._send("🔄 Restart en 3s — systemd recargará .env automáticamente")
        # Use a short-delayed exit so the reply lands first.
        def _delayed_exit():
            time.sleep(3)
            logger.warning("telegram /restart_bot: exiting for systemd restart")
            os._exit(0)
        threading.Thread(target=_delayed_exit, daemon=True).start()
        return ""  # don't double-send

    def _cmd_halt(self, _arg, _msg) -> str:
        try:
            self.halt_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.halt_file_path.write_text(
                f"{int(time.time())} manual halt via telegram\n"
            )
            return f"🛑 HALT activado. Archivo: <code>{self.halt_file_path}</code>"
        except Exception as exc:
            return f"❌ No se pudo crear halt file: {exc}"

    def _cmd_unhalt(self, _arg, _msg) -> str:
        if not self.halt_file_path.exists():
            return "ℹ️ No hay halt activo (archivo no existe)"
        try:
            self.halt_file_path.unlink()
            return (
                f"✅ Halt removido. ⚠️ El executor mantiene el latch en memoria; "
                f"hacé <code>/restart_bot</code> para re-armar LIVE."
            )
        except Exception as exc:
            return f"❌ No se pudo borrar halt file: {exc}"

    def _cmd_status(self, _arg, _msg) -> str:
        if self.status_provider is None:
            return "ℹ️ Status provider no configurado"
        try:
            data = self.status_provider() or {}
        except Exception as exc:
            return f"❌ status_provider raised: {exc}"
        lines = ["<b>📊 Estado del bot</b>"]
        for k in (
            "mode_active", "live_enabled", "halt_active",
            "bankroll_usdc", "lifetime_pnl_usdc", "open_orders",
            "fills_24h", "closes_24h",
        ):
            if k in data:
                lines.append(f"• {k}: <code>{data[k]}</code>")
        return "\n".join(lines)

    # ─── helpers ───────────────────────────────────────────────
    def _write_env_var(self, key: str, value: str) -> bool:
        """Atomic-ish replace-or-append `key=value` in .env, then chmod 600."""
        try:
            self.env_path.parent.mkdir(parents=True, exist_ok=True)
            existing_lines: list[str] = []
            if self.env_path.exists():
                existing_lines = self.env_path.read_text().splitlines()
            replaced = False
            new_lines = []
            for line in existing_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    new_lines.append(line)
                    continue
                if "=" in stripped and stripped.split("=", 1)[0].strip() == key:
                    new_lines.append(f"{key}={value}")
                    replaced = True
                else:
                    new_lines.append(line)
            if not replaced:
                new_lines.append(f"{key}={value}")
            tmp = self.env_path.with_suffix(self.env_path.suffix + ".tmp")
            tmp.write_text("\n".join(new_lines) + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.env_path)
            os.chmod(self.env_path, 0o600)
            return True
        except Exception as exc:
            logger.error(f"_write_env_var failed: {exc}")
            return False

    def _delete_message(self, chat_id, message_id) -> bool:
        if not chat_id or not message_id:
            return False
        try:
            r = requests.post(
                self.DELETE_MESSAGE_URL.format(self.token),
                json={"chat_id": chat_id, "message_id": message_id},
                timeout=5,
            )
            return r.status_code == 200 and r.json().get("ok", False)
        except Exception:
            return False

    def _send(self, text: str) -> None:
        try:
            requests.post(
                self.SEND_MESSAGE_URL.format(self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
        except Exception:
            pass
