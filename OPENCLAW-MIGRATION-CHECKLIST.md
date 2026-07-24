# OpenClaw-Host-Umzug: Checkliste externer Abhängigkeiten

Diese Datei sammelt, worüber wir beim Umzug von clawdpi1 → gastonllm
(2026-07-24) gestolpert sind — als wiederverwendbare Checkliste für den
nächsten Umzug von OpenClaw/Gaston auf einen neuen Host. Bitte weiterpflegen,
wenn wieder etwas Neues auftaucht.

> MemPalace-Verweis: Wing `gastonllm-home-jochen`, Room `technical` — dort
> liegen die ausführlichen Einzel-Drawer zu jedem der unten genannten Funde
> (Fehlerbilder, genaue Fixes, Debugging-Weg).

## 1. Architektur-Falle: node_modules/venvs nie kopieren

Bei einem Wechsel der CPU-Architektur (z.B. ARM64 Pi → x86_64) müssen
neu installiert werden statt kopiert:
- `~/.npm-global` (Node/npm-Pakete inkl. `openclaw`, `@openai/codex`)
- Python-venvs (`ow-venv` etc.) — passende Python-Version prüfen
  (Debian-Repo hat oft nur die neueste Version, pyenv nachziehen falls
  ältere Pins wie `tflite-runtime` gebraucht werden)

## 2. Lose Zugangsdaten-Dateien außerhalb von `.openclaw`

**Wiederkehrender blinder Fleck.** Ein Scan auf `.openclaw` allein reicht
nicht — Skripte referenzieren häufig Dateien direkt im Home-Verzeichnis
oder unter `~/.local/share/<tool>/`. Bisher gefunden:

| Datei | Gebraucht von | Zweck |
|---|---|---|
| `~/noris-key.txt` | openclaw.json `secrets.providers.noris` | API-Key für noris-LLM-Provider |
| `~/influxdb-raw-8086-clawdpi1-read-token.txt` | `workspace-hauswacht/watchers/aktiv/*.sh` | InfluxDB-Read-Token (raw-Instanz) |
| `~/glab-token.txt` | `scripts/gastoncode_backup.sh`, `scripts/hauswacht_backup.sh` | GitLab-Push-Token (Auto-Backup-Cronjobs) |
| `~/.local/share/gcalcli/oauth` (+`cache`) | `scripts/gcalcli_list_wrapper.py` | Google-Calendar-OAuth-Token |

**Vorgehen fürs nächste Mal:** Vor dem Abschalten der alten Maschine:
```bash
grep -rlE "Path\.home\(\)|os\.path\.expanduser\(.~.\)|~/\.[a-zA-Z]" \
  ~/.openclaw/scripts ~/.openclaw/workspace* ~/.openclaw/broker \
  --include="*.py" --include="*.sh"
```
Jeden Treffer prüfen: zeigt er auf eine Datei außerhalb `.openclaw`? Existiert
sie auf dem neuen Host? Wenn nicht: kopieren, `chmod 600` wo sinnvoll.

## 3. Hartkodierte `/home/<alterUser>`-Pfade

Nicht nur in `.py`/`.sh`/`.json`-Configs, auch in:
- **SQLite-Statusdatenbanken** (`~/.openclaw/state/openclaw.sqlite` und
  `~/.openclaw/agents/*/agent/openclaw-agent.sqlite`) — insbesondere Tabellen
  `agent_databases`, `installed_plugin_index`, `config_health_entries`,
  `plugin_state_entries`, `cron_jobs`. Volltextsuche über ALLE Spalten nötig
  (keine Annahme, welche Spalte Pfade enthält).
- **`agents/<id>/sessions/sessions.json`** — das ist der LEBENDE Index
  (Session-Key → aktuelle Datei), KEINE Historie! Beim ersten Bulk-Fix hier
  fälschlich als Archiv behandelt und übersprungen — genau das hat den
  Gateway-Fehler `EACCES: permission denied, mkdir '/home/pi'` verursacht.
- **`*.trajectory-path.json`** (ca. 2000 Stück) — ebenfalls lebende
  Zeiger-Dateien, keine Historie.
- Was tatsächlich reine Historie ist und NICHT gefixt werden muss:
  `*.trajectory.jsonl`, `*.jsonl.bak`, `*.jsonl.deleted.*`,
  `*.jsonl.codex-app-server.json`, `active-memory/transcripts/**`,
  `cron/runs/*.jsonl` — das ist Konversationsinhalt/Snapshots, kein Routing.

**Faustregel:** Alles unter `agents/*/sessions/` erst genau anschauen, bevor
es als "nur Historie" pauschal ausgeschlossen wird — nur Dateien mit einer
Session-**ID im Dateinamen selbst** (z.B. `<uuid>.jsonl`) sind reine Historie;
Dateien, die eine **Zuordnung/einen Index** darstellen (`sessions.json`,
`*.trajectory-path.json`), sind funktional.

## 4. OpenClaw-Plugins, die separat nachinstalliert werden müssen

`npm install -g openclaw` bringt NICHT automatisch mit:
- `@openclaw/codex` (Agent-Harness fürs `codex`-Modell) —
  `openclaw plugins install @openclaw/codex`
- `@openclaw/acpx` (Agent Client Protocol Backend) —
  `openclaw plugins install @openclaw/acpx`

`openclaw doctor` zeigt fehlende Plugins als Config-Warning
("plugin not installed: X — install ... openclaw plugins install @openclaw/X").

## 5. GPU/CUDA-Treiber (falls neuer Host eine NVIDIA-GPU hat)

Debians eigenes `nvidia-driver`-Paket kann zu alt für aktuelle
CUDA-Container-Images sein (CUDA-Forward-Compatibility funktioniert nur auf
Datacenter-GPUs, nicht auf Consumer-Karten). Falls
`RuntimeError: CUDA failed with error forward compatibility was attempted on
non supported HW` auftaucht: NVIDIAs eigenes CUDA-Repo einrichten
(`cuda-keyring` + `apt install cuda-drivers`) statt Debians Paket. NACH einem
Treiber-Wechsel bereits laufende GPU-Container per
`podman-compose up -d --force-recreate <service>` neu erzeugen — ein reiner
Restart reicht nicht (alte Geräte-Bindung bleibt hängen).

## 6. Sonstige Nacharbeiten nach dem Umzug

- Alte, jetzt tote MCP-Server-Einträge in `openclaw.json` bereinigen (Beispiel:
  `ssdpi1-wohnzimmer-display`, Display läuft längst auf monitorpi1).
- `openclaw cron list` + `openclaw doctor` als aktive Post-Migration-Checks
  laufen lassen, nicht nur auf Nutzer-Feedback warten — Cronjob-Fehler fallen
  sonst erst auf, wenn der nächste Lauf ansteht.
- Nach jedem Fix betroffene Cronjobs per `openclaw cron run <id>` manuell
  antriggern, um den Status sofort auf `ok` zu aktualisieren, statt aufs
  nächste Zeitfenster zu warten.
