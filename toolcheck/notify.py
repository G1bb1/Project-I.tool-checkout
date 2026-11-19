"""Optional: send each check result to the watch as one UDP JSON packet.
Disabled by default (settings.json -> notify.enabled). The watch firmware side
is not part of this project yet; this is the hook for the REQ-TR-04 / link test."""
import json
import socket


def send_result(cfg, status, tools):
    if not cfg.get("enabled"):
        return
    msg = json.dumps({"status": status, "tools": tools}).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(msg, (cfg["watch_ip"], int(cfg["watch_port"])))
