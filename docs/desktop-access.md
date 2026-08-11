# Accessing the desktop over SSH

## Quick access

```bash
ssh desktop
```

This resolves via the `desktop` alias in `~/.ssh/config` (not tracked in this
repo — it's a local machine file):

```
Host desktop
  HostName 100.80.109.96
  Port 22
  User cosenzac
  IdentityFile ~/.ssh/id_ed25519
```

Key-based auth is already set up — no password needed. Works from any
network, not just the home LAN (see below).

## Known facts (so future debugging doesn't start from zero)

| | |
|---|---|
| Desktop hostname | `DESKTOP-FPI4CHA` |
| Desktop user | `cosenzac` |
| This machine's own hostname | `ZacPC` (Tailscale name: `zacpc`) |
| Desktop's Tailscale IP | `100.80.109.96` |
| This machine's Tailscale IP | `100.93.216.96` |

Both machines are WSL2 instances (each on its own Windows PC), each running
Tailscale directly inside the WSL2 Linux environment — not on the Windows
host. `sudo tailscale up` was run inside WSL2 on both sides, both logged into
the same Tailscale account, which puts them on the same private mesh network
regardless of physical location or WiFi network.

## Why Tailscale instead of the LAN IP

Originally this used the desktop's local LAN IP (`192.168.86.x`, port `2222`
via a Windows `netsh portproxy` rule forwarding to WSL2's internal port 22).
That only worked when both machines were on the same home WiFi, and broke
whenever the desktop's DHCP-assigned IP changed (e.g. after a restart).

Tailscale gives each machine a **stable** IP (`100.x.x.x`) that works
identically regardless of network — no more re-discovering the desktop's
current LAN IP, no port-forwarding, and SSH isn't exposed to the public
internet (only reachable through the authenticated Tailscale mesh).

**Gotcha if you ever try to use it**: connect on port **22**, not 2222. Port
2222 only exists because of the old Windows-host `portproxy` rule, which
operates outside WSL2's network namespace — Tailscale traffic goes directly
into WSL2 and never touches that rule, so port 2222 gets "connection refused"
over Tailscale even though it works fine over the LAN.

## If Tailscale itself needs reinstalling

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

The second command prints a login URL — open it and log into the same
Tailscale account used on the other machine. Verify with:

```bash
tailscale status   # should list both zacpc and desktop-fpi4cha
```

## If key auth itself ever breaks (not just the IP)

The desktop's `~/.ssh/authorized_keys` (for `cosenzac`) needs to contain the
public key from this machine's `~/.ssh/id_ed25519.pub`. If it's ever missing
or a login prompts for a password unexpectedly:

```bash
# on the desktop, as cosenzac:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat >> ~/.ssh/authorized_keys   # paste this machine's id_ed25519.pub contents, then Ctrl-D
chmod 600 ~/.ssh/authorized_keys
```

Permissions matter — sshd silently refuses a key if `~/.ssh` isn't `700`,
`authorized_keys` isn't `600`, or the home directory is group/world-writable.
If it's still rejected after that, check the exact reason directly from the
desktop's own logs:

```bash
sudo journalctl -u ssh -n 30 --no-pager | grep -i -E "cosenzac|publickey|refused"
```
