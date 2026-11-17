# Homelab research preview

The saved Herdr machine `homelab` (`home-pc`, user `ideaserver`) hosts the app at
`/home/ideaserver/Projects/Forum_Hackathon`. Its origin is bound to
`127.0.0.1:8320`; the production React build and API share one origin, with API
routes under `/api`. Uvicorn uses the authenticated factory in `server.py`.

The Git repository was cloned from `git@github.com:Yassinecoder06/Forum_Hackathon.git`.
The deployment then switched to `deploy/homelab-snapshot`, fetched from a Git bundle
of the local tracked app at `a3b1480`, retaining `origin/main` separately. The
deployment wrapper is an additional working-tree change. Training datasets and
checkpoints are not deployed or started.

## Services and persistence

```bash
systemctl --user status signaculture-forum.service signaculture-forum-tunnel.service
journalctl --user -u signaculture-forum.service -n 50 --no-pager
journalctl --user -u signaculture-forum-tunnel.service -n 50 --no-pager
```

Units are in `~/.config/systemd/user/` and are enabled with restart-on-failure.
User lingering was enabled so they can start without an interactive login.
No other application's service or existing Cloudflare ingress was changed.

The dedicated tunnel is `signaculture-forum-homelab`, UUID
`1025ed22-6da8-41c1-b045-129e24a8c007`. Its own configuration is
`~/.config/signaculture-forum/tunnel.yml`; credentials remain in the remote
user's `.cloudflared` directory and must never enter Git.

Always pass the dedicated `--config` when administering this tunnel. The existing
default configuration belongs to a different homelab tunnel and can override
command assumptions.

## Authentication and assets

The entire preview is HTTP Basic authenticated, apart from the small `/healthz`
probe. This also protects the app's upload/delete endpoints and research assets.
Production startup fails without credentials. Credentials are stored mode 0600
in `~/.config/signaculture-forum/app.env`. A private local copy is available at
`.runtime/homelab-preview.env`; do not paste it in logs, commits or reports.

Only the existing selected motion library, source capture assets and local
review avatar are copied. Imported motion paths in the remote `library.json`
are rewritten for the remote storage root. Local source data remains unchanged.
Review/rights statuses are preserved; deployment does not grant new rights or
linguistic approval. The Noa model remains a restricted research preview asset.

The base application runs with CPU dependencies, FFmpeg and dictionary fallback.
Optional MediaPipe capture, emotion2vec and training runtimes are not installed
as part of this deployment. The capability endpoint reports their absence.
Raw CV re-extraction archives and ML model weights are not part of the preview
copy. Original video review and existing motion playback are separate from new
computer-vision extraction.

## DNS handoff

The user confirmed **`forum.yassinejemmalie.me`**. The exact parent domain
returns NXDOMAIN and a successful Cloudflare zone-list API request finds no
`yassinejemmalie.me` zone in the accessible account. The other spelling,
`yassinejemmali.me`, is active but is not the requested domain. No record was
created in that different zone. The dedicated tunnel ingress is scoped to the
confirmed hostname with a final 404 fallback. Public DNS/HTTPS remains blocked
until the exact domain is registered/delegated and accessible in Cloudflare.

The tunnel ingress is already scoped to the confirmed hostname and validated.
Once the exact zone is active, create this record **inside `yassinejemmalie.me`**:

```json
{"type":"CNAME","name":"forum","content":"1025ed22-6da8-41c1-b045-129e24a8c007.cfargotunnel.com","proxied":true}
```

The existing origin certificate is associated with the other zone. Do not run
`cloudflared tunnel route dns` with that old zone-bound certificate: it can append
an unexpected zone suffix. Use the exact zone's DNS dashboard or an API request
with the explicit, verified zone ID. After DNS delegation and record creation,
verify public HTTPS for the sign-in challenge, authenticated API, frontend
assets and avatar. Never overwrite an unrelated DNS record or restart the
shared tunnel. The pending record is saved remotely in `.runtime/dns-pending.json`.

## Monitoring

Remote Herdr tab `w2:t4` (`Forum-deploy`) contains deployment pane `w2:p6` and
monitor pane `w2:p7` (`forum-monitor`). The monitor was started with the requested
command:

```bash
agy1 --model gemini-3.8-flash-high --effort high
```

It is assigned read-only monitoring, with observations saved remotely to
`.runtime/deployment-monitor.md`. It is not authorized to edit credentials,
DNS, application code or service configuration.

## Verification

`deployment/test_server.py` verifies frontend/API authentication, mutation-route
protection, malformed credentials, authenticated API health, SPA deep links and
real 404s for missing assets/API routes. The production frontend build passed on
homelab. Live endpoint and browser checks are stored under local `.runtime/`.
