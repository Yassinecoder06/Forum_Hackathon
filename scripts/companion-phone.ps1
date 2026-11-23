# Open the AR companion on a phone (camera, microphone and WebXR need https). Nothing is exposed to the internet:
# the phone reaches this PC through the private Tailscale network only.
#
#   .\scripts\companion-phone.ps1                  # https on this PC's Tailscale address, local certificate
#   .\scripts\companion-phone.ps1 -Tailscale       # https by `tailscale serve` (trusted *.ts.net certificate)
#   .\scripts\companion-phone.ps1 -Stop            # stop the phone front end (and the tailscale serve entry)
#
# The API must already run (start_studio.ps1, or any port given by -ApiPort). The phone opens the printed address,
# then the Compagnon page (#compagnon).
param([int]$ApiPort = 8012, [int]$PhonePort = 5443, [switch]$Tailscale, [switch]$Stop)

$repo = Split-Path $PSScriptRoot -Parent
$certDir = Join-Path $repo ".runtime\phone-cert"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'vite' -and $_.CommandLine -match "--port $PhonePort" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
if ($Stop) {
    if ($Tailscale) { tailscale serve --https=443 off 2>$null }
    "Phone front end stopped."; return
}

$tsIp = (tailscale ip -4 2>$null | Select-Object -First 1)
$tsName = ((tailscale status --json | ConvertFrom-Json).Self.DNSName).TrimEnd('.')
if (-not $tsIp) { "Tailscale is not connected on this PC."; return }
$env:SIGNACULTURE_API_TARGET = "http://127.0.0.1:$ApiPort"

if ($Tailscale) {
    # Front end on localhost only; Tailscale terminates https with a real certificate for $tsName.
    Remove-Item Env:SIGNACULTURE_DEV_CERT, Env:SIGNACULTURE_DEV_KEY -ErrorAction SilentlyContinue
    Start-Process -FilePath "npm.cmd" -WorkingDirectory $repo -WindowStyle Hidden `
        -ArgumentList "--prefix frontend run dev -- --host 127.0.0.1 --port $PhonePort --strictPort"
    Start-Sleep -Seconds 4
    tailscale serve --bg --https=443 "http://127.0.0.1:$PhonePort"
    "On the phone (Tailscale on): https://$tsName/#compagnon"
    "Stop: .\scripts\companion-phone.ps1 -Tailscale -Stop"
    return
}

# Local certificate for the Tailscale address and name (made once, kept out of Git in .runtime/).
if (-not (Test-Path "$certDir\cert.pem")) {
    New-Item -ItemType Directory -Force $certDir | Out-Null
    $openssl = (Get-Command openssl -ErrorAction SilentlyContinue).Source
    if (-not $openssl) { $openssl = "C:\Program Files\Git\mingw64\bin\openssl.exe" }
    & $openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout "$certDir\key.pem" -out "$certDir\cert.pem" `
        -subj "/CN=SignaCulture companion" -addext "subjectAltName=IP:$tsIp,DNS:$tsName" 2>$null
}
$env:SIGNACULTURE_DEV_CERT = "$certDir\cert.pem"
$env:SIGNACULTURE_DEV_KEY = "$certDir\key.pem"
# Bound to the Tailscale address only: not reachable from the local Wi-Fi or the internet.
Start-Process -FilePath "npm.cmd" -WorkingDirectory $repo -WindowStyle Hidden `
    -ArgumentList "--prefix frontend run dev -- --host $tsIp --port $PhonePort --strictPort"
"On the phone (Tailscale on): https://${tsIp}:$PhonePort/#compagnon"
"The certificate is local: accept it once (Android: Advanced > Continue; iPhone: Show details > visit this website)."
"Stop: .\scripts\companion-phone.ps1 -Stop"
