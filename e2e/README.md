# Browser checks

Start the real API and UI with `make dev`, then in a second terminal:

```sh
cd e2e
npm ci
npx playwright install chromium
npm test
```

For an existing local Chrome installation, set `CHROME_PATH` to its executable. `APP_URL` overrides the UI URL and `E2E_OUTPUT` controls screenshot output. The check generates a WAV in memory, uses actual API responses, verifies a two-clip timeline, seeking, unknown-only clearing, audio analysis, mobile width, and visible errors when API requests are blocked. It never substitutes a fake API or records a human review.
