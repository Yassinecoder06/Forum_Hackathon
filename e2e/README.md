# Browser checks

Start the real API and UI with `make dev`, then in a second terminal:

```sh
cd e2e
npm ci
npx playwright install chromium
npm test
```

For an existing local Chrome installation, set `CHROME_PATH` to its executable. `APP_URL` overrides the UI URL and `E2E_OUTPUT` controls screenshot output. The check generates a WAV in memory, uses actual API responses, verifies a two-clip timeline, seeking, unknown-only clearing, audio analysis, mobile width, and visible errors when API requests are blocked. It never substitutes a fake API or records a human review.

The browser also exports a real watermarked WebM and checks its download size, imports an explicitly synthetic JSON fixture with provenance, verifies persistent lookup, and removes its test registration in a finally block. Optional local emotion2vec is actually invoked when available. Screenshots show the current UI, not mocked responses.

After expanding the real corpus, `npm run test:concepts` verifies manual concept selection (Bonjour and demain), reordering, real preview composition, pending-review status, review-sheet download and mobile layout. It requires the expanded local corpus; the default smoke test also works with the bundled Bonjour-only corpus.
