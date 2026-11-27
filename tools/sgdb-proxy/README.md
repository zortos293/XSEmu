# SteamGridDB proxy

A Cloudflare Worker that lets the XSEmu launcher download game art from
[SteamGridDB](https://www.steamgriddb.com) without anyone entering an API key. The Worker holds
one SteamGridDB key as a secret (`SGDB_API_KEY`); the key is never in this repository or in the
launcher.

It is deployed at `https://xsemu-sgdb.zortosgames293.workers.dev`. The launcher's default
(`CoverArtService::kProxyBase`) points there; `XSEMU_SGDB_PROXY` overrides it, for example
for a self-hosted copy.

## What it serves

Only `GET`, and only these read-only SteamGridDB API v2 paths, under the same paths, so the
launcher's code is the same with the proxy or with a personal key:

| Path | Upstream | Cached |
| --- | --- | --- |
| `/api/v2/search/autocomplete/<term>` | the same | 7 days |
| `/api/v2/{grids,heroes,logos,icons}/game/<id>` | the same | 3 days |

- The search term is normalised (Unicode NFKC, lower case, single spaces) and limited to 128
  characters; art queries keep only `styles`, `dimensions`, `mimes`, `types`, `nsfw`,
  `humor`, `epilepsy` and `page` with plain values.
- Responses are cached in Workers KV (`CACHE`). A SteamGridDB 404 is cached for a day; errors
  are not cached.
- Each client IP gets 120 requests a minute (`RATE_LIMITER`), counted per Cloudflare location.
  Over that it answers 429 with `Retry-After: 60`, and the launcher pauses its queue.
- Upstream auth errors come back as 502 and upstream rate limits as 503, never with SteamGridDB's
  body. Without the secret every proxied path answers 503.
- Images are not proxied: the launcher downloads them from SteamGridDB's CDN.

## Develop and deploy

```sh
npm ci
npm run typecheck
npm test
npx wrangler dev                  # local, with a key in .dev.vars: SGDB_API_KEY=...
npx wrangler deploy
npx wrangler secret put SGDB_API_KEY
```

`wrangler.jsonc` names the KV namespace and the rate-limit namespace of the deployment above;
change both for your own account. Get a SteamGridDB key at steamgriddb.com, Preferences, API.

Art on SteamGridDB is made by its community; the launcher credits SteamGridDB on its About page and
names each image's artist on the game's details page.
