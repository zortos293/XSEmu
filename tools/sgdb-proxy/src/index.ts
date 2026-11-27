const UPSTREAM = "https://www.steamgriddb.com/api/v2";
const ART_PARAMS = ["styles", "dimensions", "mimes", "types", "nsfw", "humor", "epilepsy", "page"];
const SEARCH_TTL_S = 7 * 24 * 3600;
const ART_TTL_S = 3 * 24 * 3600;
const MAX_TERM = 128;

export interface Env {
  SGDB_API_KEY?: string;
  CACHE: KVNamespace;
  RATE_LIMITER: RateLimit;
}

type Route = { cacheKey: string; upstream: string; ttl: number };

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", ...headers },
  });
}

function fail(status: number, message: string, headers: Record<string, string> = {}): Response {
  return json(status, { success: false, errors: [message] }, headers);
}

export function normalizeTerm(term: string): string {
  return term.normalize("NFKC").toLowerCase().replace(/\s+/g, " ").trim();
}

// The SteamGridDB API v2 paths the proxy serves, mapped to their upstream URL and cache key; null for anything else.
export function route(url: URL): Route | null {
  const path = url.pathname.replace(/\/+$/, "");
  const search = /^\/api\/v2\/search\/autocomplete\/([^/]+)$/.exec(path);
  if (search) {
    let term: string;
    try {
      term = normalizeTerm(decodeURIComponent(search[1]));
    } catch {
      return null;
    }
    if (!term || term.length > MAX_TERM) return null;
    const encoded = encodeURIComponent(term);
    return { cacheKey: `search/${encoded}`, upstream: `${UPSTREAM}/search/autocomplete/${encoded}`, ttl: SEARCH_TTL_S };
  }
  const art = /^\/api\/v2\/(grids|heroes|logos|icons)\/game\/(\d{1,10})$/.exec(path);
  if (!art) return null;
  const query = new URLSearchParams();
  for (const name of ART_PARAMS) {
    const value = url.searchParams.get(name);
    if (value !== null && /^[\w,.-]{1,64}$/.test(value)) query.set(name, value);
  }
  query.sort();
  const qs = query.toString();
  const tail = `${art[1]}/game/${Number(art[2])}${qs ? `?${qs}` : ""}`;
  return { cacheKey: tail, upstream: `${UPSTREAM}/${tail}`, ttl: ART_TTL_S };
}

async function upstreamGet(route: Route, env: Env): Promise<Response> {
  const res = await fetch(route.upstream, {
    headers: { authorization: `Bearer ${env.SGDB_API_KEY}`, accept: "application/json", "user-agent": "xsemu-sgdb-proxy" },
  });
  const body = await res.text();
  if (res.status === 401 || res.status === 403) return fail(502, "SteamGridDB refused the proxy's key.");
  if (res.status === 429) return fail(503, "SteamGridDB is rate limiting the proxy; try again later.", { "retry-after": "60" });
  if (res.status !== 200 && res.status !== 404) return fail(502, `SteamGridDB answered ${res.status}.`);
  return new Response(body, { status: res.status, headers: { "content-type": "application/json; charset=utf-8" } });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method !== "GET") return fail(405, "Only GET is allowed.", { allow: "GET" });
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "") {
      return json(200, {
        service: "XSEmu SteamGridDB proxy",
        source: "https://github.com/zortos293/xsemu/tree/main/tools/sgdb-proxy",
        attribution: "Artwork and data from SteamGridDB (https://www.steamgriddb.com), made by its community.",
      });
    }
    const r = route(url);
    if (!r) return fail(404, "Not a proxied SteamGridDB endpoint.");

    const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
    const { success } = await env.RATE_LIMITER.limit({ key: ip });
    if (!success) return fail(429, "Too many requests; slow down.", { "retry-after": "60" });

    const cacheHeaders = { "cache-control": `public, max-age=${r.ttl}` };
    const cached = await env.CACHE.get(r.cacheKey);
    if (cached !== null) {
      const { status, body } = JSON.parse(cached) as { status: number; body: string };
      return new Response(body, { status, headers: { "content-type": "application/json; charset=utf-8", "x-cache": "HIT", ...cacheHeaders } });
    }

    if (!env.SGDB_API_KEY) return fail(503, "The proxy has no SteamGridDB key configured.");
    const res = await upstreamGet(r, env);
    if (res.status !== 200 && res.status !== 404) return res;
    const body = await res.text();
    ctx.waitUntil(
      env.CACHE.put(r.cacheKey, JSON.stringify({ status: res.status, body }), { expirationTtl: res.status === 200 ? r.ttl : 24 * 3600 }).catch(() => {}),
    );
    return new Response(body, { status: res.status, headers: { "content-type": "application/json; charset=utf-8", "x-cache": "MISS", ...cacheHeaders } });
  },
} satisfies ExportedHandler<Env>;
