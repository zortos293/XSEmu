import { afterEach, describe, expect, it, vi } from "vitest";
import worker, { normalizeTerm, route, type Env } from "../src/index";

function makeEnv(opts: { key?: string; allow?: boolean } = {}) {
  const store = new Map<string, string>();
  const limited: string[] = [];
  const env = {
    SGDB_API_KEY: "key" in opts ? opts.key : "test-key",
    CACHE: {
      get: async (k: string) => store.get(k) ?? null,
      put: async (k: string, v: string) => void store.set(k, v),
    },
    RATE_LIMITER: {
      limit: async ({ key }: { key: string }) => {
        limited.push(key);
        return { success: opts.allow ?? true };
      },
    },
  } as unknown as Env;
  return { env, store, limited };
}

function makeCtx() {
  const pending: Promise<unknown>[] = [];
  return { ctx: { waitUntil: (p: Promise<unknown>) => pending.push(p), passThroughOnException() {}, props: {} } as unknown as ExecutionContext, pending };
}

async function call(path: string, env: Env, init: RequestInit = {}) {
  const { ctx, pending } = makeCtx();
  const res = await worker.fetch(new Request(`https://proxy.test${path}`, { headers: { "cf-connecting-ip": "203.0.113.7" }, ...init }), env, ctx);
  await Promise.all(pending);
  return res;
}

afterEach(() => vi.unstubAllGlobals());

describe("route", () => {
  it("maps search and art paths to SteamGridDB", () => {
    expect(route(new URL("https://p/api/v2/search/autocomplete/Signal%20%20Garden"))).toMatchObject({
      cacheKey: "search/signal%20garden",
      upstream: "https://www.steamgriddb.com/api/v2/search/autocomplete/signal%20garden",
    });
    expect(route(new URL("https://p/api/v2/grids/game/38?types=static&dimensions=600x900&evil=1"))).toMatchObject({
      cacheKey: "grids/game/38?dimensions=600x900&types=static",
      upstream: "https://www.steamgriddb.com/api/v2/grids/game/38?dimensions=600x900&types=static",
    });
    for (const kind of ["heroes", "logos", "icons"]) expect(route(new URL(`https://p/api/v2/${kind}/game/7`))?.cacheKey).toBe(`${kind}/game/7`);
  });

  it("refuses everything else", () => {
    for (const path of [
      "/api/v2/games/id/1",
      "/api/v2/grids/game/abc",
      "/api/v2/grids/steam/1",
      "/api/v2/search/autocomplete/",
      "/api/v2/search/autocomplete/%E0%A4%A",
      `/api/v2/search/autocomplete/${"a".repeat(200)}`,
      "/api/v2/grids/game/1/../../games",
    ])
      expect(route(new URL(`https://p${path}`)), path).toBeNull();
  });

  it("drops query values that are not plain tokens", () => {
    expect(route(new URL("https://p/api/v2/heroes/game/1?styles=a%26b"))?.cacheKey).toBe("heroes/game/1");
  });

  it("normalizes terms", () => {
    expect(normalizeTerm("  HALO\u00A0Infinite ")).toBe("halo infinite");
  });
});

describe("fetch", () => {
  it("allows only GET", async () => {
    const { env } = makeEnv();
    const res = await call("/api/v2/grids/game/1", env, { method: "POST" });
    expect(res.status).toBe(405);
    expect(res.headers.get("allow")).toBe("GET");
  });

  it("proxies with the secret key, then serves from the cache", async () => {
    const { env, store, limited } = makeEnv();
    const upstream = vi.fn(async (_url: string, _init: RequestInit) => new Response('{"success":true,"data":[{"id":38,"name":"Signal Garden"}]}'));
    vi.stubGlobal("fetch", upstream);
    const first = await call("/api/v2/search/autocomplete/Signal%20Garden", env);
    expect(first.status).toBe(200);
    expect(first.headers.get("x-cache")).toBe("MISS");
    expect(await first.json()).toMatchObject({ success: true, data: [{ id: 38 }] });
    expect(upstream).toHaveBeenCalledTimes(1);
    expect(upstream.mock.calls[0][0]).toBe("https://www.steamgriddb.com/api/v2/search/autocomplete/signal%20garden");
    expect((upstream.mock.calls[0][1].headers as Record<string, string>).authorization).toBe("Bearer test-key");
    expect(store.has("search/signal%20garden")).toBe(true);

    const second = await call("/api/v2/search/autocomplete/signal%20garden", env);
    expect(second.headers.get("x-cache")).toBe("HIT");
    expect(await second.text()).toContain("Signal Garden");
    expect(upstream).toHaveBeenCalledTimes(1);
    expect(limited).toEqual(["203.0.113.7", "203.0.113.7"]);
  });

  it("never returns the key or upstream auth errors verbatim", async () => {
    const { env, store } = makeEnv();
    vi.stubGlobal("fetch", async () => new Response('{"success":false,"errors":["bad key test-key"]}', { status: 401 }));
    const res = await call("/api/v2/grids/game/1", env);
    expect(res.status).toBe(502);
    expect(await res.text()).not.toContain("test-key");
    expect(store.size).toBe(0);
  });

  it("caches a 404 for unknown games", async () => {
    const { env, store } = makeEnv();
    vi.stubGlobal("fetch", async () => new Response('{"success":false,"errors":["Game not found"]}', { status: 404 }));
    expect((await call("/api/v2/logos/game/999999", env)).status).toBe(404);
    expect(store.has("logos/game/999999")).toBe(true);
  });

  it("rate limits per IP", async () => {
    const { env } = makeEnv({ allow: false });
    const res = await call("/api/v2/icons/game/1", env);
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("60");
  });

  it("says so when no key is configured", async () => {
    const { env } = makeEnv({ key: undefined });
    expect((await call("/api/v2/icons/game/1", env)).status).toBe(503);
  });

  it("does not rate limit or proxy unknown paths", async () => {
    const { env, limited } = makeEnv();
    expect((await call("/api/v2/games/id/1", env)).status).toBe(404);
    expect(limited).toEqual([]);
  });
});
