// RAD page cache: Spark streams .rad models through single Range requests and
// the server answers 206, which Cache Storage refuses to store — so each
// range's bytes are kept under a synthetic URL key and the 206 response is
// reconstructed on hits. Artifact URLs are content-fingerprinted, so cached
// pages never go stale; a byte-capped FIFO index bounds total storage.

const CACHE_PREFIX = "rad-pages-";
const CACHE_NAME = "rad-pages-v1";
const MAX_CACHE_BYTES = 3 * 1024 ** 3;
const INDEX_URL = new URL("/__rad_index__", self.location.origin).href;

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names
      .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
      .map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

function cacheKeyFor(request, rangeHeader) {
  return request.url + "?__swpage=" + encodeURIComponent(rangeHeader || "full");
}

function parseRangeHeader(header, totalSize) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match || (!match[1] && !match[2])) return null;
  let start;
  let end;
  if (match[1]) {
    start = parseInt(match[1], 10);
    end = match[2] ? parseInt(match[2], 10) : totalSize - 1;
  } else {
    start = Math.max(0, totalSize - parseInt(match[2], 10));
    end = totalSize - 1;
  }
  end = Math.min(end, totalSize - 1);
  if (start >= totalSize || end < start) return null;
  return { start, end };
}

function rangeResponse(body, contentType, contentRange) {
  return new Response(body, {
    status: 206,
    headers: {
      "Content-Type": contentType,
      "Content-Range": contentRange,
      "Content-Length": String(body.byteLength),
      "Accept-Ranges": "bytes",
    },
  });
}

function fullResponse(body, contentType) {
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": contentType,
      "Content-Length": String(body.byteLength),
      "Accept-Ranges": "bytes",
    },
  });
}

// The stored body is exactly the bytes of one range (or of the whole file for
// "full" entries); headers record what those bytes represent.
async function reconstructResponse(cached, rangeHeader) {
  const contentType = cached.headers.get("Content-Type") || "application/octet-stream";
  const storedRange = cached.headers.get("X-SW-Content-Range") || "full";
  const body = await cached.arrayBuffer();
  if (storedRange !== "full") return rangeResponse(body, contentType, storedRange);
  if (!rangeHeader) return fullResponse(body, contentType);
  const totalSize = Number(cached.headers.get("X-SW-Total-Size")) || body.byteLength;
  const range = parseRangeHeader(rangeHeader, totalSize);
  if (!range) {
    return new Response(null, {
      status: 416,
      headers: { "Content-Range": `bytes */${totalSize}` },
    });
  }
  return rangeResponse(
    body.slice(range.start, range.end + 1), contentType,
    `bytes ${range.start}-${range.end}/${totalSize}`);
}

// --- byte-capped FIFO index -------------------------------------------------
// The index lives in the cache itself under a reserved key so storage and
// accounting share one lifecycle. It is best-effort: a corrupt index is
// logged and reset, never allowed to break a fetch.

let indexPromise = null;
let indexUpdateQueue = Promise.resolve();

async function loadIndex(cache) {
  if (!indexPromise) {
    indexPromise = (async () => {
      const fresh = { bytes: 0, entries: {} };
      const response = await cache.match(INDEX_URL);
      if (!response) return fresh;
      try {
        const index = await response.json();
        if (!index || typeof index.bytes !== "number" || !index.entries) {
          throw new Error("malformed rad cache index");
        }
        return index;
      } catch (error) {
        console.error("rad cache index unreadable; starting over", error);
        return fresh;
      }
    })();
  }
  return indexPromise;
}

async function updateIndex(cache, key, storedBytes) {
  const index = await loadIndex(cache);
  if (index.entries[key] === undefined) {
    index.entries[key] = storedBytes;
    index.bytes += storedBytes;
  }
  // Object key order is insertion order, so the first key is the oldest page.
  while (index.bytes > MAX_CACHE_BYTES) {
    const oldestKey = Object.keys(index.entries)[0];
    if (oldestKey === undefined) break;
    await cache.delete(oldestKey);
    index.bytes -= index.entries[oldestKey];
    delete index.entries[oldestKey];
  }
  await cache.put(INDEX_URL, new Response(JSON.stringify(index), {
    headers: { "Content-Type": "application/json" },
  }));
}

function queueIndexUpdate(cache, key, storedBytes) {
  indexUpdateQueue = indexUpdateQueue
    .then(() => updateIndex(cache, key, storedBytes))
    .catch((error) => console.error("rad cache index update failed", error));
  return indexUpdateQueue;
}

async function storePage(cache, request, response, rangeHeader) {
  const buffer = await response.clone().arrayBuffer();
  const contentRange = response.headers.get("Content-Range") || "full";
  const totalSize = contentRange === "full"
    ? buffer.byteLength
    : Number(contentRange.split("/")[1]) || buffer.byteLength;
  const key = cacheKeyFor(request, rangeHeader);
  const stored = new Response(buffer, {
    headers: {
      "Content-Type": response.headers.get("Content-Type") || "application/octet-stream",
      "X-SW-Content-Range": contentRange,
      "X-SW-Total-Size": String(totalSize),
      "X-SW-Bytes": String(buffer.byteLength),
    },
  });
  await cache.put(key, stored);
  await queueIndexUpdate(cache, key, buffer.byteLength);
}

// --- in-flight page fetches -------------------------------------------------
// Concurrent requests for the same uncached page share one network fetch and
// one store; every caller then serves from the cache exactly like a hit, so
// no consumer reads the original network body and the FIFO index updates
// exactly once per unique page.

const inFlightPages = new Map();

// Fetches one page and stores it; resolves true once the bytes are cached,
// null on any failure (failures are never cached and never leave waiters
// hanging — they fall through to a direct network passthrough).
async function fetchAndStorePage(cache, request, rangeHeader) {
  let response;
  try {
    response = await fetch(request);
  } catch (error) {
    console.error("rad page fetch failed", error);
    return null;
  }
  if (!response.ok && response.status !== 206) return null;
  try {
    await storePage(cache, request, response, rangeHeader);
  } catch (error) {
    console.error("rad cache store failed", error);
    return null;
  }
  return true;
}

async function serveRad(request, rangeHeader) {
  const cache = await caches.open(CACHE_NAME);
  const key = cacheKeyFor(request, rangeHeader);
  const cached = await cache.match(key);
  if (cached) return reconstructResponse(cached, rangeHeader);
  let inFlight = inFlightPages.get(key);
  if (!inFlight) {
    inFlight = fetchAndStorePage(cache, request, rangeHeader)
      .finally(() => inFlightPages.delete(key));
    inFlightPages.set(key, inFlight);
  }
  const stored = await inFlight;
  if (!stored) return fetch(request);
  const fresh = await cache.match(key);
  if (!fresh) return fetch(request);
  return reconstructResponse(fresh, rangeHeader);
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith("/artifacts/") || !url.pathname.endsWith(".rad")) return;
  event.respondWith(serveRad(request, request.headers.get("Range")));
});
