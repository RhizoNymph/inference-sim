import { createServer } from "node:http";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { spawn } from "node:child_process";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";

const guiDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(guiDir, "..");
const port = Number(process.env.PORT ?? process.argv[2] ?? 8787);
const simulatorTimeoutMs = 10 * 60 * 1000;

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
]);

const server = createServer(async (request, response) => {
  try {
    if (request.method === "POST" && request.url === "/api/run") {
      await runSimulation(request, response);
      return;
    }

    if (request.method !== "GET") {
      sendJson(response, 405, { error: "method not allowed" });
      return;
    }

    await serveStatic(request.url ?? "/", response);
  } catch (error) {
    sendJson(response, 500, { error: error.message });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`inference-sim GUI: http://127.0.0.1:${port}`);
});

async function runSimulation(request, response) {
  const body = await readBody(request);
  let payload;
  try {
    payload = JSON.parse(body);
  } catch {
    sendJson(response, 400, { error: "request body must be JSON" });
    return;
  }

  if (!payload.clusterToml || !payload.workloadToml) {
    sendJson(response, 400, { error: "clusterToml and workloadToml are required" });
    return;
  }

  const dir = await mkdtemp(join(tmpdir(), "inference-sim-gui-"));
  const clusterPath = join(dir, "cluster.toml");
  const workloadPath = join(dir, "workload.toml");

  try {
    await writeFile(clusterPath, payload.clusterToml, "utf8");
    await writeFile(workloadPath, payload.workloadToml, "utf8");
    const result = await cargoRun(
      clusterPath,
      workloadPath,
      Number(payload.requestLimit ?? 8),
      Number(payload.topK ?? 1),
      Number(payload.maxRankCandidates ?? 4096),
      Number(payload.maxServingPairs ?? 32),
      Number(payload.maxSearchRuntimeMs ?? 30000),
    );
    sendJson(response, result.ok ? 200 : 422, result);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function cargoRun(clusterPath, workloadPath, requestLimit, topK, maxRankCandidates, maxServingPairs, maxSearchRuntimeMs) {
  const safeRequestLimit = String(clampInt(requestLimit, 1, 128));
  const safeTopK = String(clampInt(topK, 1, 100));
  const safeRankCandidates = String(clampInt(maxRankCandidates, 1, 100000));
  const safeServingPairs = String(clampInt(maxServingPairs, 1, 256));
  const safeRuntimeMs = String(clampInt(maxSearchRuntimeMs, 1000, 110000));
  const args = [
    "run",
    "--quiet",
    "--",
    "--cluster",
    clusterPath,
    "--workload",
    workloadPath,
    "--json",
    "--top-k",
    safeTopK,
    "--request-limit",
    safeRequestLimit,
    "--max-candidates",
    safeRankCandidates,
    "--max-serving-pairs",
    safeServingPairs,
    "--max-search-runtime-ms",
    safeRuntimeMs,
  ];

  return new Promise((resolvePromise) => {
    const child = spawn("cargo", args, { cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
    }, simulatorTimeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      resolvePromise({ ok: false, error: error.message, stdout, stderr });
    });
    child.on("close", (code, signal) => {
      clearTimeout(timeout);
      if (code !== 0) {
        resolvePromise({ ok: false, error: `cargo exited with ${signal ?? code}`, stdout, stderr });
        return;
      }

      try {
        resolvePromise({ ok: true, result: JSON.parse(stdout), stderr });
      } catch (error) {
        resolvePromise({ ok: false, error: `simulator returned invalid JSON: ${error.message}`, stdout, stderr });
      }
    });
  });
}

async function serveStatic(url, response) {
  const rawPath = decodeURIComponent(new URL(url, "http://127.0.0.1").pathname);
  const requested = rawPath === "/" ? "/index.html" : rawPath;
  const normalized = normalize(requested).replace(/^(\.\.[/\\])+/, "");
  const filePath = resolve(guiDir, `.${normalized}`);

  if (!filePath.startsWith(guiDir)) {
    sendJson(response, 403, { error: "forbidden" });
    return;
  }

  try {
    await readFile(filePath);
  } catch {
    sendJson(response, 404, { error: "not found" });
    return;
  }

  response.writeHead(200, {
    "content-type": contentTypes.get(extname(filePath)) ?? "application/octet-stream",
  });
  createReadStream(filePath).pipe(response);
}

function readBody(request) {
  return new Promise((resolvePromise, reject) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 2_000_000) {
        reject(new Error("request body too large"));
        request.destroy();
      }
    });
    request.on("end", () => resolvePromise(body));
    request.on("error", reject);
  });
}

function sendJson(response, status, payload) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function clampInt(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, Math.round(value)));
}
