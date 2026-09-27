const GPU_PRESETS = {
  h100_sxm: { label: "H100 SXM5", hbmGb: 80, bandwidth: 3350, tflops: 989.5 },
  a100_80gb: { label: "A100 80GB", hbmGb: 80, bandwidth: 2039, tflops: 312 },
  mixed_h100_a100: { label: "Mixed H100/A100", hbmGb: 80, bandwidth: 2694, tflops: 650 },
};

const SCATTER_METRICS = [
  { key: "e2el_ms", label: "E2EL", format: formatCompactMs },
  { key: "ttft_ms", label: "TTFT", format: formatCompactMs },
  { key: "tpot_ms", label: "TPOT", format: formatCompactMs },
  { key: "throughput_tokens_per_s", label: "Throughput", format: formatCompactNumber },
  { key: "memory_pressure_peak_fraction", label: "Memory pressure", format: formatCompactPct },
  { key: "slo_miss_rate", label: "SLO miss rate", format: formatCompactPct },
];

const DEMO_RESULT = {
  schema_version: 1,
  mode: "serving",
  cluster_inventory: {
    node_count: 3,
    total_gpus: 24,
    available_gpus: 24,
    gpu_types: [
      { gpu: "H100 SXM5", count: 16, hbm_gb: 80 },
      { gpu: "A100 SXM4", count: 8, hbm_gb: 80 },
    ],
    nodes: [
      demoNode(0, "rack-a", "island-0", "H100 SXM5", 8, 8),
      demoNode(1, "rack-a", "island-0", "H100 SXM5", 8, 8),
      demoNode(2, "rack-b", "island-1", "A100 SXM4", 8, 4),
    ],
    inter_node_topology: {
      kind: "fat_tree",
      oversubscription: 1,
      default_link: { label: "IB NDR", bandwidth_gbps: 400, latency_us: 1.2 },
    },
  },
  results: [
    {
      rank: 1,
      status: "ok",
      feasible: true,
      deployment_mode: "fully_disaggregated",
      pool: "prefill:nodes[0,1] decode:nodes[2]",
      candidate_id: "demo-fully-disaggregated",
      prefill_config: { tensor_ranks: 2, pipeline_ranks: 1, expert_ranks: 1, data_ranks: 2 },
      decode_config: { tensor_ranks: 1, pipeline_ranks: 1, expert_ranks: 1, data_ranks: 1 },
      prefill_nodes: [0, 1],
      decode_nodes: [2],
      metrics: {
        ttft_ms: 68.4,
        tpot_ms: 10.9,
        e2el_ms: 232.7,
        throughput_tokens_per_s: 728,
        scheduled_makespan_ms: 238,
      },
      serving_services: [
        { phase: "prefill", health: "healthy", node_count: 2, gpu_count: 4, service_ms: 54, queue_ms: 6 },
        { phase: "kv_transfer", health: "healthy", node_count: 3, gpu_count: 8, service_ms: 12, queue_ms: 2 },
        { phase: "decode", health: "healthy", node_count: 1, gpu_count: 2, service_ms: 159, queue_ms: 18 },
      ],
      request_observations: [
        {
          request_id: "demo-0",
          status: "completed",
          lifecycle_events: [
            { kind: "arrived", phase: "arrival", time_ms: 0 },
            { kind: "prefill_started", phase: "prefill", time_ms: 4 },
            { kind: "prefill_finished", phase: "prefill", time_ms: 52 },
            { kind: "kv_transfer_started", phase: "kv_transfer", time_ms: 52 },
            { kind: "kv_transfer_finished", phase: "kv_transfer", time_ms: 64 },
            { kind: "decode_iteration_started", phase: "decode", time_ms: 72 },
            { kind: "decode_iteration_finished", phase: "decode", time_ms: 215 },
            { kind: "completed", phase: "terminal", time_ms: 232 },
          ],
        },
        {
          request_id: "demo-1",
          status: "completed",
          lifecycle_events: [
            { kind: "arrived", phase: "arrival", time_ms: 10 },
            { kind: "prefill_started", phase: "prefill", time_ms: 18 },
            { kind: "prefill_finished", phase: "prefill", time_ms: 70 },
            { kind: "kv_transfer_started", phase: "kv_transfer", time_ms: 70 },
            { kind: "kv_transfer_finished", phase: "kv_transfer", time_ms: 84 },
            { kind: "decode_iteration_started", phase: "decode", time_ms: 91 },
            { kind: "decode_iteration_finished", phase: "decode", time_ms: 238 },
            { kind: "completed", phase: "terminal", time_ms: 246 },
          ],
        },
      ],
    },
  ],
};

const state = {
  result: null,
  running: false,
  scatterXMetric: "e2el_ms",
  scatterYMetric: "throughput_tokens_per_s",
};

const form = document.querySelector("#configForm");
const jsonInput = document.querySelector("#jsonInput");
const scatterXMetricSelect = document.querySelector("#scatterXMetric");
const scatterYMetricSelect = document.querySelector("#scatterYMetric");

function demoNode(nodeId, rack, island, gpu, gpuCount, nicCount) {
  return {
    node_id: nodeId,
    topology: { rack, island, failure_domain: rack, labels: [] },
    gpu_count: gpuCount,
    available_gpu_count: gpuCount,
    gpu_types: [{ gpu, count: gpuCount }],
    gpus: Array.from({ length: gpuCount }, (_, local_gpu_id) => ({
      local_gpu_id,
      gpu,
      available: true,
      rail_ids: [local_gpu_id % Math.max(1, nicCount)],
      nic_ids: [local_gpu_id % Math.max(1, nicCount)],
    })),
    nics: { count: nicCount, active_count: nicCount, rail_count: nicCount },
  };
}

function readConfig() {
  const data = new FormData(form);
  const number = (name) => Number(data.get(name));
  const config = {
    gpuPreset: String(data.get("gpuPreset")),
    nodeCount: clamp(number("nodeCount"), 1, 64),
    gpusPerNode: clamp(number("gpusPerNode"), 1, 16),
    nicsPerNode: clamp(number("nicsPerNode"), 0, 16),
    railCount: clamp(number("railCount"), 0, 16),
    fabricKind: String(data.get("fabricKind")),
    oversubscription: Math.max(1, number("oversubscription")),
    modelId: sanitizeTomlString(String(data.get("modelId") || "gui generated bf16 transformer")),
    dtype: String(data.get("dtype")),
    layers: clamp(number("layers"), 1, 1000),
    hiddenSize: clamp(number("hiddenSize"), 1, 100000),
    attentionHeads: clamp(number("attentionHeads"), 1, 1000),
    kvHeads: clamp(number("kvHeads"), 1, 1000),
    ffnHiddenSize: clamp(number("ffnHiddenSize"), 1, 1000000),
    vocabSize: clamp(number("vocabSize"), 1, 10000000),
    servingMode: "flexible",
    targetMetric: String(data.get("targetMetric")),
    topK: clamp(number("topK"), 1, 100),
    maxRankCandidates: clamp(number("maxRankCandidates"), 1, 100000),
    maxServingPairs: clamp(number("maxServingPairs"), 1, 256),
    maxPoolCandidates: clamp(number("maxPoolCandidates"), 1, 10000),
    maxSearchRuntimeMs: clamp(number("maxSearchRuntimeMs"), 1000, 110000),
    maxE2elS: clampFloat(number("maxE2elS"), 0, 1000000),
    requestCount: clamp(number("requestCount"), 1, 100000),
    arrivalPattern: ["fixed", "poisson"].includes(String(data.get("arrivalPattern"))) ? String(data.get("arrivalPattern")) : "poisson",
    arrivalRatePerS: clampFloat(number("arrivalRatePerS"), 0.01, 100000),
    arrivalGapMs: Math.max(0, number("arrivalGapMs")),
    maxPrefillBatchTokens: clamp(number("maxPrefillBatchTokens"), 1, 10000000),
    maxPrefillChunkTokens: clamp(number("maxPrefillChunkTokens"), 1, 10000000),
    maxDecodeBatchTokens: clamp(number("maxDecodeBatchTokens"), 1, 10000000),
    maxDecodeSequences: clamp(number("maxDecodeSequences"), 1, 1000000),
    maxResidentTokens: clamp(number("maxResidentTokens"), 1, 100000000),
    kvBlockTokens: clamp(number("kvBlockTokens"), 1, 1000000),
    maxKvBlocks: clamp(number("maxKvBlocks"), 1, 10000000),
    promptMedianTokens: clamp(number("promptMedianTokens"), 1, 1000000),
    promptSigma: clampFloat(number("promptSigma"), 0.01, 5),
    promptMinTokens: clamp(number("promptMinTokens"), 1, 1000000),
    promptMaxTokens: clamp(number("promptMaxTokens"), 1, 1000000),
    decodeMinTokens: clamp(number("decodeMinTokens"), 1, 1000000),
    decodeMaxTokens: clamp(number("decodeMaxTokens"), 1, 1000000),
    maxSequenceTokens: clamp(number("maxSequenceTokens"), 1, 2000000),
  };
  config.promptMaxTokens = Math.max(config.promptMinTokens, config.promptMaxTokens);
  config.promptMedianTokens = clamp(config.promptMedianTokens, config.promptMinTokens, config.promptMaxTokens);
  config.decodeMaxTokens = Math.max(config.decodeMinTokens, config.decodeMaxTokens);
  config.maxPrefillChunkTokens = Math.min(config.maxPrefillChunkTokens, config.maxPrefillBatchTokens);
  config.maxSequenceTokens = Math.max(config.maxSequenceTokens, config.promptMaxTokens + config.decodeMaxTokens);
  return config;
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, Math.round(value)));
}

function clampFloat(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function estimatedModelParameters(config) {
  const headDim = config.hiddenSize / Math.max(1, config.attentionHeads);
  const kvDim = config.kvHeads * headDim;
  const attentionPerLayer =
    config.hiddenSize * config.hiddenSize +
    config.hiddenSize * kvDim * 2 +
    config.hiddenSize * config.hiddenSize;
  const ffnPerLayer = 3 * config.hiddenSize * config.ffnHiddenSize;
  const normPerLayer = 4 * config.hiddenSize;
  const embedding = config.vocabSize * config.hiddenSize;
  const finalNorm = config.hiddenSize;
  return config.layers * (attentionPerLayer + ffnPerLayer + normPerLayer) + embedding + finalNorm;
}

function dtypeBytes(dtype) {
  if (dtype === "fp8") return 1;
  return 2;
}

function estimatedModelWeightGb(config) {
  return (estimatedModelParameters(config) * dtypeBytes(config.dtype)) / 1_000_000_000;
}

function representativePromptTokens(config) {
  return config.promptMedianTokens;
}

function representativeDecodeTokens(config) {
  return Math.round((config.decodeMinTokens + config.decodeMaxTokens) / 2);
}

function arrivalTimeMs(config, requestIndex) {
  if (config.arrivalPattern === "poisson") {
    return (requestIndex / Math.max(config.arrivalRatePerS, 0.01)) * 1000;
  }
  return requestIndex * config.arrivalGapMs;
}

function sanitizeTomlString(value) {
  return value.replace(/["\\]/g, " ").trim() || "gui generated bf16 transformer";
}

function liveInventory(config) {
  const preset = GPU_PRESETS[config.gpuPreset] ?? GPU_PRESETS.h100_sxm;
  const nodes = Array.from({ length: config.nodeCount }, (_, nodeId) => {
    const gpuLabel =
      config.gpuPreset === "mixed_h100_a100" && nodeId >= Math.ceil(config.nodeCount / 2)
        ? "A100 SXM4"
        : preset.label === "Mixed H100/A100"
          ? "H100 SXM5"
          : preset.label;
    return demoNode(nodeId, `rack-${Math.floor(nodeId / 8)}`, `island-${Math.floor(nodeId / 4)}`, gpuLabel, config.gpusPerNode, config.nicsPerNode);
  });
  return {
    node_count: config.nodeCount,
    total_gpus: config.nodeCount * config.gpusPerNode,
    available_gpus: config.nodeCount * config.gpusPerNode,
    gpu_types: summarizeGpuTypes(nodes),
    nodes,
    inter_node_topology: {
      kind: "fat_tree",
      oversubscription: config.oversubscription,
      default_link: {
        label: config.fabricKind.toUpperCase(),
        bandwidth_gbps: config.fabricKind === "ethernet" ? 100 : 400,
        latency_us: config.fabricKind === "ethernet" ? 8 : 1.2,
      },
    },
  };
}

function summarizeGpuTypes(nodes) {
  const counts = new Map();
  for (const node of nodes) {
    for (const entry of node.gpu_types ?? []) {
      counts.set(entry.gpu, (counts.get(entry.gpu) ?? 0) + entry.count);
    }
  }
  return [...counts.entries()].map(([gpu, count]) => ({ gpu, count, hbm_gb: 80 }));
}

function selectedResult() {
  return state.result?.results?.[0] ?? null;
}

function activeInventory(config) {
  return state.result?.cluster_inventory ?? liveInventory(config);
}

function activePlacement(config) {
  const result = selectedResult();
  const allNodes = Array.from({ length: config.nodeCount }, (_, id) => id);
  return {
    prefill: numericArray(result?.prefill_nodes) ?? allNodes,
    decode: numericArray(result?.decode_nodes) ?? allNodes,
  };
}

function numericArray(value) {
  if (!Array.isArray(value)) return null;
  const out = value.filter((item) => Number.isInteger(item));
  return out.length ? out : null;
}

function render() {
  const config = readConfig();
  const inventory = activeInventory(config);
  const placement = activePlacement(config);
  scatterXMetricSelect.value = state.scatterXMetric;
  scatterYMetricSelect.value = state.scatterYMetric;
  document.querySelector("#configSummary").textContent = `${config.nodeCount} nodes, auto search`;
  document.querySelector("#modelWeightEstimate").value = `${estimatedModelWeightGb(config).toFixed(2)} GB (${(estimatedModelParameters(config) / 1_000_000_000).toFixed(2)}B params)`;
  document.querySelector("#clusterToml").value = buildClusterToml(config);
  document.querySelector("#workloadToml").value = buildWorkloadToml(config);
  document.querySelector("#runCommand").textContent =
    `node gui/server.mjs\n# open http://127.0.0.1:8787\n\ncargo run -- --cluster cluster.toml --workload workload.toml --json --top-k ${config.topK} --request-limit 2 --max-candidates ${config.maxRankCandidates} --max-serving-pairs ${config.maxServingPairs} --max-search-runtime-ms ${config.maxSearchRuntimeMs}`;
  document.querySelector("#serverHint").textContent = "The Run simulation button requires the local server: node gui/server.mjs";
  renderTopology(inventory, placement);
  renderInventory(inventory, config);
  renderParallelism(config, selectedResult());
  renderTimeline(config, selectedResult());
  renderResultSummary(config, selectedResult());
}

function renderTopology(inventory, placement) {
  const svg = document.querySelector("#topologySvg");
  svg.textContent = "";
  const nodes = inventory.nodes ?? [];
  const width = 960;
  const cols = Math.max(1, Math.min(4, Math.ceil(Math.sqrt(nodes.length || 1))));
  const rows = Math.max(1, Math.ceil((nodes.length || 1) / cols));
  const cardW = 190;
  const cardH = 128;
  const gapX = 34;
  const gapY = 34;
  const height = 78 + rows * cardH + Math.max(0, rows - 1) * gapY + 36;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const startX = Math.max(24, (width - cols * cardW - (cols - 1) * gapX) / 2);
  addLine(svg, 80, 44, 880, 44, "fabric-line");
  addText(svg, 480, 30, `${inventory.inter_node_topology?.default_link?.label ?? "fabric"} fabric`, "svg-label", "middle");

  nodes.forEach((node, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const x = startX + col * (cardW + gapX);
    const y = 78 + row * (cardH + gapY);
    const role = nodeRole(node.node_id, placement);
    addLine(svg, x + cardW / 2, 44, x + cardW / 2, y, "fabric-line");
    addRect(svg, x, y, cardW, cardH, `node-box ${role}`);
    addText(svg, x + 14, y + 24, `node ${node.node_id}`, "svg-label");
    addText(svg, x + 14, y + 44, node.topology?.rack ?? node.topology?.island ?? "default domain", "svg-small");
    addText(svg, x + cardW - 14, y + 24, roleLabel(role), "svg-small", "end");

    const gpus = node.gpus ?? Array.from({ length: node.gpu_count ?? 0 }, (_, local_gpu_id) => ({ local_gpu_id, available: true }));
    gpus.slice(0, 16).forEach((gpu, gpuIndex) => {
      const gx = x + 18 + (gpuIndex % 8) * 20;
      const gy = y + 68 + Math.floor(gpuIndex / 8) * 22;
      addCircle(svg, gx, gy, 7, gpu.available === false ? "gpu-dot disabled" : "gpu-dot");
    });

    const nicCount = node.nics?.active_count ?? node.nics?.count ?? 0;
    for (let nic = 0; nic < Math.min(nicCount, 8); nic += 1) {
      addCircle(svg, x + 20 + nic * 20, y + 112, 4.5, "nic-dot");
    }
  });

  document.querySelector("#topologySubtitle").textContent = state.result
    ? "Rendered from simulator JSON cluster_inventory."
    : "Live preview from the editable cluster config.";
}

function nodeRole(nodeId, placement) {
  const prefill = placement.prefill.includes(nodeId);
  const decode = placement.decode.includes(nodeId);
  if (prefill && decode) return "shared";
  if (prefill) return "prefill";
  if (decode) return "decode";
  return "";
}

function roleLabel(role) {
  if (role === "shared") return "prefill + decode";
  if (role === "prefill") return "prefill";
  if (role === "decode") return "decode";
  return "idle";
}

function renderInventory(inventory, config) {
  const inventoryEl = document.querySelector("#inventory");
  const totalHbm = (inventory.gpu_types ?? []).reduce((sum, gpu) => sum + (gpu.count ?? 0) * (gpu.hbm_gb ?? 80), 0);
  inventoryEl.innerHTML = [
    statCard(inventory.node_count ?? config.nodeCount, "nodes"),
    statCard(inventory.total_gpus ?? inventory.available_gpus ?? 0, "total GPUs"),
    statCard(`${Math.round(totalHbm).toLocaleString()} GB`, "aggregate HBM"),
    statCard(inventory.inter_node_topology?.default_link?.bandwidth_gbps ? `${inventory.inter_node_topology.default_link.bandwidth_gbps} Gb/s` : "configured", "fabric link"),
    statCard((inventory.gpu_types ?? []).map((gpu) => `${gpu.count} ${gpu.gpu}`).join(", ") || "GPU inventory", "GPU types"),
  ].join("");
}

function statCard(value, label) {
  return `<article class="stat-card"><strong>${escapeHtml(String(value))}</strong><span>${escapeHtml(label)}</span></article>`;
}

function generatedParallelismSearch(config) {
  const totalGpus = Math.max(1, config.nodeCount * config.gpusPerNode);
  const tp = divisorsUpTo(config.attentionHeads, Math.min(totalGpus, config.gpusPerNode, config.attentionHeads)).filter(
    (rank) => config.hiddenSize % rank === 0,
  );
  const pp = integersUpTo(Math.min(totalGpus, config.layers));
  const ep = [1];
  const dp = integersUpTo(totalGpus);
  return { tp, pp, ep, dp };
}

function generatedPoolNodeCounts(config) {
  return integersUpTo(config.nodeCount);
}

function generatedSearchSummary(config) {
  const search = generatedParallelismSearch(config);
  const nodeCounts = generatedPoolNodeCounts(config);
  return {
    search,
    nodeCounts,
    rankCombos: search.tp.length * search.pp.length * search.ep.length * search.dp.length,
  };
}

function maxServingPairs(config) {
  return config.maxServingPairs;
}

function maxSearchRuntimeMs(config) {
  return config.maxSearchRuntimeMs;
}

function divisorsUpTo(value, max) {
  const upper = Math.max(1, Math.min(Math.trunc(value), Math.trunc(max)));
  const out = [];
  for (let candidate = 1; candidate <= upper; candidate += 1) {
    if (value % candidate === 0) out.push(candidate);
  }
  return out.length ? out : [1];
}

function integersUpTo(max) {
  const upper = Math.max(1, Math.trunc(max));
  return Array.from({ length: upper }, (_, index) => index + 1);
}

function renderParallelism(config, result) {
  const prefill = result?.prefill_config ?? parallelConfig(config);
  const decode = result?.decode_config ?? parallelConfig(config);
  const generated = generatedSearchSummary(config);
  document.querySelector("#parallelismCards").innerHTML = [
    rankCard("Target metric", objectiveLabel(config.targetMetric)),
    rankCard("Reported configs", state.result?.results?.length ?? config.topK),
    rankCard("Rank combos", generated.rankCombos),
    rankCard("Rank candidate budget", config.maxRankCandidates),
    rankCard("Serving pair budget", config.maxServingPairs),
    rankCard("Pool candidate cap", config.maxPoolCandidates),
  ].join("");
  renderCandidateRankings(config);
  renderMetricScatter(config);

  const svg = document.querySelector("#parallelismSvg");
  svg.textContent = "";
  const phases = [
    { name: "prefill", config: prefill, x: 80, color: "#0e8f9f" },
    { name: "kv transfer", config: null, x: 390, color: "#b7791f" },
    { name: "decode", config: decode, x: 660, color: "#2463eb" },
  ];
  phases.forEach((phase) => {
    addRect(svg, phase.x, 64, 220, 220, "phase-box");
    addText(svg, phase.x + 110, 94, phase.name, "svg-label", "middle");
    if (phase.config) {
      const ranks = [
        ["TP", phase.config.tensor_ranks],
        ["PP", phase.config.pipeline_ranks],
        ["EP", phase.config.expert_ranks],
        ["DP", phase.config.data_ranks],
      ];
      ranks.forEach(([label, value], idx) => {
        const y = 126 + idx * 36;
        addRect(svg, phase.x + 34, y, 152, 25, "rank-box");
        addText(svg, phase.x + 52, y + 17, `${label} ${value}`, "svg-small");
        addText(svg, phase.x + 170, y + 17, `${value} ranks`, "svg-small", "end");
      });
      addText(svg, phase.x + 110, 260, `${totalRanks(phase.config)} total ranks`, "svg-label", "middle");
    } else {
      addText(svg, phase.x + 110, 160, "KV cache handoff", "svg-label", "middle");
      addText(svg, phase.x + 110, 188, "route and rail costs", "svg-small", "middle");
    }
  });
  addLine(svg, 300, 174, 390, 174, "fabric-line");
  addLine(svg, 610, 174, 660, 174, "fabric-line");
}

function rankCard(label, value) {
  return `<article class="rank-card"><strong>${escapeHtml(String(value ?? 1))}</strong><span>${escapeHtml(label)}</span></article>`;
}

function parallelConfig(config) {
  const search = generatedParallelismSearch(config);
  return {
    tensor_ranks: search.tp[0] ?? 1,
    pipeline_ranks: search.pp[0] ?? 1,
    expert_ranks: search.ep[0] ?? 1,
    data_ranks: search.dp[0] ?? 1,
  };
}

function renderCandidateRankings(config) {
  const rankings = document.querySelector("#candidateRankings");
  const results = state.result?.results ?? [];
  if (!results.length) {
    const generated = generatedSearchSummary(config);
    rankings.innerHTML = `
      <div class="empty-ranking">
        Simulator rankings appear here after running. The solver will search TP ${generated.search.tp.join(", ")}, PP ${generated.search.pp.join(", ")}, EP ${generated.search.ep.join(", ")}, DP ${generated.search.dp.join(", ")} across pool sizes ${generated.nodeCounts.join(", ")}.
      </div>
    `;
    return;
  }

  rankings.innerHTML = `
    ${renderRejectionSummary(state.result)}
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Status</th>
          <th>Prefill</th>
          <th>Decode</th>
          <th>Reason</th>
          <th>TTFT</th>
          <th>TPOT</th>
          <th>E2EL</th>
          <th>Throughput</th>
          <th>Memory</th>
        </tr>
      </thead>
      <tbody>
        ${results
          .map((candidate, index) => {
            const metrics = candidate.metrics ?? {};
            return `
              <tr>
                <td>${candidate.rank ?? index + 1}</td>
                <td>${escapeHtml(candidate.status ?? (candidate.feasible ? "ok" : "rejected"))}</td>
                <td>${escapeHtml(configSummary(candidate.prefill_config))}</td>
                <td>${escapeHtml(configSummary(candidate.decode_config))}</td>
                <td class="reason-cell">${escapeHtml(candidateRejectionSummary(candidate))}</td>
                <td>${formatMaybeMs(metrics.ttft_ms)}</td>
                <td>${formatMaybeMs(metrics.tpot_ms)}</td>
                <td>${formatMaybeMs(metrics.e2el_ms)}</td>
                <td>${formatMaybeNumber(metrics.throughput_tokens_per_s)}</td>
                <td>${formatMaybePct(candidate.memory_pressure_peak_fraction ?? candidate.max_memory_pressure_fraction)}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderRejectionSummary(result) {
  const summary = result?.rejection_summary;
  const groups = summary?.groups ?? [];
  const rejected = Number(summary?.rejected_candidate_count ?? 0);
  if (!rejected && !groups.length) return "";

  const topGroups = groups.slice(0, 4);
  return `
    <section class="rejection-summary">
      <div class="rejection-summary-title">
        <strong>${rejected} rejected</strong>
        <span>${Number(summary?.total_rejection_count ?? 0)} rejection events across ${Number(summary?.candidate_with_rejections_count ?? 0)} candidates</span>
      </div>
      <div class="rejection-list">
        ${topGroups
          .map((group) => {
            const label = [group.phase, group.category, group.resource, group.code].filter(Boolean).join(" / ");
            const message = (group.messages ?? [])[0] ?? "";
            const remediation = (group.remediations ?? [])[0] ?? "";
            return `
              <article>
                <strong>${escapeHtml(label)}</strong>
                <span>${escapeHtml(message)}</span>
                ${remediation ? `<em>${escapeHtml(remediation)}</em>` : ""}
              </article>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function candidateRejectionSummary(candidate) {
  if (candidate.feasible !== false && candidate.status !== "rejected") return "ok";
  if (candidate.rejected_reason) return candidate.rejected_reason;
  const first = (candidate.rejections ?? [])[0];
  if (!first) return "rejected";
  const message = first.message || [first.phase, first.category, first.resource, first.code].filter(Boolean).join(" / ");
  const remediation = first.remediation ? ` Remediation: ${first.remediation}` : "";
  return `${message}${remediation}`;
}

function renderMetricScatter(config) {
  const svg = document.querySelector("#metricScatterSvg");
  svg.textContent = "";
  const results = state.result?.results ?? [];
  const xMetric = state.scatterXMetric;
  const yMetric = state.scatterYMetric;
  const xInfo = SCATTER_METRICS.find((item) => item.key === xMetric) ?? SCATTER_METRICS[0];
  const yInfo = SCATTER_METRICS.find((item) => item.key === yMetric) ?? SCATTER_METRICS[1];

  const width = 960;
  const height = 320;
  const pad = { left: 86, right: 36, top: 32, bottom: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  addRect(svg, 0, 0, width, height, "scatter-bg");
  addLine(svg, pad.left, pad.top, pad.left, height - pad.bottom, "scatter-axis");
  addLine(svg, pad.left, height - pad.bottom, width - pad.right, height - pad.bottom, "scatter-axis");

  if (!results.length) {
    const generated = generatedSearchSummary(config);
    addText(svg, width / 2, height / 2 - 8, "Run the simulator to plot ranked candidates", "scatter-empty", "middle");
    addText(svg, width / 2, height / 2 + 18, `Solver-owned search: ${generated.rankCombos} rank combos, pool sizes ${generated.nodeCounts.join(", ")}`, "svg-small", "middle");
    addText(svg, width / 2, height - 16, xInfo.label, "scatter-axis-label", "middle");
    addRotatedText(svg, 20, height / 2, yInfo.label, "scatter-axis-label", -90);
    return;
  }

  const points = results
    .map((candidate, index) => ({
      candidate,
      rank: Number(candidate.rank ?? index + 1),
      x: candidateMetricValue(candidate, xMetric),
      y: candidateMetricValue(candidate, yMetric),
    }))
    .filter((point) => Number.isFinite(point.rank) && Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!points.length) {
    const rejected = Number(state.result?.rejection_summary?.rejected_candidate_count ?? 0);
    const message = rejected ? "All plotted candidates were rejected; see rejection reasons below" : `No paired ${xInfo.label} / ${yInfo.label} values in ranked results`;
    addText(svg, width / 2, height / 2, message, "scatter-empty", "middle");
    addText(svg, width / 2, height - 16, xInfo.label, "scatter-axis-label", "middle");
    addRotatedText(svg, 20, height / 2, yInfo.label, "scatter-axis-label", -90);
    return;
  }

  const xScale = paddedScale(points.map((point) => point.x));
  const yScale = paddedScale(points.map((point) => point.y));

  for (let i = 0; i <= 4; i += 1) {
    const ratio = i / 4;
    const y = pad.top + plotH * ratio;
    const value = yScale.max - yScale.span * ratio;
    addLine(svg, pad.left, y, width - pad.right, y, "scatter-grid-line");
    addText(svg, pad.left - 10, y + 4, yInfo.format(value), "scatter-tick", "end");
  }

  scatterTicks(xScale).forEach((value) => {
    const x = pad.left + ((value - xScale.min) / xScale.span) * plotW;
    addLine(svg, x, pad.top, x, height - pad.bottom, "scatter-grid-line");
    addLine(svg, x, height - pad.bottom, x, height - pad.bottom + 6, "scatter-axis");
    addText(svg, x, height - pad.bottom + 24, xInfo.format(value), "scatter-tick", "middle");
  });

  points.forEach((point) => {
    const x = pad.left + ((point.x - xScale.min) / xScale.span) * plotW;
    const y = pad.top + ((yScale.max - point.y) / yScale.span) * plotH;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", point.candidate.pareto_frontier ? 7.5 : point.candidate.feasible === false ? 5 : 6.2);
    circle.setAttribute(
      "class",
      `scatter-point${point.candidate.pareto_frontier ? " frontier" : ""}${point.candidate.feasible === false ? " rejected" : ""}`,
    );
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `rank ${point.rank}${point.candidate.pareto_frontier ? " / Pareto frontier" : ""}: ${configSummary(point.candidate.prefill_config)} / ${configSummary(point.candidate.decode_config)} / ${xInfo.label} ${xInfo.format(point.x)} / ${yInfo.label} ${yInfo.format(point.y)}`;
    circle.append(title);
    svg.append(circle);
    addText(svg, x, y - 11, String(point.rank), "scatter-point-label", "middle");
  });

  addText(svg, width / 2, height - 16, xInfo.label, "scatter-axis-label", "middle");
  addRotatedText(svg, 20, height / 2, yInfo.label, "scatter-axis-label", -90);
}

function configSummary(config) {
  if (!config) return "n/a";
  return `TP${config.tensor_ranks ?? 1} PP${config.pipeline_ranks ?? 1} EP${config.expert_ranks ?? 1} DP${config.data_ranks ?? 1}`;
}

function totalRanks(config) {
  return (config.tensor_ranks ?? 1) * (config.pipeline_ranks ?? 1) * (config.expert_ranks ?? 1) * (config.data_ranks ?? 1);
}

function renderTimeline(config, result) {
  const timeline = document.querySelector("#timeline");
  const status = document.querySelector("#resultStatus");
  let rows = result ? timelineFromResult(result) : estimatedTimeline(config);
  rows = rows.slice(0, 24);
  const max = Math.max(...rows.map((row) => row.finish), 1);
  timeline.innerHTML = rows
    .map((row) => {
      const left = (row.start / max) * 100;
      const width = Math.max(((row.finish - row.start) / max) * 100, 0.8);
      return `
        <div class="timeline-row">
          <div class="timeline-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</div>
          <div class="timeline-track">
            <div class="timeline-bar ${escapeHtml(row.phase)}" style="left:${left}%;width:${width}%"></div>
          </div>
          <div class="timeline-time">${formatMs(row.finish - row.start)}</div>
        </div>
      `;
    })
    .join("");
  if (!state.running) {
    status.className = `result-status ${result ? "ok" : ""}`;
    status.textContent = result
      ? `Loaded ${result.candidate_id ?? "candidate"}: ${result.status ?? "unknown"}${result.metrics?.e2el_ms ? `, E2EL ${formatMs(result.metrics.e2el_ms)}` : ""}.`
      : "No simulator JSON loaded. Showing estimated phase flow from form inputs.";
  }
  document.querySelector("#processSubtitle").textContent = result
    ? "Request lifecycle bars rendered from simulator JSON."
    : "Estimated phase flow from the editable config.";
}

function renderResultSummary(config, result) {
  const metricsEl = document.querySelector("#resultMetrics");
  const resourceEl = document.querySelector("#resourceUtilization");
  if (!result) {
    metricsEl.innerHTML = [
      statCard(`${config.requestCount}`, "configured requests"),
      statCard(`${config.promptMedianTokens} p50 / ${config.promptMinTokens}-${config.promptMaxTokens}`, "prompt tokens"),
      statCard(`${config.decodeMinTokens}-${config.decodeMaxTokens}`, "decode tokens"),
      statCard(objectiveLabel(config.targetMetric), "target metric"),
      statCard(`${config.topK}`, "ranked configs"),
    ].join("");
    resourceEl.innerHTML = "";
    return;
  }

  const metrics = result.metrics ?? {};
  metricsEl.innerHTML = [
    statCard(formatMaybeMs(metrics.ttft_ms), "TTFT"),
    statCard(formatMaybeMs(metrics.tpot_ms), "TPOT"),
    statCard(formatMaybeMs(metrics.e2el_ms), "E2EL"),
    statCard(formatMaybeNumber(metrics.throughput_tokens_per_s), "tokens / s"),
  ].join("");

  const utilization = (result.phase_resource_utilization ?? result.resource_utilization ?? []).slice(0, 8);
  resourceEl.innerHTML = utilization.length
    ? `
      <h3>Resource Utilization</h3>
      <div class="resource-grid">
        ${utilization
          .map((item) => {
            const label = item.phase ? `${item.phase}: ${item.resource}` : item.resource;
            const value = Number(item.utilization ?? 0);
            return `
              <div class="resource-row">
                <span title="${escapeHtml(label)}">${escapeHtml(label)}</span>
                <div class="resource-meter"><div style="width:${Math.max(0, Math.min(100, value * 100))}%"></div></div>
                <strong>${Math.round(value * 100)}%</strong>
              </div>
            `;
          })
          .join("")}
      </div>
    `
    : "";
}

function timelineFromResult(result) {
  const observations = result.request_observations ?? result.requests ?? [];
  const rows = [];
  observations.slice(0, 8).forEach((request, requestIndex) => {
    const events = request.lifecycle_events ?? [];
    addEventSegment(rows, events, requestIndex, "prefill_started", "prefill_finished", "prefill");
    addEventSegment(rows, events, requestIndex, "kv_transfer_started", "kv_transfer_finished", "kv_transfer");
    addEventSegment(rows, events, requestIndex, "decode_iteration_started", "decode_iteration_finished", "decode");
  });
  if (rows.length) return rows;

  const operations = result.scheduled_operations ?? [];
  operations.slice(0, 24).forEach((operation) => {
    const start = Number(operation.start_ms ?? 0);
    const finish = Number(operation.finish_ms ?? start + (operation.duration_ms ?? 1));
    rows.push({
      label: operation.name ?? `operation ${operation.id ?? rows.length}`,
      phase: phaseClass(operation.phase ?? operation.name ?? ""),
      start,
      finish,
    });
  });
  if (rows.length) return rows;

  let cursor = 0;
  const services = result.service_observations ?? result.serving_services ?? [];
  for (const service of services) {
    const queue = Number(service.queue_ms ?? 0);
    const serviceMs = Math.max(1, Number(service.service_ms ?? 1));
    if (queue > 0) {
      rows.push({ label: `${service.phase} queue`, phase: "queue", start: cursor, finish: cursor + queue });
      cursor += queue;
    }
    rows.push({ label: `${service.phase} service`, phase: phaseClass(service.phase), start: cursor, finish: cursor + serviceMs });
    cursor += serviceMs;
  }
  return rows.length ? rows : [{ label: "candidate evaluated", phase: "prefill", start: 0, finish: Number(result.metrics?.e2el_ms ?? 1) }];
}

function addEventSegment(rows, events, requestIndex, startKind, finishKind, phase) {
  const start = eventTime(events, startKind);
  const finish = phase === "decode" ? lastEventTime(events, finishKind) : eventTime(events, finishKind);
  if (start == null || finish == null || finish < start) return;
  rows.push({
    label: `request ${requestIndex} ${phase.replace("_", " ")}`,
    phase,
    start,
    finish,
  });
}

function eventTime(events, kind) {
  const event = events.find((item) => item.kind === kind || item.event === kind);
  if (!event) return null;
  return readEventTime(event);
}

function lastEventTime(events, kind) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.kind === kind || event.event === kind) return readEventTime(event);
  }
  return null;
}

function readEventTime(event) {
  const raw =
    event.time_ms ??
    event.timestamp_ms ??
    event.at_ms ??
    (event.time_s != null ? event.time_s * 1000 : null) ??
    (event.at_s != null ? event.at_s * 1000 : null);
  return Number.isFinite(Number(raw)) ? Number(raw) : null;
}

function estimatedTimeline(config) {
  const requestCount = Math.min(config.requestCount, 10);
  const rows = [];
  const representative = parallelConfig(config);
  const rankScale = Math.max(1, totalRanks(representative));
  const promptTokens = representativePromptTokens(config);
  const decodeTokens = representativeDecodeTokens(config);
  const prefillMs = Math.max(8, promptTokens / (5 * rankScale));
  const transferMs = Math.max(1, promptTokens / 96);
  const decodeMs = Math.max(12, (decodeTokens * 6) / Math.max(1, representative.tensor_ranks));
  for (let request = 0; request < requestCount; request += 1) {
    const arrival = arrivalTimeMs(config, request);
    rows.push({ label: `request ${request} prefill`, phase: "prefill", start: arrival, finish: arrival + prefillMs });
    rows.push({ label: `request ${request} kv transfer`, phase: "kv_transfer", start: arrival + prefillMs, finish: arrival + prefillMs + transferMs });
    rows.push({ label: `request ${request} decode`, phase: "decode", start: arrival + prefillMs + transferMs, finish: arrival + prefillMs + transferMs + decodeMs });
  }
  return rows;
}

function phaseClass(phase) {
  if (String(phase).includes("kv")) return "kv_transfer";
  if (String(phase).includes("decode")) return "decode";
  if (String(phase).includes("prefill")) return "prefill";
  return "queue";
}

function buildClusterToml(config) {
  if (config.gpuPreset === "mixed_h100_a100") {
    return buildMixedClusterToml(config);
  }

  if (config.gpuPreset === "a100_80gb") {
    return buildA100ClusterToml(config);
  }

  return `schema_version = 1

[cluster]
preset = "${config.gpuPreset === "mixed_h100_a100" ? "h100_sxm" : config.gpuPreset}"
node_count = ${config.nodeCount}

[interconnect]
kind = "${config.fabricKind}"
variant = "${config.fabricKind === "ib" ? "ndr" : config.fabricKind}"
oversubscription = ${config.oversubscription.toFixed(2)}

[nics]
count = ${config.nicsPerNode}
affinity = "dedicated"
bandwidth_gbps = ${config.fabricKind === "ethernet" ? "100.0" : "400.0"}
rail_count = ${config.railCount}
`;
}

function buildA100ClusterToml(config) {
  const nics = Math.max(1, config.nicsPerNode);
  const rails = Math.max(1, config.railCount);
  return `schema_version = 1

[cluster]
preset = "custom"

[interconnect]
kind = "${config.fabricKind}"
variant = "${config.fabricKind === "ib" ? "ndr" : config.fabricKind}"
oversubscription = ${config.oversubscription.toFixed(2)}

[[node_groups]]
label = "a100"
start_id = 0
count = ${config.nodeCount}
gpu = "a100_80gb"
gpu_count = ${config.gpusPerNode}
intra = "nvlink_v3"
nics = { count = ${nics}, affinity = "shared", gpus_per_nic = ${Math.max(1, Math.ceil(config.gpusPerNode / nics))}, bandwidth_gbps = ${config.fabricKind === "ethernet" ? "100.0" : "200.0"}, rail_count = ${rails} }
`;
}

function buildMixedClusterToml(config) {
  const h100Count = Math.max(1, Math.ceil(config.nodeCount / 2));
  const a100Count = Math.max(0, config.nodeCount - h100Count);
  const h100Nics = Math.max(1, config.nicsPerNode);
  const h100Rails = Math.max(1, config.railCount);
  const a100Nics = Math.max(1, Math.min(config.nicsPerNode, 4));
  const a100Rails = Math.max(1, Math.min(config.railCount, 4));
  return `schema_version = 1

[cluster]
preset = "custom"

[interconnect]
kind = "${config.fabricKind}"
variant = "${config.fabricKind === "ib" ? "ndr" : config.fabricKind}"
oversubscription = ${config.oversubscription.toFixed(2)}

[[node_groups]]
label = "h100"
start_id = 0
count = ${h100Count}
rack = "rack-a"
island = "h100-island"
failure_domain = "az-a"
gpu = "h100_sxm"
gpu_count = ${config.gpusPerNode}
intra = "nvlink_v4"
nics = { count = ${h100Nics}, affinity = "dedicated", bandwidth_gbps = ${config.fabricKind === "ethernet" ? "100.0" : "400.0"}, rail_count = ${h100Rails} }
${a100Count > 0 ? `
[[node_groups]]
label = "a100"
start_id = ${h100Count}
count = ${a100Count}
rack = "rack-b"
island = "a100-island"
failure_domain = "az-b"
gpu = "a100_80gb"
gpu_count = ${config.gpusPerNode}
intra = "nvlink_v3"
nics = { count = ${a100Nics}, affinity = "shared", gpus_per_nic = ${Math.max(1, Math.ceil(config.gpusPerNode / a100Nics))}, bandwidth_gbps = ${config.fabricKind === "ethernet" ? "100.0" : "200.0"}, rail_count = ${a100Rails} }
` : ""}`;
}

function trafficArrivalToml(config) {
  if (config.arrivalPattern === "poisson") {
    return `arrival = "poisson"
arrival_rate_per_s = ${config.arrivalRatePerS.toFixed(3)}
arrival_seed = 42`;
  }
  return `arrival = "fixed"
arrival_gap_ms = ${config.arrivalGapMs}`;
}

function servingMetricCeilingsToml(config) {
  if (config.maxE2elS <= 0) return "";
  return `max_e2el_s = ${config.maxE2elS.toFixed(3)}
`;
}

function buildWorkloadToml(config) {
  const generated = generatedSearchSummary(config);
  const ranks = generated.search;
  const nodeCounts = generated.nodeCounts.join(", ");
  const promptTokens = representativePromptTokens(config);
  const decodeTokens = representativeDecodeTokens(config);
  return `schema_version = 1

[model]
id = "${config.modelId}"
layers = ${config.layers}
hidden_size = ${config.hiddenSize}
attention_heads = ${config.attentionHeads}
kv_heads = ${config.kvHeads}
vocab_size = ${config.vocabSize}
parameters_gb = ${estimatedModelWeightGb(config).toFixed(3)}
dtype = "${config.dtype}"

[request]
batch_size = 1
prompt_tokens = ${promptTokens}
decode_tokens = ${decodeTokens}
max_sequence_tokens = ${config.maxSequenceTokens}
phase = "end_to_end"

[calibration]
compute_efficiency = 0.35
prefill_compute_scale = 1.0
decode_compute_scale = 1.1
decode_memory_bandwidth_scale = 0.85
collective_bandwidth_scale = 0.85
kv_transfer_scale = 1.0
scheduler_overhead_us = 20.0
serving_pipeline_depth = 4
allow_compute_comm_overlap = true

[search]
tensor_ranks = [${ranks.tp.join(", ")}]
pipeline_ranks = [${ranks.pp.join(", ")}]
expert_ranks = [${ranks.ep.join(", ")}]
data_ranks = [${ranks.dp.join(", ")}]

[serving]
mode = "${config.servingMode}"
objective = "${config.targetMetric}"
require_routable_pools = true
${servingMetricCeilingsToml(config)}

[serving.pool_search]
prefill_groups = ["all"]
decode_groups = ["all"]
prefill_node_counts = [${nodeCounts}]
decode_node_counts = [${nodeCounts}]
allow_overlap = true
max_candidates = ${config.maxPoolCandidates}

[serving.traffic]
request_count = ${config.requestCount}
${trafficArrivalToml(config)}
routing_policy = "topology_aware"
shape_seed = 7
prefill_batching = "continuous"
decode_batching = "continuous"
decode_capacity_policy = "request_reject"
max_prefill_batch_tokens = ${config.maxPrefillBatchTokens}
max_prefill_chunk_tokens = ${config.maxPrefillChunkTokens}
max_decode_batch_tokens = ${config.maxDecodeBatchTokens}
max_decode_sequences = ${config.maxDecodeSequences}
max_resident_tokens = ${config.maxResidentTokens}
kv_block_tokens = ${config.kvBlockTokens}
max_kv_blocks = ${config.maxKvBlocks}
ttft_slo_ms = 500.0
tpot_slo_ms = 100.0
itl_slo_ms = 100.0
e2el_slo_ms = 5000.0
max_queue_delay_ms = 500.0
max_kv_queue_delay_ms = 500.0
max_decode_queue_delay_ms = 500.0
max_decode_iteration_queue_delay_ms = 500.0
request_timeout_ms = 10000.0

[serving.traffic.batch_size_distribution]
kind = "weighted"
values = [1, 2, 4]
weights = [0.70, 0.20, 0.10]

[serving.traffic.prompt_tokens_distribution]
kind = "lognormal"
median = ${config.promptMedianTokens.toFixed(1)}
sigma = ${config.promptSigma.toFixed(2)}
min = ${config.promptMinTokens}
max = ${config.promptMaxTokens}

[serving.traffic.decode_tokens_distribution]
kind = "uniform"
min = ${config.decodeMinTokens}
max = ${config.decodeMaxTokens}

[serving.prefill_search]
tensor_ranks = [${ranks.tp.join(", ")}]
pipeline_ranks = [${ranks.pp.join(", ")}]
expert_ranks = [${ranks.ep.join(", ")}]
data_ranks = [${ranks.dp.join(", ")}]

[serving.decode_search]
tensor_ranks = [${ranks.tp.join(", ")}]
pipeline_ranks = [${ranks.pp.join(", ")}]
expert_ranks = [${ranks.ep.join(", ")}]
data_ranks = [${ranks.dp.join(", ")}]
`;
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function addRect(svg, x, y, width, height, className) {
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("width", width);
  rect.setAttribute("height", height);
  rect.setAttribute("rx", 8);
  rect.setAttribute("class", className);
  svg.append(rect);
}

function addCircle(svg, cx, cy, r, className) {
  const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  circle.setAttribute("cx", cx);
  circle.setAttribute("cy", cy);
  circle.setAttribute("r", r);
  circle.setAttribute("class", className);
  svg.append(circle);
}

function addLine(svg, x1, y1, x2, y2, className) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("class", className);
  svg.append(line);
}

function addText(svg, x, y, text, className, anchor = "start") {
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.setAttribute("x", x);
  label.setAttribute("y", y);
  label.setAttribute("class", className);
  label.setAttribute("text-anchor", anchor);
  label.textContent = text;
  svg.append(label);
}

function addRotatedText(svg, x, y, text, className, degrees) {
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.setAttribute("x", x);
  label.setAttribute("y", y);
  label.setAttribute("class", className);
  label.setAttribute("text-anchor", "middle");
  label.setAttribute("transform", `rotate(${degrees} ${x} ${y})`);
  label.textContent = text;
  svg.append(label);
}

function candidateMetricValue(candidate, metric) {
  const metrics = candidate.metrics ?? {};
  if (metric === "memory_pressure_peak_fraction") {
    return Number(candidate.memory_pressure_peak_fraction ?? candidate.max_memory_pressure_fraction);
  }
  if (metric === "slo_miss_rate") {
    return Number(
      metrics.deadline_miss_rate ??
        metrics.e2el_slo_miss_rate ??
        metrics.ttft_slo_miss_rate ??
        metrics.tpot_slo_miss_rate ??
        metrics.itl_slo_miss_rate,
    );
  }
  return Number(metrics[metric]);
}

function paddedScale(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    const pad = Math.max(1, Math.abs(min) * 0.1);
    return { min: min - pad, max: max + pad, span: pad * 2 };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad, span: max - min + pad * 2 };
}

function scatterTicks(scale) {
  return Array.from({ length: 5 }, (_, index) => scale.min + scale.span * (index / 4));
}

function formatMs(value) {
  return `${Number(value).toFixed(value >= 100 ? 0 : 1)} ms`;
}

function formatMaybeMs(value) {
  return Number.isFinite(Number(value)) ? formatMs(Number(value)) : "n/a";
}

function formatMaybeNumber(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(Number(value) >= 100 ? 0 : 1) : "n/a";
}

function formatMaybePct(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : "n/a";
}

function formatCompactMs(value) {
  if (!Number.isFinite(Number(value))) return "n/a";
  const numeric = Number(value);
  return `${numeric.toFixed(Math.abs(numeric) >= 100 ? 0 : 1)} ms`;
}

function formatCompactNumber(value) {
  if (!Number.isFinite(Number(value))) return "n/a";
  const numeric = Number(value);
  if (Math.abs(numeric) >= 1000) return numeric.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return numeric.toFixed(Math.abs(numeric) >= 100 ? 0 : 1);
}

function formatCompactPct(value) {
  if (!Number.isFinite(Number(value))) return "n/a";
  const numeric = Number(value);
  return `${(numeric * 100).toFixed(Math.abs(numeric) < 0.1 ? 1 : 0)}%`;
}

function objectiveLabel(objective) {
  const labels = {
    e2el: "E2EL",
    ttft: "TTFT",
    tpot: "TPOT",
    throughput: "Throughput",
    memory_pressure: "Memory pressure",
    slo_miss_rate: "SLO miss rate",
  };
  return labels[objective] ?? objective;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

function setRunStatus(kind, message) {
  const status = document.querySelector("#resultStatus");
  status.className = `result-status ${kind}`;
  status.textContent = message;
}

async function runSimulation() {
  const button = document.querySelector("#runSimulationButton");
  const config = readConfig();
  state.running = true;
  button.disabled = true;
  button.textContent = "Running...";
  setRunStatus("", "Running simulator through the local GUI server...");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        clusterToml: buildClusterToml(config),
        workloadToml: buildWorkloadToml(config),
        requestLimit: 2,
        topK: config.topK,
        maxRankCandidates: config.maxRankCandidates,
        maxServingPairs: maxServingPairs(config),
        maxSearchRuntimeMs: maxSearchRuntimeMs(config),
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      const details = [payload.error, payload.stderr, payload.stdout].filter(Boolean).join("\n").trim();
      throw new Error(details || `server returned ${response.status}`);
    }

    state.result = payload.result;
    jsonInput.value = JSON.stringify(payload.result, null, 2);
    setActiveTab("process");
    state.running = false;
    render();
  } catch (error) {
    setRunStatus("error", `Run failed: ${error.message}`);
  } finally {
    state.running = false;
    button.disabled = false;
    button.textContent = "Run simulation";
  }
}

function setActiveTab(tabName) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabName));
  document.querySelectorAll(".tab-page").forEach((page) => page.classList.toggle("active", page.id === tabName));
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.tab);
  });
});

scatterXMetricSelect.addEventListener("change", () => {
  state.scatterXMetric = scatterXMetricSelect.value;
  render();
});

scatterYMetricSelect.addEventListener("change", () => {
  state.scatterYMetric = scatterYMetricSelect.value;
  render();
});

form.addEventListener("input", () => {
  state.result = null;
  jsonInput.value = "";
  render();
});

jsonInput.addEventListener("input", () => {
  const value = jsonInput.value.trim();
  if (!value) {
    state.result = null;
    render();
    return;
  }
  try {
    state.result = JSON.parse(value);
    render();
  } catch (error) {
    const status = document.querySelector("#resultStatus");
    status.className = "result-status error";
    status.textContent = `JSON parse error: ${error.message}`;
  }
});

document.querySelector("#loadDemoButton").addEventListener("click", () => {
  state.result = DEMO_RESULT;
  jsonInput.value = JSON.stringify(DEMO_RESULT, null, 2);
  render();
});

document.querySelector("#runSimulationButton").addEventListener("click", runSimulation);

document.querySelector("#downloadClusterButton").addEventListener("click", () => {
  downloadText("cluster.toml", buildClusterToml(readConfig()));
});

document.querySelector("#downloadWorkloadButton").addEventListener("click", () => {
  downloadText("workload.toml", buildWorkloadToml(readConfig()));
});

render();
