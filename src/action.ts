// attribution-persona-ngo — attribution persona sprite.
// Triggered by a new FishKillEvent in chesapeake-attribution. Queries upstream
// observations + releases + reports for context, asks an LLM (OpenRouter) to
// produce per-cause attribution beliefs with Subjective Logic (SL) opinions,
// and commits one AttributionBelief assertion per causal factor.

import type { AddOperation, Operation, FilterResult, ThingGet } from "@warmhub/sdk-ts";
import { clientFromEnv, homeRepo, splitRepo } from "./warmhub";

// ── PERSONA CONFIG (varies per sprite) ─────────────────────────────────────
const PERSONA = "ngo" as "ngo" | "industry" | "agency";
const PROMPT_VERSION = "v1-2026-04-23-ngo";
const PERSONA_PRIORS = `Your priors favor causes linked to industrial pollution, agricultural runoff, stormwater discharge, and regulatory failures. You weigh evidence of EPA TRI releases, nutrient loading (nitrate/phosphorus), and recent pollution events heavily. You are skeptical of "natural causes" explanations when upstream industrial activity or agricultural runoff is present in the evidence window, though you acknowledge natural causes when evidence overwhelmingly supports them.`;
// ───────────────────────────────────────────────────────────────────────────

const MODEL = process.env.PERSONA_MODEL ?? "anthropic/claude-3.5-sonnet";
const OPENROUTER_BASE = process.env.OPENROUTER_BASE ?? "https://openrouter.ai/api/v1";
const CONTEXT_LIMIT = 40;
const RADIUS_DEG = 0.75;   // ~80 km around the event
const DAYS_BEFORE = 14;

const CAUSES = ["agricultural","thermal","industrial","stormflow","biological","unknown"] as const;
type Cause = (typeof CAUSES)[number];

const UPSTREAM_REPOS = {
  noaa: "fish-kill-attribution/noaa-sst-daily",
  usgs: "fish-kill-attribution/usgs-nwis",
  epa:  "fish-kill-attribution/epa-tri",
  reports: "fish-kill-attribution/state-fishkills",
} as const;

interface MatchedOp { name?: string; kind?: string; operation?: string }
interface SpritePayload { matchedOperations?: MatchedOp[] }

interface EventData {
  lat: number; lon: number; date: string; status: string;
  watershed: string; location_name: string;
  source_report?: string;
  primary_species?: string | null;
  estimated_mortality?: number;
}

interface ContextItem { wref: string; summary: string }

interface LlmBelief {
  cause: Cause;
  share: number;
  rationale: string;
  sl_belief: number;
  sl_disbelief: number;
  sl_uncertainty: number;
  sl_base_rate?: number;
  evidence_ids?: string[];
}

interface LlmResponse { beliefs: LlmBelief[] }

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function normalizeSl(b: LlmBelief): LlmBelief {
  const belief = clamp01(b.sl_belief);
  const disbelief = clamp01(b.sl_disbelief);
  const uncertainty = clamp01(b.sl_uncertainty);
  const sum = belief + disbelief + uncertainty;
  if (sum === 0) return { ...b, sl_belief: 0, sl_disbelief: 0, sl_uncertainty: 1 };
  return {
    ...b,
    sl_belief: belief / sum,
    sl_disbelief: disbelief / sum,
    sl_uncertainty: uncertainty / sum,
    sl_base_rate: clamp01(b.sl_base_rate ?? 0.5),
    share: clamp01(b.share),
  };
}

function toDateMs(iso: string | undefined): number {
  if (!iso) return 0;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : 0;
}

function haversineDegDistance(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  // Rough degree-distance; plenty for "is it near" filtering.
  return Math.sqrt((a.lat - b.lat) ** 2 + (a.lon - b.lon) ** 2);
}

async function queryNearby(
  client: ReturnType<typeof clientFromEnv>,
  repo: string,
  shape: string,
  event: EventData,
  limit = CONTEXT_LIMIT,
): Promise<ContextItem[]> {
  const { orgName, repoName } = splitRepo(repo);
  let result: FilterResult;
  try {
    result = await client.thing.query(orgName, repoName, { shape, limit: limit * 3 });
  } catch (err) {
    console.error(`query ${repo}/${shape} failed:`, (err as Error).message);
    return [];
  }

  const eventMs = toDateMs(event.date);
  const windowMs = DAYS_BEFORE * 86_400_000;
  const picks: ContextItem[] = [];

  for (const row of result.items ?? []) {
    const data = (row as any).data ?? (row as any).head?.data ?? {};
    const lat = Number(data.lat);
    const lon = Number(data.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      if (haversineDegDistance({ lat, lon }, event) > RADIUS_DEG) continue;
    }
    // Date filter: skip items with timestamps outside the window if we have a timestamp.
    const ts = data.timestamp ?? data.date ?? null;
    if (ts) {
      const tMs = toDateMs(ts);
      if (tMs && eventMs && (tMs > eventMs || eventMs - tMs > windowMs)) continue;
    }
    const wref = `wh:${repo}/${shape}/${row.name}`;
    const summary = JSON.stringify({
      name: row.name,
      ...Object.fromEntries(Object.entries(data).slice(0, 10)),
    });
    picks.push({ wref, summary });
    if (picks.length >= limit) break;
  }
  return picks;
}

async function gatherContext(
  client: ReturnType<typeof clientFromEnv>,
  event: EventData,
): Promise<ContextItem[]> {
  const [obsNoaa, obsUsgs, releases, reports] = await Promise.all([
    queryNearby(client, UPSTREAM_REPOS.noaa, "Observation", event),
    queryNearby(client, UPSTREAM_REPOS.usgs, "Observation", event),
    queryNearby(client, UPSTREAM_REPOS.epa, "Release", event),
    queryNearby(client, UPSTREAM_REPOS.reports, "FishKillReport", event),
  ]);
  return [...obsNoaa, ...obsUsgs, ...releases, ...reports];
}

const SYSTEM_PROMPT = `You are an attribution analyst with ${PERSONA} priors.
${PERSONA_PRIORS}

Given a fish-kill event and a list of nearby upstream observations, releases,
and prior reports, produce causal attribution beliefs across these causes:
${CAUSES.join(", ")}.

For each cause you assign non-zero share, produce a Subjective Logic opinion
(sl_belief + sl_disbelief + sl_uncertainty = 1.0) and cite concrete evidence
wrefs from the provided context.

Output STRICT JSON matching:
{
  "beliefs": [
    {
      "cause": "<one of: ${CAUSES.join(" | ")}>",
      "share": <0..1, shares across beliefs should sum to ~1.0>,
      "rationale": "<2-4 sentences of reasoning>",
      "sl_belief": <0..1>,
      "sl_disbelief": <0..1>,
      "sl_uncertainty": <0..1>,
      "sl_base_rate": <0..1, your prior rate for this cause>,
      "evidence_ids": ["<wref>", "..."]
    }
  ]
}

Only include causes whose share > 0. Shares should sum to approximately 1.0.
Be honest about uncertainty — high uncertainty is acceptable when evidence is
sparse. Evidence_ids MUST be wrefs drawn from the provided context; do not
invent wrefs.`;

async function askLlm(event: EventData, context: ContextItem[]): Promise<LlmResponse> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY not set (credential binding?)");

  const userMessage = JSON.stringify({
    event,
    context: context.map((c) => ({ wref: c.wref, data: c.summary })),
  }, null, 2);

  const resp = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/ntoft/attribution-persona-ngo",
      "X-Title": "fish-kill-attribution-ngo",
    },
    body: JSON.stringify({
      model: MODEL,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user",   content: userMessage },
      ],
    }),
  });
  if (!resp.ok) {
    throw new Error(`OpenRouter ${resp.status}: ${await resp.text()}`);
  }
  const res = (await resp.json()) as { choices?: Array<{ message?: { content?: string } }> };

  const raw = res.choices?.[0]?.message?.content ?? "{}";
  try {
    const parsed = JSON.parse(raw) as LlmResponse;
    if (!Array.isArray(parsed.beliefs)) throw new Error("no beliefs[] in response");
    return parsed;
  } catch (err) {
    throw new Error(`LLM returned unparseable JSON: ${(err as Error).message}\n--- raw: ${raw.slice(0, 500)}`);
  }
}

function extractEventWref(payload: SpritePayload): string | null {
  const ops = payload.matchedOperations ?? [];
  for (const op of ops) {
    // Sprite runtime nests the commit op under .operation; support both shapes.
    const nm = (op as any)?.operation?.name ?? (op as any)?.name;
    if (typeof nm === "string" && nm.startsWith("FishKillEvent/")) return nm;
  }
  return null;
}

function slug(s: string): string {
  return s.replace(/[^a-zA-Z0-9-]/g, "-").toLowerCase();
}

async function main() {
  const client = clientFromEnv();
  const { orgName, repoName } = splitRepo(homeRepo()); // chesapeake-attribution

  const raw = await Bun.stdin.text();
  const payload: SpritePayload = raw ? JSON.parse(raw) : {};
  const eventWref = extractEventWref(payload);
  if (!eventWref) {
    console.log(JSON.stringify({ skipped: true, reason: "no FishKillEvent in payload" }));
    return;
  }

  const eventThing = await client.thing.get(orgName, repoName, eventWref) as ThingGet;
  const eventData = ((eventThing as any).head?.data ?? (eventThing as any).data ?? {}) as EventData;
  if (!eventData?.date || eventData.lat == null || eventData.lon == null) {
    console.log(JSON.stringify({ skipped: true, reason: "event missing required fields", eventWref }));
    return;
  }

  const context = await gatherContext(client, eventData);
  const llm = await askLlm(eventData, context);

  const beliefs = llm.beliefs
    .filter((b) => CAUSES.includes(b.cause))
    .map(normalizeSl)
    .filter((b) => b.share > 0);

  if (beliefs.length === 0) {
    console.log(JSON.stringify({ eventWref, beliefsEmitted: 0, note: "LLM returned no viable beliefs" }));
    return;
  }

  const eventSlug = slug(eventWref.replace("FishKillEvent/", ""));
  const ops: AddOperation[] = beliefs.map((b) => ({
    operation: "add",
    kind: "assertion",
    name: `AttributionBelief/${eventSlug}-${PERSONA}-${b.cause}`,
    about: eventWref,
    data: {
      cause: b.cause,
      share: b.share,
      persona: PERSONA,
      model: MODEL,
      prompt_version: PROMPT_VERSION,
      rationale: b.rationale,
      sl_belief: b.sl_belief,
      sl_disbelief: b.sl_disbelief,
      sl_uncertainty: b.sl_uncertainty,
      sl_base_rate: b.sl_base_rate ?? 0.5,
      evidence_ids: b.evidence_ids ?? [],
    },
    skipExisting: false,
  }));

  const result = await client.commit.apply(
    orgName, repoName,
    `${PERSONA} attribution for ${eventWref}`,
    ops as Operation[],
  );

  console.log(JSON.stringify({
    eventWref,
    persona: PERSONA,
    model: MODEL,
    commitId: result.commitId,
    beliefsEmitted: beliefs.length,
    causes: beliefs.map((b) => b.cause),
    contextSize: context.length,
  }));
}

main().catch((err) => { console.error(err); process.exit(1); });
