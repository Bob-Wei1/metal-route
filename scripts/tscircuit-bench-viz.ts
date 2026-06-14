// tscircuit benchmark VISUALIZER.
//
// For each category + seed it generates the problem, solves it against the HTTP
// solver (the same `mr-server /solve` the benchmark drives), runs the harness
// checks for an accurate pass/fail, renders the combined problem+solution to a
// real PCB SVG (`circuit-to-svg`), and writes one SVG per board plus a
// self-contained `index.html` gallery. Copied into the cloned harness dir by
// `scripts/bench-tscircuit.sh` (so the `./module/...` imports resolve).
//
// Env:
//   SOLVER_URL       solver endpoint (default http://localhost:1234/solve)
//   VIZ_OUT_DIR      output directory (required)
//   VIZ_CATEGORIES   csv of categories, or "all" (default all four)
//   VIZ_SAMPLES      samples per category (default 6)
//   VIZ_SEED         first seed (default 0)
//   MR_SOLVE_LAYERS  informational, shown in the gallery header
import { mkdirSync, writeFileSync } from "fs"
import { join } from "path"
import { getDatasetGenerator } from "./module/lib/generators"
import { getSimpleRouteJson } from "./module/lib/solver-utils/getSimpleRouteJson"
import { runChecks } from "./module/lib/benchmark/run-checks"
import { convertCircuitJsonToPcbSvg } from "circuit-to-svg"

const SOLVER_URL = process.env.SOLVER_URL ?? "http://localhost:1234/solve"
const OUT_DIR = process.env.VIZ_OUT_DIR
if (!OUT_DIR) {
  console.error("VIZ_OUT_DIR is required")
  process.exit(2)
}
const ALL = ["single-trace", "distant-single-trace", "traces", "keyboards"]
const CATEGORIES =
  !process.env.VIZ_CATEGORIES || process.env.VIZ_CATEGORIES === "all"
    ? ALL
    : process.env.VIZ_CATEGORIES.split(",").map((s) => s.trim()).filter(Boolean)
const SAMPLES = parseInt(process.env.VIZ_SAMPLES ?? "6")
const SEED0 = parseInt(process.env.VIZ_SEED ?? "0")
const SOLVE_LAYERS = process.env.MR_SOLVE_LAYERS ?? "2"

mkdirSync(OUT_DIR, { recursive: true })

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")

async function solve(problemSoup: any[], srj: any): Promise<any[]> {
  const resp = await fetch(SOLVER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ problem_soup: problemSoup, simple_route_json: srj }),
  })
  if (!resp.ok) throw new Error(`solver HTTP ${resp.status}`)
  const data = await resp.json()
  return data.solution_soup ?? []
}

interface Card {
  category: string
  seed: number
  file: string
  pass: boolean
  errors: number
  nets: number
  traces: number
  ms: number
  note?: string
}

const cards: Card[] = []
// SVG markup per card, inlined into index.html so the gallery is self-contained.
const svgs = new Map<string, string>()

for (const category of CATEGORIES) {
  const gen = getDatasetGenerator(category as any)
  for (let i = 0; i < SAMPLES; i++) {
    const seed = SEED0 + i
    const file = `${category}_seed${seed}.svg`
    const card: Card = {
      category,
      seed,
      file,
      pass: false,
      errors: -1,
      nets: 0,
      traces: 0,
      ms: 0,
    }
    const addNote = (s: string) => {
      card.note = card.note ? `${card.note}; ${s}` : s
    }
    try {
      const problemSoup = await gen.getExample({ seed })
      const srj = await getSimpleRouteJson(problemSoup)
      card.nets = srj.connections?.length ?? 0
      const t0 = performance.now()
      const solutionSoup = await solve(problemSoup, srj)
      card.ms = performance.now() - t0
      card.traces = solutionSoup.filter(
        (e: any) => e?.type === "pcb_trace",
      ).length
      // Checks and render are independent: a board that breaks the checker (which
      // counts as a benchmark FAIL) is still rendered so we can see what we produced.
      try {
        const errs = runChecks(problemSoup, solutionSoup)
        card.errors = errs.length
        card.pass = errs.length === 0
      } catch (e: any) {
        addNote(`checks: ${e?.message ?? e}`)
      }
      let svg: string
      try {
        svg = convertCircuitJsonToPcbSvg([
          ...problemSoup,
          ...solutionSoup,
        ] as any)
      } catch (e: any) {
        addNote(`render: ${e?.message ?? e}`)
        svg = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="120"><text x="10" y="60" fill="red">render failed</text></svg>`
      }
      writeFileSync(join(OUT_DIR, file), svg)
      svgs.set(file, svg)
    } catch (e: any) {
      // Problem generation / solve itself failed — nothing to render.
      addNote(String(e?.message ?? e))
      const ph = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="120"><text x="10" y="60" fill="red">solve failed: ${esc(card.note ?? "")}</text></svg>`
      writeFileSync(join(OUT_DIR, file), ph)
      svgs.set(file, ph)
    }
    cards.push(card)
    console.log(
      `VIZ ${category} seed${seed} ${card.pass ? "PASS" : "FAIL"} traces=${card.traces}/${card.nets} errs=${card.errors} ${card.ms.toFixed(0)}ms`,
    )
  }
}

// ---- gallery ----
const total = cards.length
const passed = cards.filter((c) => c.pass).length
const byCat = (cat: string) => cards.filter((c) => c.category === cat)

const cardHtml = (c: Card) => `
  <figure class="card ${c.pass ? "pass" : "fail"}">
    <a href="${c.file}" target="_blank" class="svgbox">${svgs.get(c.file) ?? ""}</a>
    <figcaption>
      <span class="badge">${c.pass ? "PASS" : "FAIL"}</span>
      <b>${esc(c.category)}</b> seed ${c.seed}
      <small>traces ${c.traces}/${c.nets} · ${c.errors >= 0 ? `${c.errors} drc` : "err"} · ${c.ms.toFixed(0)}ms${c.note ? ` · ${esc(c.note)}` : ""}</small>
    </figcaption>
  </figure>`

const sections = CATEGORIES.map((cat) => {
  const cs = byCat(cat)
  const p = cs.filter((c) => c.pass).length
  return `<section><h2>${esc(cat)} <small>${p}/${cs.length} pass</small></h2>
  <div class="grid">${cs.map(cardHtml).join("")}</div></section>`
}).join("\n")

const html = `<!doctype html><meta charset="utf-8">
<title>tscircuit bench — ${total} boards</title>
<style>
  body{font:14px/1.4 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
  header{padding:16px 24px;background:#161922;position:sticky;top:0;border-bottom:1px solid #2a2f3a}
  h1{font-size:18px;margin:0 0 4px} h2{font-size:15px;margin:24px 24px 8px;font-weight:600}
  small{color:#9aa4b2;font-weight:400}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px;padding:0 24px}
  .card{margin:0;background:#1b1f29;border:1px solid #2a2f3a;border-radius:8px;overflow:hidden;border-left:4px solid #444}
  .card.pass{border-left-color:#3fb950} .card.fail{border-left-color:#f85149}
  .svgbox{display:block;height:200px;background:#0a0c10;overflow:hidden}
  .svgbox svg{width:100%;height:100%;object-fit:contain;display:block}
  figcaption{padding:8px 10px;display:flex;flex-direction:column;gap:2px}
  .badge{font-size:11px;font-weight:700;letter-spacing:.04em}
  .pass .badge{color:#3fb950} .fail .badge{color:#f85149}
</style>
<header>
  <h1>tscircuit autorouting benchmark — ${passed}/${total} boards pass</h1>
  <small>solve layers ≥ ${esc(SOLVE_LAYERS)} · ${SAMPLES} samples/category · solver ${esc(SOLVER_URL)}</small>
</header>
${sections}`

writeFileSync(join(OUT_DIR, "index.html"), html)
writeFileSync(
  join(OUT_DIR, "summary.json"),
  JSON.stringify({ solveLayers: SOLVE_LAYERS, total, passed, cards }, null, 2),
)
console.log(`VIZ wrote ${total} boards (${passed} pass) -> ${OUT_DIR}/index.html`)
