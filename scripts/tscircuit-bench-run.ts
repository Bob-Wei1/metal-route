// Standalone tscircuit benchmark runner.
//
// Bypasses the harness `cli.ts` (which imports the dev-server vfs build that is
// not present in a fresh clone). Drives `runBenchmark` directly against an HTTP
// solver. Copied into the cloned harness dir by `scripts/bench-tscircuit.sh`.
//
// Env: SOLVER_URL, PROBLEM_TYPE (single category or "all"), SAMPLE_COUNT,
// SAMPLE_SEED.
import { runBenchmark } from "./module/lib/benchmark/run-benchmark"
import { createSolverFromUrl } from "./module/lib/solver-utils/createSolverFromUrl"

const solverUrl = process.env.SOLVER_URL ?? "http://localhost:1234/solve"
const problemType = (process.env.PROBLEM_TYPE ?? "all") as any
const sampleCount = parseInt(process.env.SAMPLE_COUNT ?? "20")
const sampleSeed = parseInt(process.env.SAMPLE_SEED ?? "0")

const results = await runBenchmark({
  solver: createSolverFromUrl(solverUrl),
  solverName: solverUrl,
  verbose: true,
  sampleCount,
  problemType,
  sampleSeed,
  noSkipping: true,
})

// Machine-readable lines for the wrapper script to scrape.
for (const r of results) {
  const pct = ((r.successfulSamples / r.samplesRun) * 100).toFixed(1)
  console.log(
    `RESULT ${r.problemType} ${r.successfulSamples}/${r.samplesRun} ${pct}% avg=${r.averageTime.toFixed(1)}ms`,
  )
}
