import { execFile } from "child_process";
import { promisify } from "util";
import { mkdir, readFile, writeFile, rename, rm, copyFile } from "fs/promises";
import { randomUUID } from "crypto";
import path from "path";
import { DATA_DIR, scanFramesDir, scanSessionDir } from "@/lib/storage";
import type { ReconstructionResult, ScanData } from "./reconstruction-service";

export const NATIVE_ROUTE_VERSION = "truedepth-v3-calibrated-20260917";
const run = promisify(execFile);

export async function publishNativeBaseline(patientDir: string, outputDir: string) {
  const report = JSON.parse(await readFile(path.join(outputDir, "reconstruction_report.json"), "utf8"));
  const meta = JSON.parse(await readFile(path.join(outputDir, "baseline.json"), "utf8"));
  if (report.reconstructionStatus !== "completed" || meta.reconstructionStatus !== "completed"
      || !Array.isArray(report.rejectionReasons) || report.rejectionReasons.length
      || report.independentFrameCount < 7 || report.independentFrameCount > 10
      || !(report.textureMetrics?.observedFraction >= .95)
      || !(report.textureMetrics?.nearBlackFraction <= .01)
      || !(report.registrationErrorMm <= 5)) {
    throw new Error(`Native QC rejected: ${(report.rejectionReasons ?? ["missing QC evidence"]).join("; ")}`);
  }
  // Preserve every older artifact. One atomic pointer selects a complete,
  // immutable accepted bundle; rejected runs never touch this pointer.
  const version = path.join(patientDir, "native-baselines", randomUUID());
  await mkdir(version, { recursive: true });
  for (const name of ["baseline.glb", "baseline.obj", "baseline.json", "face_HD.png", "reconstruction_report.json", "texture_report.json", "input_report.json"]) {
    await copyFile(path.join(outputDir, name), path.join(version, name));
  }
  const temp = path.join(patientDir, `native-baseline.${randomUUID()}.tmp`);
  await writeFile(temp, JSON.stringify({ directory: path.relative(patientDir, version), routeVersion: NATIVE_ROUTE_VERSION }));
  await rename(temp, path.join(patientDir, "native-baseline.json"));
}

export async function reconstructNative(data: ScanData): Promise<ReconstructionResult> {
  const provider = "Native TrueDepth calibrated fusion";
  const evaluatedAt = new Date().toISOString();
  const patientDir = path.join(DATA_DIR, "patients", data.patientId);
  const lock = path.join(patientDir, "native-reconstruction.lock");
  await mkdir(patientDir, { recursive: true });
  try { await mkdir(lock); } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") return { status: "failed", provider, evaluatedAt, reason: "Native reconstruction already running; dataset preserved." };
    throw error;
  }
  const output = path.join(scanSessionDir(data.patientId, data.sessionId), "native-results", randomUUID());
  try {
    await mkdir(output, { recursive: true });
    const engine = path.join(process.cwd(), "ai-engine");
    console.log(`[native-route:${NATIVE_ROUTE_VERSION}] patient=${data.patientId} session=${data.sessionId} output=${output}`);
    try {
      const result = await run(process.env.PYTHON_BIN || path.join(engine, "venv/bin/python3"), [
        "-B", path.join(engine, "reconstruct_native_truedepth.py"),
        "--package-dir", scanFramesDir(data.patientId, data.sessionId),
        "--patient-id", data.patientId, "--session-id", data.sessionId, "--output-dir", output,
      ], { cwd: engine, timeout: 480000, maxBuffer: 20*1024*1024,
        env: { ...process.env, OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1" } });
      await writeFile(path.join(output, "worker.log"), result.stdout + result.stderr);
    } catch (error) {
      const err = error as Error & { stdout?: string; stderr?: string };
      await writeFile(path.join(output, "worker.log"), (err.stdout ?? "") + (err.stderr ?? "") + err.message);
      let reason = "Native worker failed; inspect session native-results/worker.log";
      try { reason = JSON.parse(await readFile(path.join(output, "reconstruction_report.json"), "utf8")).rejectionReasons.join("; "); } catch {}
      return { status: "failed", provider, evaluatedAt, reason };
    }
    await publishNativeBaseline(patientDir, output);
    return { status: "completed", provider, evaluatedAt, artifacts: {
      baselineModelFileName: "baseline.glb", baselineObjFileName: "baseline.obj",
      textureFileName: "face_HD.png", landmarksCount: 1220, providerVersion: NATIVE_ROUTE_VERSION,
    } };
  } catch (error) {
    return { status: "failed", provider, evaluatedAt, reason: String(error) };
  } finally { await rm(lock, { recursive: true, force: true }); }
}
