import fs from "fs";
import path from "path";
import { promisify } from "util";
import { execFile } from "child_process";

const execFileAsync = promisify(execFile);

async function runEndToEndVerification() {
  console.log("==================================================================");
  console.log("CHECK 7 — END-TO-END PIPELINE AUTOMATED VERIFICATION");
  console.log("==================================================================");

  const patientId = "test-patient-native-ios";
  const sessionId = "test-session-native-001";
  const packageDir = path.resolve("fixtures/scans/sample_native_ios_package");
  const outputDir = path.resolve(`public/models/patients/${patientId}/reconstruction`);
  const venvPython = path.resolve("ai-engine/venv/bin/python");
  const scriptPath = path.resolve("ai-engine/reconstruct_native_truedepth.py");

  // 1. Verify Native Package Fixture
  console.log("[1/5] Verifying Native Package Structure...");
  const manifestPath = path.join(packageDir, "scan-manifest.json");
  if (!fs.existsSync(manifestPath)) {
    throw new Error("Missing scan-manifest.json in native package");
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  console.log(`  ✓ Schema: ${manifest.schemaVersion || manifest.packageVersion}, Device: ${manifest.device?.model}, Source: ${manifest.captureSource || "fixture"}`);
  console.log(`  ✓ Frames found: ${manifest.frames.length}`);

  // 2. Run Reconstruction Service CLI
  console.log("[2/5] Running TrueDepth & ARKit Reconstruction Engine...");
  const t0 = Date.now();
  const { stdout } = await execFileAsync(venvPython, [
    scriptPath,
    "--package-dir", packageDir,
    "--patient-id", patientId,
    "--session-id", sessionId,
    "--output-dir", outputDir,
  ], { cwd: path.resolve("ai-engine") });

  const durationSec = ((Date.now() - t0) / 1000).toFixed(2);
  const jsonMatch = stdout.slice(stdout.indexOf("{"), stdout.lastIndexOf("}") + 1);
  const result = JSON.parse(jsonMatch);
  if (!result.ok) {
    throw new Error(`Reconstruction failed: ${result.error}`);
  }
  console.log(`  ✓ Reconstruction finished successfully in ${durationSec}s`);
  console.log(`  ✓ Model Version: ${result.baselineMeta.modelVersion}`);
  console.log(`  ✓ Vertices: ${result.baselineMeta.vertexCount}, Triangles: ${result.baselineMeta.triangleCount}`);

  // 3. Inspect Generated Canonical baseline.glb
  console.log("[3/5] Validating Canonical baseline.glb Binary...");
  const glbPath = path.join(outputDir, "baseline.glb");
  if (!fs.existsSync(glbPath)) throw new Error("Missing baseline.glb");

  const glbBuf = fs.readFileSync(glbPath);
  const magic = glbBuf.toString("utf8", 0, 4);
  const version = glbBuf.readUInt32LE(4);
  const length = glbBuf.readUInt32LE(8);

  if (magic !== "glTF" || version !== 2 || length !== glbBuf.length) {
    throw new Error(`Invalid GLB format: magic=${magic}, version=${version}, length=${length}`);
  }
  console.log(`  ✓ Valid glTF 2.0 binary (${(length / (1024*1024)).toFixed(2)} MB)`);

  // 4. Validate Landmarks & Quality Metadata
  console.log("[4/5] Checking Landmarks and Quality Metadata...");
  const landmarksPath = path.join(outputDir, "landmarks.json");
  const qualityPath = path.join(outputDir, "quality.json");
  const baselineJsonPath = path.join(outputDir, "baseline.json");

  const landmarks = JSON.parse(fs.readFileSync(landmarksPath, "utf8"));
  const quality = JSON.parse(fs.readFileSync(qualityPath, "utf8"));
  const baseline = JSON.parse(fs.readFileSync(baselineJsonPath, "utf8"));

  console.log(`  ✓ Pronasale (Nose Tip): [${landmarks.pronasale.map(v => v.toFixed(4)).join(", ")}]`);
  console.log(`  ✓ Pogonion (Chin): [${landmarks.pogonion.map(v => v.toFixed(4)).join(", ")}]`);
  console.log(`  ✓ Registration Error: ${Number(quality.registrationErrorMm).toFixed(2)} mm`);
  console.log(`  ✓ Overall Quality: ${quality.overall}`);

  // This repository fixture is intentionally non-clinical synthetic input.
  // Its useful assertion is that the render-back gate rejects it rather than
  // letting a plausible-looking GLB be attached to a patient record.
  if (baseline.reconstructionStatus !== "reconstruction_failed" || quality.overall !== "fail") {
    throw new Error("Safety gate regression: synthetic fixture was not rejected by render-back validation.");
  }
  console.log("  ✓ Safety gate rejected the non-clinical fixture; no baseline may be published.");

  // 5. Verification Summary
  console.log("[5/5] End-to-End Pipeline Check Completed Successfully!");
  console.log("==================================================================");
  console.log("STATUS: [SAFETY GATE VERIFIED — REAL-DEVICE VALIDATION REQUIRED]");
  console.log("==================================================================");
}

runEndToEndVerification().catch(err => {
  console.error("Verification failed:", err);
  process.exit(1);
});
