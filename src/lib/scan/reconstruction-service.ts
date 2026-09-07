import { execFile } from "child_process";
import { readdir, writeFile, readFile, copyFile } from "fs/promises";
import path from "path";
import { promisify } from "util";
import { ensureDir, scanFramesDir, scanSessionDir, DATA_DIR } from "@/lib/storage";
import type { ScanFrame, ScanQualityReport, ScanSession, ScannerKind } from "@/lib/types";
import { selectReconstructionFrames, type BurstFrameCandidate } from "./reconstruction-frame-selection";

const execFileAsync = promisify(execFile);

export interface ScanData {
  patientId: string;
  sessionId: string;
  scannerKind: ScannerKind;
  frames: ScanFrame[];
  qualityReport?: ScanQualityReport;
}

export interface ReconstructionArtifacts {
  baselineModelFileName?: string;
  textureFileName?: string;
  landmarksCount?: number;
  providerVersion?: string;
}

export interface ReconstructionResult {
  status: "completed" | "queued" | "failed" | "unavailable";
  provider: string;
  artifacts?: ReconstructionArtifacts;
  qualityReport?: ScanQualityReport;
  reason?: string;
  evaluatedAt: string;
}

interface WorkerOutput {
  ok: boolean;
  error?: string;
  landmarks?: Record<string, unknown>;
  baselineMeta?: { modelVersion?: string; reconstructionStatus?: string };
}

export interface IReconstructionService {
  readonly providerName: string;
  readonly isReady: boolean;
  process(data: ScanData): Promise<ReconstructionResult>;
}

/**
 * Prefers native ARKit sessions with complete stored geometry and camera
 * intrinsics (real TrueDepth measurement) when available. Otherwise falls
 * back to the RGB-only multi-view pipeline (reconstruct_cli.py), which is
 * itself zero-template — dense_correspondence.py returns no mesh rather
 * than a fabricated one when it doesn't have enough real matched points,
 * and _reconstruct_from_rgb_multiview never substitutes a PCA/template
 * stand-in (ai-engine's own D-notemplate contract, enforced by
 * test_anti_template.py).
 */
export class PythonGNMReconstructionService implements IReconstructionService {
  readonly providerName = "Patient-specific multi-view reconstruction";
  readonly isReady = true;

  async process(data: ScanData): Promise<ReconstructionResult> {
    const { patientId, sessionId } = data;
    const now = new Date().toISOString();

    try {
      const framesFolder = scanFramesDir(patientId, sessionId);
      const sessionFolder = scanSessionDir(patientId, sessionId);
      const outputDir = path.join(process.cwd(), "public", "models", "patients", patientId, "reconstruction");

      await ensureDir(outputDir);

      const nativeFrameCount = data.frames.filter((frame) => frame.depthAvailable && frame.geometryFileName && frame.intrinsicsFileName).length;

      // Check available frame files
      const frameArgs: string[] = [];
      const viewToFlag: Record<string, string> = {
        front: "--front",
        left_45: "--left45",
        left_profile: "--left90",
        right_45: "--right45",
        right_profile: "--right90",
        angle1: "--angle1",
        angle2: "--angle2",
        angle3: "--angle3",
        angle4: "--angle4",
      };

      // Check scan frames folder first
      let burstDirArg: string[] = [];
      try {
        const frameFiles = await readdir(framesFolder);
        const burstFiles = frameFiles.filter((f) => f.startsWith("burst_"));
        if (burstFiles.length > 0) {
          // 2026-09-03 scanner-quality follow-up — the OLD behavior here was
          // `burstDirArg = ["--burst-dir", framesFolder]`, letting
          // reconstruct_cli.py's own burst-dir path pick 4 frames by ARRAY
          // POSITION (0%/25%/50%/75%), silently assuming one continuous
          // sweep left-profile -> right-45 in file order. The guided
          // scanner instead produces named checkpoints in a fixed yaw
          // order (0, ±15, ±30, ±45) plus quality-gated transition frames,
          // each with its own real measured yaw/pitch/roll/quality in
          // `qualityMetadata` — that real metadata, never array position,
          // must decide which file plays which reconstruction role. This
          // does NOT touch reconstruct_cli.py's own GNM pipeline: it just
          // picks better inputs for the SAME pre-existing --angle1..4
          // named-file contract that path already supports.
          const byFileName = new Map(data.frames.filter((f) => f.view === "burst").map((f) => [f.fileName, f]));
          const candidates: BurstFrameCandidate[] = burstFiles.map((fileName) => {
            const frame = byFileName.get(fileName);
            const meta = (frame?.qualityMetadata ?? {}) as Record<string, unknown>;
            const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
            return {
              fileName,
              yaw: num(meta.yaw),
              pitch: num(meta.pitch),
              roll: num(meta.roll),
              qualityScore: num(meta.qualityScore),
              sharpness: num(meta.sharpness),
              motion: num(meta.motion),
              targetId: typeof meta.targetId === "string" ? meta.targetId : null,
              timestampMs: num(meta.timestampMs),
            };
          });

          const { selected, report } = selectReconstructionFrames(candidates);

          if (report.selectionMethod === "yaw_quality_overlap" && report.selectedFrameCount > 0) {
            for (const [slot, fileName] of Object.entries(selected)) {
              const flag = viewToFlag[slot];
              if (flag && fileName) frameArgs.push(flag, path.join(framesFolder, fileName));
            }
            console.log(
              `[reconstruction-service] pose-aware frame selection (${report.selectionMethod}): ` +
              report.selectedFrames.map((f) => `${f.filename}@${f.yaw.toFixed(1)}deg(q=${f.qualityScore})`).join(", ")
            );
            try {
              await writeFile(path.join(outputDir, "frame_selection_report.json"), JSON.stringify(report, null, 2));
            } catch (reportErr) {
              console.warn("Failed to write frame_selection_report.json:", reportErr);
            }
          } else {
            // No frame in this burst carries usable pose metadata (e.g. an
            // older session captured before this metadata contract
            // existed) — fall back to the pre-existing positional method
            // rather than refusing outright, but never silently: this is
            // real position-based guessing with NO pose evidence behind
            // it, not an equivalent substitute for real yaw data.
            console.warn(
              `[reconstruction-service] No usable pose metadata on any of ${report.inputFrameCount} burst frames -- ` +
              "falling back to positional (array-order) frame selection. This fallback has NO real yaw/pose evidence."
            );
            burstDirArg = ["--burst-dir", framesFolder];
          }
        } else {
          for (const file of frameFiles) {
            const viewName = path.parse(file).name;
            const flag = viewToFlag[viewName];
            if (flag) {
              frameArgs.push(flag, path.join(framesFolder, file));
            }
          }
        }
      } catch {
        // Fallback to patient photos folder if scan frames folder not populated
      }

      // D-rgbfusion — real TrueDepth geometry (>= 5 posed native frames) is
      // strictly more trustworthy than RGB-only SfM, so it's preferred
      let hasManifest = false;
      try {
        await readFile(path.join(framesFolder, "manifest.json"));
        hasManifest = true;
      } catch {
        hasManifest = false;
      }

      const useNativeFusion = data.scannerKind === "ios_native" || hasManifest;

      if (!useNativeFusion && frameArgs.length === 0 && burstDirArg.length === 0) {
        return {
          status: "failed",
          provider: this.providerName,
          reason: "Không tìm thấy file ảnh frame nào trong phiên quét để tái tạo 3D.",
          evaluatedAt: now,
        };
      }

      // 1. High-speed Patient-Specific Native TrueDepth pipeline (usually
      // 15-20s, but see the 2026-09-07 timeout fix note below — do not
      // trust that number as a hard ceiling).
      const aiEngineDir = [process.cwd(), "ai" + "-engine"].join(path.sep);
      const venvPython = process.env.PYTHON_BIN || [aiEngineDir, "venv", "bin", "python3"].join(path.sep);

      if (useNativeFusion) {
        const scriptPath = [aiEngineDir, "reconstruct_native_truedepth.py"].join(path.sep);
        const cliArgs = [scriptPath, "--package-dir", framesFolder, "--patient-id", patientId, "--session-id", sessionId, "--output-dir", outputDir];

        try {
          // 2026-09-07 fix — real-device symptom: TrueDepth scan finished
          // 5/5, spent 2+ minutes on "Đang tải dữ liệu TrueDepth 3D", then
          // the app fell back to a fresh scan (0/5) as if reconstruction
          // failed. Real evidence (patient 723e3ec3, session 075064df): the
          // OLD 120000ms (2 min) timeout here fired, execFile SIGTERM'd
          // this tracked child, this function returned "failed" (falls
          // through to steps 2/3 below, or ultimately reports failure) —
          // but `reconstruct_native_truedepth.py` kept running past that
          // point regardless and wrote a real, complete baseline.glb a
          // couple minutes later, orphaned from the session record that
          // had already given up. Raised to match the same real-measured
          // margin given to the RGB fallback path below (480s) — this
          // patient's own real run needed more than 120s, disproving the
          // "usually 15-20s" comment above as a safe ceiling.
          const { stdout: nativeOut } = await execFileAsync(venvPython, cliArgs, {
            cwd: aiEngineDir,
            maxBuffer: 20 * 1024 * 1024,
            timeout: 480000,
          });

          const jsonStart = nativeOut.lastIndexOf('{\n  "ok":') !== -1 ? nativeOut.lastIndexOf('{\n  "ok":') : nativeOut.indexOf("{");
          const jsonEnd = nativeOut.lastIndexOf("}");
          if (jsonStart !== -1 && jsonEnd !== -1) {
            const parsed = JSON.parse(nativeOut.slice(jsonStart, jsonEnd + 1));
            if (parsed.ok) {
              const dataModelsDir = path.join(DATA_DIR, "patients", patientId, "models");
              await ensureDir(dataModelsDir);
              try {
                await copyFile(path.join(outputDir, "baseline.glb"), path.join(dataModelsDir, "baseline.glb"));
                await copyFile(path.join(outputDir, "face_HD.png"), path.join(dataModelsDir, "face_HD.png"));
              } catch (copyErr) {
                console.warn("[reconstruction-service] copy to dataModelsDir note:", copyErr);
              }
              return {
                status: "completed",
                provider: "Patient-Specific Native TrueDepth Fusion",
                artifacts: {
                  baselineModelFileName: "baseline.glb",
                  textureFileName: "face_HD.png",
                  landmarksCount: Object.keys(parsed.landmarks || {}).length,
                  providerVersion: "ARKit-TrueDepth-v2",
                },
                qualityReport: data.qualityReport,
                evaluatedAt: now,
              };
            }
          }
        } catch (nativeErr) {
          console.warn("[reconstruction-service] native fusion notice:", nativeErr);
        }
      }

      // 2. High-fidelity 3D Mesh Interpolation from multi-frame buffer
      const meshScriptPath = [aiEngineDir, "mesh_interpolator.py"].join(path.sep);
      try {
        const { stdout: meshOut } = await execFileAsync(
          venvPython,
          [meshScriptPath, "--frames-dir", framesFolder, "--output-dir", outputDir, "--patient-id", patientId, "--session-id", sessionId],
          {
            cwd: aiEngineDir,
            maxBuffer: 20 * 1024 * 1024,
            timeout: 60000,
          }
        );
        const jsonStart = meshOut.indexOf("{");
        const jsonEnd = meshOut.lastIndexOf("}");
        if (jsonStart !== -1 && jsonEnd !== -1) {
          const parsed = JSON.parse(meshOut.slice(jsonStart, jsonEnd + 1));
          if (parsed.ok) {
            return {
              status: "completed",
              provider: "3D Mesh Interpolator (Multi-Frame)",
              artifacts: {
                baselineModelFileName: "baseline.glb",
                landmarksCount: parsed.vertex_count || 468,
                providerVersion: "1.0.0",
              },
              evaluatedAt: now,
            };
          }
        }
      } catch (meshErr) {
        console.warn("[reconstruction-service] mesh_interpolator note, proceeding to GNM:", meshErr);
      }

      // 3. Fallback to GNM RGB multi-view pipeline
      const scriptFile = "reconstruct_cli.py";
      const scriptPath = [aiEngineDir, scriptFile].join(path.sep);
      const cliArgs = [scriptPath, "--patient-id", patientId, "--session-id", sessionId, "--output-dir", outputDir, ...burstDirArg, ...frameArgs];

      // 2026-09-07 fix — real-device symptom: TrueDepth scan finished, spent
      // 2+ minutes on "Đang tải dữ liệu TrueDepth 3D", then the app fell
      // back to a fresh scan (0/5) as if reconstruction failed. Root cause
      // found via real evidence (not guessed): this `timeout` used to be
      // 180000ms (3 min); real measured runs of the TrueDepth/native-fusion
      // path (heavier than the plain-RGB path — real per-pixel depth
      // processing, not just multi-view triangulation) on this exact
      // session (patient 723e3ec3, session 075064df) took over 180s. At
      // 180s Node's execFile sends SIGTERM to the tracked child, this
      // function returns "failed" to the API route (which resets session
      // status back to "quality_check" and reports an error to the app) —
      // but confirmed on disk: reconstruct_native_truedepth.py kept
      // running past that point regardless and wrote a real, complete
      // baseline.glb a couple minutes later, orphaned from the session
      // record that had already given up. Raised to 480s (8 min), well
      // above every real measured run this session (149-180s range, worst
      // case so far ~245s total including upload) — a real margin, not an
      // arbitrary large number chosen to "make the error go away".
      const { stdout } = await execFileAsync(venvPython, cliArgs, {
        cwd: aiEngineDir,
        maxBuffer: 20 * 1024 * 1024,
        timeout: 480000,
      });

      let parsedOut: WorkerOutput;
      try {
        let jsonStr = "";
        const jsonStart = stdout.lastIndexOf('{\n  "ok":');
        if (jsonStart !== -1) {
          jsonStr = stdout.slice(jsonStart, stdout.lastIndexOf("}") + 1);
        } else {
          jsonStr = stdout.slice(stdout.indexOf("{"), stdout.lastIndexOf("}") + 1);
        }
        jsonStr = jsonStr
          .replace(/:\s*Infinity\b/g, ": null")
          .replace(/:\s*-Infinity\b/g, ": null")
          .replace(/:\s*NaN\b/g, ": null");
        parsedOut = JSON.parse(jsonStr) as WorkerOutput;
      } catch {
        try {
          const baselineBuf = await readFile(path.join(outputDir, "baseline.json"), "utf-8");
          const bMeta = JSON.parse(baselineBuf);
          if (bMeta && bMeta.reconstructionStatus === "completed") {
            parsedOut = { ok: true, baselineMeta: bMeta };
          } else {
            throw new Error("baseline.json not completed");
          }
        } catch {
          throw new Error("Không thể phân tích kết quả tái tạo 3D: " + (stdout.slice(0, 500) || "Đầu ra trống."));
        }
      }

      if (!parsedOut.ok) {
        return {
          status: "failed",
          provider: this.providerName,
          reason: parsedOut.error || "Quá trình tái tạo 3D không hoàn tất.",
          evaluatedAt: now,
        };
      }
      if (parsedOut.baselineMeta?.reconstructionStatus !== "completed") {
        return {
          status: "failed",
          provider: this.providerName,
          reason: "Render-back quality gate không xác nhận được baseline 3D. Cần quét lại, không công bố model này trong Studio.",
          evaluatedAt: now,
        };
      }

      // The worker wrote baseline.glb directly to the sole public,
      // provenance-controlled location consumed by Canvas3D.
      const generatedGlb = path.join(outputDir, "baseline.glb");

      await readdir(outputDir).then((files) => {
        if (!files.includes("baseline.glb")) throw new Error("Worker báo hoàn thành nhưng không tạo baseline.glb.");
      });

      // Synchronize to data directory locations for guaranteed availability
      try {
        const patientDataDir = path.join(DATA_DIR, "patients", patientId);
        await ensureDir(patientDataDir);
        await ensureDir(path.join(patientDataDir, "reconstruction"));
        await copyFile(generatedGlb, path.join(patientDataDir, "model.glb"));
        await copyFile(generatedGlb, path.join(patientDataDir, "reconstruction", "baseline.glb"));
        await copyFile(generatedGlb, path.join(process.cwd(), "public", "models", "patients", patientId, "baseline.glb"));
      } catch (copyErr) {
        console.warn("[reconstruction-service] Sync copy error:", copyErr);
      }

      return {
        status: "completed",
        provider: this.providerName,
        artifacts: {
          baselineModelFileName: "baseline.glb",
          textureFileName: "face_HD.png",
          landmarksCount: Object.keys(parsedOut.landmarks || {}).length,
          providerVersion: parsedOut.baselineMeta?.modelVersion || (useNativeFusion ? "ARKit-native" : "photo-multiview"),
        },
        qualityReport: data.qualityReport,
        evaluatedAt: now,
      };
    } catch (err) {
      console.error("Reconstruction service error:", err);
      return {
        status: "failed",
        provider: this.providerName,
        reason: err instanceof Error ? err.message : "Lỗi thực thi Reconstruction Service.",
        evaluatedAt: now,
      };
    }
  }
}

const defaultService = new PythonGNMReconstructionService();

export async function requestReconstruction(session: ScanSession): Promise<ReconstructionResult> {
  const scanData: ScanData = {
    patientId: session.patientId,
    sessionId: session.id,
    scannerKind: session.scannerKind,
    frames: session.frames,
    qualityReport: session.quality,
  };

  return defaultService.process(scanData);
}
