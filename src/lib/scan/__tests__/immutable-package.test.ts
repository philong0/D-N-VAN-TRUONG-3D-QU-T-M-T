import { describe, it, expect } from "vitest";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "fs/promises";
import os from "os";
import path from "path";
import { beginPackage, PackageConflict } from "../immutable-package";
import { publishNativeBaseline } from "../native-reconstruction";

function upload(value = "original") {
  const form = new FormData();
  form.set("manifest", JSON.stringify({ reconstructionFrames: ["frame.jpg"] }));
  form.set("frame.jpg", new File([value], "frame.jpg"));
  return form;
}

describe("immutable native dataset", () => {
  it("blocks competing finalizers, publishes once and retries without writing", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "scan-seal-"));
    try {
      const dest = path.join(dir, "frames");
      const first = await beginPackage(dest, upload());
      if (first.replay) throw new Error("Unexpected replay");
      await expect(beginPackage(dest, upload())).rejects.toBeInstanceOf(PackageConflict);
      await expect(readFile(path.join(dest, "frame.jpg"))).rejects.toThrow();
      await writeFile(path.join(first.stage, "frame.jpg"), "original");
      await writeFile(path.join(first.stage, "manifest.json"), "{}");
      await first.commit(); await first.close();
      expect((await beginPackage(dest, upload())).replay).toBe(true);
      await expect(beginPackage(dest, upload("replacement"))).rejects.toBeInstanceOf(PackageConflict);
      expect(await readFile(path.join(dest, "frame.jpg"), "utf8")).toBe("original");
    } finally { await rm(dir, { recursive: true, force: true }); }
  });

  it("aborts partial serialization without sealing and accepts retry", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "scan-abort-"));
    try {
      const dest = path.join(dir, "frames");
      const tx = await beginPackage(dest, upload());
      if (tx.replay) throw new Error("Unexpected replay");
      await writeFile(path.join(tx.stage, "partial.jpg"), "partial");
      await tx.close();
      const retry = await beginPackage(dest, upload());
      expect(retry.replay).toBe(false);
      if (!retry.replay) await retry.close();
    } finally { await rm(dir, { recursive: true, force: true }); }
  });

  it("protects unsealed legacy datasets", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "scan-legacy-"));
    try {
      await writeFile(path.join(dir, "manifest.json"), "legacy");
      await expect(beginPackage(dir, upload())).rejects.toBeInstanceOf(PackageConflict);
      expect(await readFile(path.join(dir, "manifest.json"), "utf8")).toBe("legacy");
    } finally { await rm(dir, { recursive: true, force: true }); }
  });

  it("never replaces the accepted pointer when a generated GLB fails QC", async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), "scan-qc-"));
    try {
      const output = path.join(dir, "rejected"); await mkdir(output);
      await writeFile(path.join(dir, "native-baseline.json"), "original accepted pointer");
      await writeFile(path.join(output, "baseline.glb"), "bad mesh");
      await writeFile(path.join(output, "baseline.json"), JSON.stringify({ reconstructionStatus: "completed" }));
      await writeFile(path.join(output, "reconstruction_report.json"), JSON.stringify({ reconstructionStatus: "reconstruction_failed", rejectionReasons: ["projection failed"] }));
      await expect(publishNativeBaseline(dir, output)).rejects.toThrow("projection failed");
      expect(await readFile(path.join(dir, "native-baseline.json"), "utf8")).toBe("original accepted pointer");
    } finally { await rm(dir, { recursive: true, force: true }); }
  });
});
