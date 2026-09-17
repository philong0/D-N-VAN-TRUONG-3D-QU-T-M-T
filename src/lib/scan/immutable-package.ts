import { createHash, randomUUID } from "crypto";
import { mkdir, readFile, rename, rm, writeFile } from "fs/promises";
import path from "path";

export class PackageConflict extends Error {}

/** Cross-process lock + staged atomic directory publication. Never edit a seal. */
export async function beginPackage(destination: string, form: FormData) {
  const parent = path.dirname(destination);
  await mkdir(parent, { recursive: true });
  const lock = `${destination}.lock`;
  try { await mkdir(lock); } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "EEXIST") throw new PackageConflict("Package đang được finalization; hãy thử lại cùng package.");
    throw e;
  }
  const stage = `${destination}.staging-${randomUUID()}`;
  let published = false;
  try {
    const hash = createHash("sha256");
    const entries = [...form.entries()].sort(([a], [b]) => a.localeCompare(b));
    for (const [key, value] of entries) {
      const bytes = typeof value === "string" ? Buffer.from(value) : Buffer.from(await value.arrayBuffer());
      hash.update(JSON.stringify([key, bytes.length]));
      hash.update(bytes);
    }
    const digest = hash.digest("hex");
    let seal: string | undefined;
    try { seal = await readFile(path.join(destination, "package.sha256"), "utf8"); } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
    }
    if (seal) {
      if (seal !== digest) throw new PackageConflict("Session đã có immutable package khác. Hãy tạo session mới.");
      await rm(lock, { recursive: true });
      return { replay: true as const };
    }
    try {
      await readFile(path.join(destination, "manifest.json"));
      throw new PackageConflict("Session cũ đã có package; không ghi đè dữ liệu đã lưu.");
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
    }
    await mkdir(stage);
    return {
      replay: false as const, stage,
      async commit() {
        await writeFile(path.join(stage, "package.sha256"), digest, { flag: "wx" });
        // Never remove a nonempty destination, even if left by an older uploader.
        await rename(stage, destination);
        published = true;
      },
      async close() {
        if (!published) await rm(stage, { recursive: true, force: true });
        await rm(lock, { recursive: true, force: true });
      },
    };
  } catch (e) {
    await rm(stage, { recursive: true, force: true });
    await rm(lock, { recursive: true, force: true });
    throw e;
  }
}

export function safePackageName(value: unknown): string {
  if (typeof value !== "string" || !/^[a-zA-Z0-9_-]+\.(jpg|jpeg|png|raw)$/.test(value)) {
    throw new Error("Tên file package không hợp lệ");
  }
  return value;
}
