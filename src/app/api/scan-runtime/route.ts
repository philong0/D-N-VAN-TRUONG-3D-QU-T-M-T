import { readFile } from "fs/promises";
import path from "path";
import { NATIVE_ROUTE_VERSION } from "@/lib/scan/native-reconstruction";

export const dynamic = "force-dynamic";
export async function GET() {
  const buildId = await readFile(path.join(process.cwd(), process.env.NEXT_DIST_DIR || ".next", "BUILD_ID"), "utf8").catch(() => "development");
  return Response.json({ routeVersion: NATIVE_ROUTE_VERSION, buildId: buildId.trim(),
    nativeWorker: "reconstruct_native_truedepth.py", nativeFallback: false,
    captureSchema: "3.0.0", legacyReader: true }, { headers: { "Cache-Control": "no-store" } });
}
