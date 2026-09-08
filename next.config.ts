import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Temporary cloudflared quick-tunnel domain, used to let a real phone
  // (not on the same LAN as this dev sandbox) reach the dev server to test
  // the scan flow. Next.js dev blocks cross-origin access to dev-only
  // resources (HMR client, etc.) by default — without this, the page's own
  // JS never finishes loading through the tunnel, so every button appears
  // completely inert (looks fine, does nothing) with no visible error.
  allowedDevOrigins: [
    "*.trycloudflare.com",
    "claimed-buffer-ipod-thickness.trycloudflare.com",
    "absolute-real-phys-motor.trycloudflare.com",
    "attitudes-connection-franchise-piece.trycloudflare.com",
  ],
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
  outputFileTracingExcludes: {
    "*": [
      "./ai-engine/**",
      "./scratchpad/**",
      "./HRN/**",
    ],
  },
};

export default nextConfig;

