import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Generated model assets, Python environments and vendor viewers are
    // not application source. Scanning them made `npm run lint` appear to
    // hang for minutes before it ever reached the Next.js code.
    ".vercel/**",
    "ai-engine/**",
    "HRN/**",
    "public/models/**",
    "server/**",
    "scratchpad/**",
    "van-truong-his-crm/**",
  ]),
]);

export default eslintConfig;
