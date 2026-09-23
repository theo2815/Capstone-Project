import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypeScript from "eslint-config-next/typescript";

const READABILITY_FLOOR_MESSAGE =
  "text-[9px], text-[10px], and text-[11px] are below the Quiet Studio comfort floor (12px). " +
  "Use <Kicker> from @/components/ui/kicker (12px floor) or text-[12px] if you must hand-roll. " +
  "See QuickPitik Vault/website/notes/ui-pitfalls.md (2026-05-07 floor-bump).";

const TEXT_XS_BODY_FLOOR_MESSAGE =
  "text-xs (12px sans) is below the Quiet Studio comfort floor for body copy (14px / text-sm). " +
  "Use text-sm for informational text. Compact mono uppercase button/nav labels should follow the " +
  "responsive kicker pattern: text-[13px] min-[400px]:text-[14px] md:text-[12px]. " +
  "See QuickPitik Vault/website/notes/ui-pitfalls.md (2026-05-07 floor-bump cont. 6).";

const eslintConfig = [
  ...nextCoreWebVitals,
  ...nextTypeScript,
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "out/**",
      "build/**",
      "next-env.d.ts",
    ],
  },
  {
    files: ["src/{components,hooks,lib,store,types}/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": ["error", {
        patterns: [{
          group: ["@/app/**", "**/app/**"],
          message: "Reusable modules must not import routes. Move shared contracts into their domain or types/.",
        }],
      }],
    },
  },
  {
    rules: {
      // ponytail: Next 16 enables React Compiler rules that flag 72 existing
      // sites; migrate those behaviors in a dedicated React refactor.
      "react-hooks/preserve-manual-memoization": "off",
      "react-hooks/purity": "off",
      "react-hooks/refs": "off",
      "react-hooks/set-state-in-effect": "off",
      // Severity is "warn" during the gradual readability migration so the
      // backlog of legacy text-[10px] / text-xs usages stays visible in PR
      // review without blocking builds. Promote to "error" once the backlog
      // hits zero. See plans/alright-base-on-your-fizzy-pillow.md Phase D.
      "no-restricted-syntax": [
        "warn",
        {
          selector: "Literal[value=/text-\\[(9|10|11)px\\]/]",
          message: READABILITY_FLOOR_MESSAGE,
        },
        {
          selector: "TemplateElement[value.raw=/text-\\[(9|10|11)px\\]/]",
          message: READABILITY_FLOOR_MESSAGE,
        },
        {
          selector: "Literal[value=/\\btext-xs\\b/]",
          message: TEXT_XS_BODY_FLOOR_MESSAGE,
        },
        {
          selector: "TemplateElement[value.raw=/\\btext-xs\\b/]",
          message: TEXT_XS_BODY_FLOOR_MESSAGE,
        },
      ],
    },
  },
];

export default eslintConfig;
