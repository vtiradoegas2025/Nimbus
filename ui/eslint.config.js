import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import reactHooks from "eslint-plugin-react-hooks";

/*
 * A deliberately narrow lint gate.
 *
 * Why it exists
 * -------------
 * PlotStudio.tsx is ~3,400 lines with a `if (!dataset || !runDir) return ...`
 * partway down its body. Three hooks were once added below that return, so
 * opening a dataset changed the hook count between renders and React threw
 * #310 -- which does not degrade one panel, it unmounts the entire Plots view.
 * A comment in the file recorded the invariant; the comment did not enforce it.
 * `rules-of-hooks` decides this statically and is the whole reason this config
 * exists.
 *
 * Why it is not a style overhaul
 * ------------------------------
 * A 50k-line codebase that has never had a linter will light up under any
 * `recommended` set, and a gate that reports thousands of findings on day one
 * is a gate everyone learns to skip. Only rules that catch defects are errors
 * here. Formatting is not linted at all.
 *
 * `exhaustive-deps` is a warning rather than an error on purpose: it is right
 * often enough to be worth reading and wrong often enough -- refs, stable
 * setters, deliberately-stale closures -- that failing a build on it would
 * mean silencing it with comments, which trains people to silence the useful
 * one next to it. Warnings do not affect the exit code, so CI stays green
 * while the findings stay visible.
 *
 * No type-aware rules (no `parserOptions.project`), so a full lint is seconds
 * rather than a second type-check; `tsc -b --noEmit` already owns types.
 */

export default [
  {
    ignores: [
      "ui/build/**",
      "build/**",
      "ui/dist/**",
      "**/node_modules/**",
      // Rust, and generated Tauri scaffolding.
      "ui/src-tauri/**",
      // Pyodide, copied in for the browser tests. Third-party and minified:
      // linting it is slow and every finding is someone else's.
      "ui/build/test-public/**",
    ],
  },
  {
    // Test scripts and the integration harness are included: the harness mounts
    // real components, so it can break the same rule the app can.
    files: [
      "ui/src/**/*.{ts,tsx}",
      "tests/ui/**/*.{ts,tsx}",
      "ui/*.config.{ts,js}",
    ],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: { jsx: true },
      },
    },
    /*
     * typescript-eslint is registered with none of its rules enabled. Two
     * modules that wrap Pyodide's untyped API carry
     * `/* eslint-disable @typescript-eslint/no-explicit-any *​/`, and eslint
     * treats a directive naming an unknown rule as an ERROR -- so without the
     * plugin present, this gate would fail on comments that predate it and say
     * nothing about hooks.
     */
    plugins: { "react-hooks": reactHooks, "@typescript-eslint": tsPlugin },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
    linterOptions: {
      /*
       * This config deliberately runs a narrow subset of what a developer's
       * editor runs, so it is not in a position to declare a directive
       * pointless: "no problem was reported" here usually means "that rule is
       * not enabled here", and acting on the advice would break linting
       * somewhere this config cannot see.
       */
      reportUnusedDisableDirectives: "off",
    },
  },
];
