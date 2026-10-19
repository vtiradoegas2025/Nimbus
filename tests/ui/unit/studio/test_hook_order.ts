/*
 * Static guard for the Rules of Hooks invariant that React only enforces at
 * runtime: a hook must never be called after an early return.
 *
 * `PlotStudio.tsx` shipped exactly this bug. Three hooks sat below the
 * `if (!dataset || !runDir) return <Welcome/>` guard, so they ran only once a
 * dataset was open. Opening one raised the hook count mid-lifecycle and React
 * threw #310 ("rendered more hooks than during the previous render"), which
 * blanked the entire Plots view behind an error boundary.
 *
 * The project has no ESLint install, and adding one (plus the React plugin and
 * a TS parser) to catch a single rule is a lot of dependency surface. The
 * TypeScript compiler is already a dependency, so this walks the AST directly.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs ../tests/ui/unit/studio/test_hook_order.ts
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const UI_ROOT = fileURLToPath(new URL("../../../../ui", import.meta.url));
const SRC_ROOT = join(UI_ROOT, "src");

/** `useState`, `useMemo`, `useVtkPreparation`, ... — the React hook convention. */
const HOOK_NAME = /^use[A-Z]/;
/** Components are PascalCase; hooks are `useX`. Both are subject to the rule. */
const COMPONENT_NAME = /^[A-Z]/;

interface Violation {
  file: string;
  hookName: string;
  hookLine: number;
  ownerName: string;
  returnLine: number;
}

// ---------- Tiny assertion API (matches test_quiver.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

// ---------- AST helpers ----------

type FunctionLike =
  | ts.FunctionDeclaration
  | ts.FunctionExpression
  | ts.ArrowFunction
  | ts.MethodDeclaration;

function isFunctionLike(node: ts.Node): node is FunctionLike {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node)
  );
}

/**
 * Walk `node`'s descendants without crossing into a nested function.
 *
 * Crossing would produce false positives in both directions: a `return` inside
 * a `useEffect` callback is not an early return of the component, and a hook
 * inside a nested component belongs to that component's own hook sequence.
 */
function walkOwnScope(node: ts.Node, visit: (child: ts.Node) => void): void {
  node.forEachChild((child) => {
    if (isFunctionLike(child)) return;
    visit(child);
    walkOwnScope(child, visit);
  });
}

function findOwnReturn(statement: ts.Node): ts.Node | null {
  if (ts.isReturnStatement(statement)) return statement;
  let found: ts.Node | null = null;
  walkOwnScope(statement, (child) => {
    if (found) return;
    if (ts.isReturnStatement(child)) found = child;
  });
  return found;
}

/** The called hook's identifier, covering both `useX()` and `React.useX()`. */
function hookIdentifier(call: ts.CallExpression): string | null {
  const callee = call.expression;
  if (ts.isIdentifier(callee) && HOOK_NAME.test(callee.text)) return callee.text;
  if (ts.isPropertyAccessExpression(callee) && HOOK_NAME.test(callee.name.text)) {
    return callee.name.text;
  }
  return null;
}

function findOwnHookCall(statement: ts.Node): { node: ts.Node; name: string } | null {
  let found: { node: ts.Node; name: string } | null = null;
  const inspect = (child: ts.Node) => {
    if (found || !ts.isCallExpression(child)) return;
    const name = hookIdentifier(child);
    if (name) found = { node: child, name };
  };
  inspect(statement);
  walkOwnScope(statement, inspect);
  return found;
}

/** Name a function by its own identifier, or by the variable it is assigned to. */
function functionName(node: FunctionLike): string | null {
  if (!ts.isArrowFunction(node) && node.name && ts.isIdentifier(node.name)) {
    return node.name.text;
  }
  const parent = node.parent;
  if (parent && ts.isVariableDeclaration(parent) && ts.isIdentifier(parent.name)) {
    return parent.name.text;
  }
  return null;
}

function isHookBearing(name: string): boolean {
  return COMPONENT_NAME.test(name) || HOOK_NAME.test(name);
}

// ---------- The rule ----------

function checkFunctionBody(
  body: ts.Block,
  ownerName: string,
  file: string,
  source: ts.SourceFile,
  out: Violation[],
): void {
  const lineOf = (node: ts.Node) =>
    source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;

  let earlyReturn: ts.Node | null = null;
  const statements = body.statements;

  for (let i = 0; i < statements.length; i++) {
    const statement = statements[i];

    if (earlyReturn) {
      const hook = findOwnHookCall(statement);
      if (hook) {
        out.push({
          file,
          hookName: hook.name,
          hookLine: lineOf(hook.node),
          ownerName,
          returnLine: lineOf(earlyReturn),
        });
      }
    }

    // A return in the final statement is the normal exit, not an early one.
    if (i < statements.length - 1 && !earlyReturn) {
      earlyReturn = findOwnReturn(statement);
    }
  }
}

function checkSourceFile(absolutePath: string, out: Violation[]): void {
  const text = readFileSync(absolutePath, "utf8");
  const source = ts.createSourceFile(
    absolutePath,
    text,
    ts.ScriptTarget.Latest,
    /* setParentNodes */ true,
    absolutePath.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const file = relative(UI_ROOT, absolutePath);

  const visit = (node: ts.Node) => {
    if (isFunctionLike(node) && node.body && ts.isBlock(node.body)) {
      const name = functionName(node);
      if (name && isHookBearing(name)) {
        checkFunctionBody(node.body, name, file, source, out);
      }
    }
    node.forEachChild(visit);
  };
  visit(source);
}

function sourceFiles(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      sourceFiles(full, acc);
    } else if (full.endsWith(".ts") || full.endsWith(".tsx")) {
      acc.push(full);
    }
  }
  return acc;
}

// ---------- Self-test: the rule must actually catch the bug ----------

function checkSnippet(code: string): Violation[] {
  const source = ts.createSourceFile(
    "snippet.tsx",
    code,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const out: Violation[] = [];
  const visit = (node: ts.Node) => {
    if (isFunctionLike(node) && node.body && ts.isBlock(node.body)) {
      const name = functionName(node);
      if (name && isHookBearing(name)) {
        checkFunctionBody(node.body, name, "snippet.tsx", source, out);
      }
    }
    node.forEachChild(visit);
  };
  visit(source);
  return out;
}

// The exact shape of the PlotStudio #310 regression.
assertTrue(
  checkSnippet(`
    function Studio({ dataset }) {
      const [a, setA] = useState(0);
      if (!dataset) {
        return <Empty />;
      }
      const [b, setB] = useState(1);
      return <div>{a}{b}</div>;
    }
  `).length === 1,
  "detects a hook declared after an early return",
);

assertTrue(
  checkSnippet(`
    function Studio({ dataset }) {
      const [a, setA] = useState(0);
      const [b, setB] = useState(1);
      if (!dataset) {
        return <Empty />;
      }
      return <div>{a}{b}</div>;
    }
  `).length === 0,
  "accepts hooks declared before an early return",
);

// A `return` inside an effect callback is not an early return of the component.
assertTrue(
  checkSnippet(`
    function Studio({ dataset }) {
      useEffect(() => {
        if (!dataset) return;
        subscribe(dataset);
      }, [dataset]);
      const [a, setA] = useState(0);
      return <div>{a}</div>;
    }
  `).length === 0,
  "ignores returns inside nested callbacks",
);

// A hook inside a nested component belongs to that component's own sequence.
assertTrue(
  checkSnippet(`
    function Outer({ dataset }) {
      if (!dataset) {
        return <Empty />;
      }
      const Inner = () => {
        const [a, setA] = useState(0);
        return <div>{a}</div>;
      };
      return <Inner />;
    }
  `).length === 0,
  "ignores hooks inside nested components",
);

assertTrue(
  checkSnippet(`
    function useThing({ ready }) {
      const [a, setA] = useState(0);
      if (!ready) {
        return null;
      }
      const memo = useMemo(() => a, [a]);
      return memo;
    }
  `).length === 1,
  "applies the rule to custom hooks, not just components",
);

// A plain helper is not hook-bearing and must not be flagged.
assertTrue(
  checkSnippet(`
    function pickDefault(fields) {
      if (!fields.length) {
        return null;
      }
      const chosen = useful(fields);
      return chosen;
    }
  `).length === 0,
  "ignores non-component, non-hook functions",
);

// ---------- The codebase must be clean ----------

const violations: Violation[] = [];
for (const file of sourceFiles(SRC_ROOT)) {
  checkSourceFile(file, violations);
}

assertTrue(violations.length === 0, "no hook is called after an early return in src/");

const appSource = readFileSync(join(SRC_ROOT, "App.tsx"), "utf8");
assertTrue(
  appSource.includes("const PageView = memo(function PageView"),
  "PageView stays memoized so run progress cannot rerender inactive heavy pages",
);
assertTrue(
  appSource.includes('selectedRun={activePage === "runs" ? selectedRun : null}'),
  "selected run objects only flow into the active runs page",
);
assertTrue(
  appSource.includes("initialRunDir={initialRunDir}"),
  "Plot Studio receives stable run primitives instead of the live queue object",
);

// ---------- Summary ----------

if (violations.length > 0) {
  console.error("\nRules of Hooks violations (hook called after an early return):\n");
  for (const v of violations) {
    console.error(
      `  ${v.file}:${v.hookLine}  ${v.hookName}() in ${v.ownerName}()` +
        ` — unreachable when the return on line ${v.returnLine} is taken`,
    );
  }
  console.error(
    "\nMove these hooks above the early return. React counts hooks per render;" +
      "\na hook below a guard changes the count when the guard stops firing (React #310).\n",
  );
}

for (const line of failures) console.error(line);
console.log(`\nhook-order: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
