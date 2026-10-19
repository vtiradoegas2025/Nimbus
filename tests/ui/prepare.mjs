import { existsSync, lstatSync, realpathSync, symlinkSync } from "node:fs";
import { fileURLToPath } from "node:url";

const dependencies = fileURLToPath(new URL("../../ui/node_modules", import.meta.url));
const testDependencies = fileURLToPath(new URL("./node_modules", import.meta.url));
if (!existsSync(dependencies)) throw new Error("Install Studio dependencies with: npm --prefix ui ci");
if (existsSync(testDependencies)) {
  if (!lstatSync(testDependencies).isSymbolicLink()
      || realpathSync(testDependencies) !== realpathSync(dependencies)) {
    throw new Error("tests/ui/node_modules must reference the installed Studio dependencies");
  }
} else {
  symlinkSync(dependencies, testDependencies, process.platform === "win32" ? "junction" : "dir");
}
