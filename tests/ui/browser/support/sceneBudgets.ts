import type { ConsoleMessage, Page } from "@playwright/test";

// Cold full-quality software rendering can exceed a functional test's short
// setup wait. This bounds setup only; interaction latency assertions stay separate.
export const SCENE_SETUP_TIMEOUT_MS = 180_000;

export async function withSceneErrors(page: Page, wait: () => Promise<unknown>): Promise<void> {
  let reject!: (error: Error) => void;
  const failure = new Promise<never>((_, fail) => { reject = fail; });
  const pageError = (error: Error) => reject(error);
  const consoleError = (message: ConsoleMessage) => {
    if (message.type() === "error") reject(new Error(message.text()));
  };
  page.on("pageerror", pageError);
  page.on("console", consoleError);
  try { await Promise.race([wait(), failure]); }
  finally { page.off("pageerror", pageError); page.off("console", consoleError); }
}
