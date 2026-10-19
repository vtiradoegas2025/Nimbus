import { expect, type Page } from "@playwright/test";

interface PauseSnapshot {
  frame: number | null;
  clickedAtMs: number | null;
  observedFrames: number[];
  error: string | null;
}
interface PauseWitness { snapshot(): PauseSnapshot; dispose(): void }

declare global {
  interface Window { __nimbusPauseWitness?: PauseWitness }
}

export async function observePauseClick(page: Page): Promise<void> {
  await page.evaluate(() => {
    window.__nimbusPauseWitness?.dispose();
    let slider = document.querySelector<HTMLInputElement>('input[aria-label="Selected frame"]');
    if (!slider) throw new Error("Playback controls are unavailable");
    const state: PauseSnapshot = { frame: null, clickedAtMs: null, observedFrames: [], error: null };
    const sample = (records: MutationRecord[] = []) => {
      for (const record of records) if (record.attributeName === "value" && record.oldValue != null) {
        state.observedFrames.push(Number(record.oldValue));
      }
      state.observedFrames.push(Number(slider!.value));
    };
    const observer = new MutationObserver(sample);
    const clicked = (event: MouseEvent) => {
      const button = event.target instanceof Element ? event.target.closest("button") : null;
      if (button?.getAttribute("aria-label") !== "Pause frame playback") return;
      document.removeEventListener("click", clicked, true);
      slider = document.querySelector<HTMLInputElement>('input[aria-label="Selected frame"]');
      if (!slider) { state.error = "The selected-frame control disappeared at Pause"; return; }
      state.frame = Number(slider.value);
      state.clickedAtMs = performance.now();
      if (!Number.isSafeInteger(state.frame) || state.frame < 0 || state.frame > Number(slider.max)) {
        state.error = "The requested frame at Pause was invalid";
      }
      sample();
      observer.observe(slider, { attributes: true, attributeOldValue: true, attributeFilter: ["value", "aria-valuetext"] });
    };
    // Observe the actual click before React handles it; actionability work may advance playback first.
    document.addEventListener("click", clicked, true);
    window.__nimbusPauseWitness = {
      snapshot() {
        if (state.frame != null) sample(observer.takeRecords());
        return { ...state, observedFrames: [...state.observedFrames] };
      },
      dispose() { document.removeEventListener("click", clicked, true); observer.disconnect(); },
    };
  });
}

export async function capturedPausedFrame(page: Page): Promise<number> {
  try {
    const pause = await page.evaluate(() => window.__nimbusPauseWitness!.snapshot());
    expect(pause.error).toBeNull();
    expect(pause.frame).not.toBeNull();
    expect(pause.clickedAtMs).not.toBeNull();
    await expect(page.getByRole("button", { name: "Play frames", exact: true })).toBeVisible();
    await expect(page.getByRole("slider", { name: "Selected frame", exact: true })).toHaveValue(String(pause.frame));
    return pause.frame!;
  } catch (error) {
    await page.evaluate(() => window.__nimbusPauseWitness?.dispose()).catch(() => {});
    throw error;
  }
}

export async function clickPauseAndCaptureFrame(page: Page): Promise<number> {
  await observePauseClick(page);
  try {
    await page.getByRole("button", { name: "Pause frame playback", exact: true }).click();
    return await capturedPausedFrame(page);
  } catch (error) {
    await page.evaluate(() => window.__nimbusPauseWitness?.dispose()).catch(() => {});
    throw error;
  }
}

export async function expectPausedFrameStable(page: Page, frame: number): Promise<PauseSnapshot> {
  try {
    // Observe two intervals of the slowest (2 fps) fixture after outstanding presentation settles.
    await page.waitForTimeout(1000);
    const pause = await page.evaluate(() => window.__nimbusPauseWitness!.snapshot());
    expect(pause.frame).toBe(frame);
    expect(pause.observedFrames.length).toBeGreaterThan(0);
    expect(pause.observedFrames.every(value => value === frame)).toBe(true);
    await expect(page.getByRole("slider", { name: "Selected frame", exact: true })).toHaveValue(String(frame));
    await expect(page.getByRole("button", { name: "Play frames", exact: true })).toBeVisible();
    return pause;
  } finally {
    await page.evaluate(() => window.__nimbusPauseWitness?.dispose()).catch(() => {});
  }
}
