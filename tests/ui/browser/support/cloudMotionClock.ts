import type { Page } from "@playwright/test";

interface CloudMotionClock {
  arm(): void;
  advance(milliseconds: number): number;
  resume(): void;
  state(): { held: boolean; now: number; frames: number };
}

declare global {
  interface Window { __nimbusCloudMotionClock: CloudMotionClock }
}

export async function installCloudMotionClock(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const realNow = performance.now.bind(performance);
    const realFrame = window.requestAnimationFrame.bind(window);
    let offset = 0, floor = 0;
    let held: number | null = null;
    let armed = false;
    let frames = 0;
    const now = () => held ?? Math.max(floor, realNow() - offset);
    Object.defineProperty(performance, "now", { configurable: true, value: now });
    window.requestAnimationFrame = (callback) => realFrame((timestamp) =>
      callback(held ?? Math.max(floor, timestamp - offset)));
    const active = () => document.querySelector<HTMLElement>(".plot3d-overlay-host")
      ?.dataset.cloudMotionActive === "true";
    const observe = (changes: MutationRecord[]) => {
      if (armed || held != null) frames += changes.filter(change => change.attributeName === "data-cloud-motion-fraction").length;
      if (armed && active()) {
        held = now(); floor = held; armed = false;
      }
    };
    // Hold at publication, before a slow screenshot or driver round trip can
    // consume the transition. Worker I/O, timers, and GPU fences stay real.
    new MutationObserver(observe).observe(document, {
      subtree: true, attributes: true, attributeFilter: ["data-cloud-motion-active", "data-cloud-motion-fraction"],
    });
    window.__nimbusCloudMotionClock = {
      arm() {
        if (held != null || active()) throw new Error("Cloud clock must be idle before arming");
        frames = 0; armed = true;
      },
      advance(milliseconds) {
        if (held == null || !Number.isFinite(milliseconds) || milliseconds <= 0) {
          throw new Error("Cloud clock advancement requires a held clock and positive finite duration");
        }
        held += milliseconds; floor = held;
        return held;
      },
      resume() {
        armed = false;
        if (held != null) { floor = held; offset = realNow() - held; held = null; }
      },
      state: () => ({ held: held != null, now: now(), frames }),
    };
  });
}

export async function waitForCloudMotionFrame(page: Page): Promise<void> {
  await page.waitForFunction(() => {
    const clock = window.__nimbusCloudMotionClock.state();
    return clock.held && clock.frames >= 2;
  }, undefined, { timeout: 30_000 });
}

export async function armCloudMotionClock(page: Page): Promise<void> {
  await page.evaluate(() => window.__nimbusCloudMotionClock.arm());
}

export async function advanceCloudMotionClock(page: Page, milliseconds: number): Promise<void> {
  await page.evaluate((duration) => {
    const before = window.__nimbusCloudMotionClock.state();
    const after = window.__nimbusCloudMotionClock.advance(duration);
    if (!before.held || after !== before.now + duration || performance.now() !== after) {
      throw new Error("Cloud clock advanced inconsistently");
    }
  }, milliseconds);
}

export async function resumeCloudMotionClock(page: Page): Promise<void> {
  await page.evaluate(() => {
    const before = window.__nimbusCloudMotionClock.state();
    window.__nimbusCloudMotionClock.resume();
    const after = window.__nimbusCloudMotionClock.state();
    if (after.held || after.now < before.now) {
      throw new Error("Cloud clock resume changed the transition's time origin");
    }
  });
}
