import { chromium, expect } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
process.chdir(fileURLToPath(new URL("..", import.meta.url)));
await mkdir("test-results", { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined,
  headless: true,
  args: [
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
const context = await browser.newContext();
const page = await context.newPage();
await page.goto("http://localhost:8088");
await page.waitForFunction(() => navigator.serviceWorker.controller);
await page.waitForTimeout(1200);
await context.setOffline(true);
await page.reload({ waitUntil: "domcontentloaded" });
await expect(page.locator("h1")).toHaveText("Robot workspace");
await expect(page.locator("#connection")).toHaveText("Disconnected", {
  timeout: 10000,
});
await expect(page.locator('[data-variant="VARb"]')).toBeDisabled();
await expect(page.locator("#sort")).toBeDisabled();
await expect(page.locator("#talk")).toBeDisabled();
await page.locator("#chat-input").fill("pick two variant B boards");
await page.locator("#send").click();
await context.setOffline(false);
await expect(page.locator("#connection")).toHaveText("Connected", {
  timeout: 10000,
});
const token = await page.evaluate(() => localStorage.getItem("workcell-token"));
const result = await (
  await context.request.get("http://localhost:8088/api/state", {
    headers: { "X-Workcell-Token": token },
  })
).json();
if (result.panel.busy) throw new Error("Offline input replayed");
await writeFile(
  "test-results/offline.json",
  JSON.stringify({
    passed: true,
    shell_available: true,
    controls_disabled: true,
    no_offline_replay: true,
  }),
);
console.log(
  "Offline shell, disabled controls and no queued command replay passed",
);
await browser.close();
