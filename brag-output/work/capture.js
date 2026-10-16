// node capture.js stills 1.0 4.5 ...   → stills/t-<sec>.png
// node capture.js video                → frames.mp4 (silent, 30fps, 23s)
const { chromium } = require("playwright-core");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const FFMPEG = require("ffmpeg-static");
const FPS = 30, DUR = 23;

(async () => {
  const [mode, ...times] = process.argv.slice(2);
  const browser = await chromium.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe" });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto("file:///" + path.resolve(__dirname, "index.html").replace(/\\/g, "/"));
  await page.evaluate(() => window.ready);
  await page.waitForTimeout(300);

  if (mode === "stills") {
    fs.mkdirSync(path.join(__dirname, "stills"), { recursive: true });
    for (const s of times) {
      await page.evaluate(t => render(t), +s);
      await page.screenshot({ path: path.join(__dirname, "stills", `t-${s}.png`) });
    }
  } else {
    const ff = spawn(FFMPEG, ["-y", "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", path.join(__dirname, "frames.mp4")],
      { stdio: ["pipe", "ignore", "inherit"] });
    for (let f = 0; f < FPS * DUR; f++) {
      await page.evaluate(t => render(t), f / FPS);
      const buf = await page.screenshot({ type: "jpeg", quality: 95 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
      if (f % 60 === 0) process.stdout.write(`frame ${f}\n`);
    }
    ff.stdin.end();
    await new Promise(r => ff.on("close", r));
  }
  await browser.close();
})();
