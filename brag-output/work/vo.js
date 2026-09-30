// Kokoro af_heart narration → vo/<n>.wav, one clip per line. node vo.js
const { KokoroTTS } = require("kokoro-js");
const fs = require("fs");

// [start sec, latest end sec, text]
const LINES = [
  [0.35, 2.9, "Fifteen thousand photos from one race."],
  [3.3, 5.9, "Meet QuickPitik, made in Cebu."],
  [6.3, 10.4, "Plug your camera into your phone, and every shot goes live."],
  [10.75, 14.4, "Crossed the finish line? One selfie finds every shot of you."],
  [14.7, 16.9, "No selfie? Your bib number works too."],
  [17.25, 19.9, "Pay by QR, and the full photo is yours."],
  [20.25, 22.9, "QuickPitik. Race photos, delivered in minutes."],
];
module.exports = LINES;

if (require.main === module) (async () => {
  const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", { dtype: "fp32", device: "cpu" });
  fs.mkdirSync("vo", { recursive: true });
  for (const [i, [start, end, text]] of LINES.entries()) {
    const speed = +(process.env[`SPEED_${i}`] || 1);
    const audio = await tts.generate(text, { voice: "af_heart", speed });
    await audio.save(`vo/${i}.wav`);
    const dur = audio.audio.length / audio.sampling_rate;
    console.log(`${i} ${dur.toFixed(2)}s  slot ${(end - start).toFixed(2)}s  ${dur > end - start ? "OVER" : "ok"}  "${text}"`);
  }
})();
