// Deterministic 21x21 SVG that looks like a Version-1 QR code. NOT scannable
// — purely a visual placeholder so the admin's TRANSFER TO card has a QR for
// the demo flow before real photographer uploads land. Real uploads (Phase F)
// arrive as `data:image/...` URLs on PayoutAccountSnapshot.qr.dataUrl and
// render as an <img> directly — this generator only fires when the dataUrl
// is the `mock://qr/...` sentinel.
//
// Output is a self-contained SVG string. Caller wraps it in an <a download>
// blob for the Download QR action, or drops the markup into the DOM via
// dangerouslySetInnerHTML for display.

const GRID_SIZE = 21;
const QUIET_CELLS = 4;

export function isMockQrUrl(url: string | null | undefined): boolean {
  return !!url && url.startsWith("mock://qr/");
}

export function mockQrSvgString(payload: string, pixelSize = 6): string {
  const grid = buildGrid(payload);
  const total = (GRID_SIZE + QUIET_CELLS * 2) * pixelSize;
  const offset = QUIET_CELLS * pixelSize;

  const cells: string[] = [];
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      if (grid[y][x] === 1) {
        cells.push(
          `<rect x="${offset + x * pixelSize}" y="${offset + y * pixelSize}" width="${pixelSize}" height="${pixelSize}"/>`,
        );
      }
    }
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${total}" height="${total}" viewBox="0 0 ${total} ${total}" shape-rendering="crispEdges"><rect width="${total}" height="${total}" fill="#ffffff"/><g fill="#0a0a0a">${cells.join("")}</g></svg>`;
}

function buildGrid(payload: string): number[][] {
  const grid: number[][] = Array.from({ length: GRID_SIZE }, () =>
    Array<number>(GRID_SIZE).fill(0),
  );
  paintFinder(grid, 0, 0);
  paintFinder(grid, GRID_SIZE - 7, 0);
  paintFinder(grid, 0, GRID_SIZE - 7);

  let h = fnv1a(payload);
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      if (isFinderArea(x, y)) continue;
      h = (h ^ ((h << 13) >>> 0)) >>> 0;
      h = (h ^ (h >>> 7)) >>> 0;
      h = (h ^ ((h << 17) >>> 0)) >>> 0;
      grid[y][x] = h & 1;
    }
  }
  return grid;
}

function paintFinder(grid: number[][], ox: number, oy: number): void {
  for (let dy = 0; dy < 7; dy++) {
    for (let dx = 0; dx < 7; dx++) {
      const onOuterRing = dx === 0 || dx === 6 || dy === 0 || dy === 6;
      const onInnerSquare = dx >= 2 && dx <= 4 && dy >= 2 && dy <= 4;
      grid[oy + dy][ox + dx] = onOuterRing || onInnerSquare ? 1 : 0;
    }
  }
}

function isFinderArea(x: number, y: number): boolean {
  if (x < 8 && y < 8) return true;
  if (x >= GRID_SIZE - 8 && y < 8) return true;
  if (x < 8 && y >= GRID_SIZE - 8) return true;
  return false;
}

function fnv1a(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h = (h ^ s.charCodeAt(i)) >>> 0;
    h = (h * 16777619) >>> 0;
  }
  return h;
}
