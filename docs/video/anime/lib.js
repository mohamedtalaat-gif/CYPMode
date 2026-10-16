'use strict';
// Drawing helpers of the animation: easing, a few geometry builders and cel-shading. Every drawing function returns SVG markup; gradients, masks and filters
// it needs are pushed to DEFS and emitted once per frame by render() in main.js.

const W = 1920, H = 1080;
let DEFS = [], UID = 0;
const uid = (prefix = 'i') => prefix + (UID++);
const def = (markup) => { DEFS.push(markup); };

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const seg = (t, a, b) => (b > a ? clamp((t - a) / (b - a)) : (t >= b ? 1 : 0));
const smooth = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
const outCubic = (x) => 1 - Math.pow(1 - clamp(x), 3);
const inOutCubic = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const outBack = (x, k = 1.70158) => { x = clamp(x); return 1 + (k + 1) * Math.pow(x - 1, 3) + k * Math.pow(x - 1, 2); };
const outElastic = (x) => { x = clamp(x); return x === 0 || x === 1 ? x : Math.pow(2, -9 * x) * Math.sin((x * 10 - 0.75) * (2 * Math.PI) / 3) + 1; };
const hump = (x) => Math.sin(clamp(x) * Math.PI);
const n1 = (v) => Math.round(v * 10) / 10;
const rad = (deg) => deg * Math.PI / 180;

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const place = (x, y, rot = 0, sx = 1, sy = sx) =>
  `translate(${n1(x)} ${n1(y)})` + (rot ? ` rotate(${n1(rot)})` : '') + (sx !== 1 || sy !== 1 ? ` scale(${n1(sx * 1000) / 1000} ${n1(sy * 1000) / 1000})` : '');
const group = (transform, inner, extra = '') => `<g transform="${transform}" ${extra}>${inner}</g>`;

// -- geometry -------------------------------------------------------------------------------------------------------------------------

function pathBox(d) {
  const nums = d.match(/-?\d+(?:\.\d+)?/g).map(Number);
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (let i = 0; i + 1 < nums.length; i += 2) { x0 = Math.min(x0, nums[i]); x1 = Math.max(x1, nums[i]); y0 = Math.min(y0, nums[i + 1]); y1 = Math.max(y1, nums[i + 1]); }
  return [x0, y0, x1, y1];
}

// A shape with a base colour, a hard-edged shadow tone on the side away from the light, and a thin dark outline (the cel look).
function cel(d, fill, shade, o = {}) {
  const stroke = o.stroke ?? '#2b2140', sw = o.sw ?? 2.4, light = o.light ?? [-0.8, -0.9], shift = o.shift ?? 14;
  let out = `<path d="${d}" fill="${fill}"/>`;
  if (shade) {
    const id = uid('m'), [x0, y0, x1, y1] = pathBox(d);
    def(`<mask id="${id}" maskUnits="userSpaceOnUse" x="${n1(x0 - 40)}" y="${n1(y0 - 40)}" width="${n1(x1 - x0 + 80)}" height="${n1(y1 - y0 + 80)}">` +
        `<path d="${d}" fill="#fff"/><path d="${d}" fill="#000" transform="translate(${n1(light[0] * shift)} ${n1(light[1] * shift)})"/></mask>`);
    out += `<rect x="${n1(x0 - 40)}" y="${n1(y0 - 40)}" width="${n1(x1 - x0 + 80)}" height="${n1(y1 - y0 + 80)}" fill="${shade}" mask="url(#${id})" opacity="${o.shadeOpacity ?? 1}"/>`;
  }
  if (sw) out += `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round"/>`;
  return out;
}

function flat(d, fill, o = {}) {
  const stroke = o.stroke, sw = o.sw ?? 0;
  return `<path d="${d}" fill="${fill}"${o.opacity !== undefined ? ` opacity="${o.opacity}"` : ''}${stroke ? ` stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round" stroke-linecap="round"` : ''}${o.extra ? ' ' + o.extra : ''}/>`;
}

// Points on a Catmull-Rom spline through the given control points.
function spline(points, perSegment = 10) {
  const p = [points[0], ...points, points[points.length - 1]], out = [];
  for (let i = 1; i < p.length - 2; i++) {
    for (let k = 0; k < perSegment; k++) {
      const t = k / perSegment, t2 = t * t, t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p[i - 1][0], p[i][0], p[i + 1][0], p[i + 2][0]), f(p[i - 1][1], p[i][1], p[i + 1][1], p[i + 2][1])]);
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

// The outline (path data) of a stroke whose width changes along a curve: used for lashes, brows, hair locks and fingers.
function taperD(points, w0, w1, o = {}) {
  const pts = spline(points, o.detail ?? 8), n = pts.length, left = [], right = [];
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
    let dx = b[0] - a[0], dy = b[1] - a[1]; const len = Math.hypot(dx, dy) || 1; dx /= len; dy /= len;
    const u = i / (n - 1), w = lerp(w0, w1, o.curve ? Math.pow(u, o.curve) : u) / 2 * (o.bulge ? 1 + o.bulge * Math.sin(u * Math.PI) : 1);
    left.push([pts[i][0] - dy * w, pts[i][1] + dx * w]); right.push([pts[i][0] + dy * w, pts[i][1] - dx * w]);
  }
  const all = [...left, ...right.reverse()];
  return `M${all.map((q) => `${n1(q[0])} ${n1(q[1])}`).join(' L')} Z`;
}

// A stroke whose width changes along a curve, filled.
function taper(points, w0, w1, fill, o = {}) {
  return `<path d="${taperD(points, w0, w1, o)}" fill="${fill}"${o.opacity !== undefined ? ` opacity="${o.opacity}"` : ''}/>`;
}

// Outline of a smooth closed curve through the points.
function blobPath(points, perSegment = 8) {
  const p = [points[points.length - 1], ...points, points[0], points[1]], out = [];
  for (let i = 1; i < p.length - 2; i++) {
    for (let k = 0; k < perSegment; k++) {
      const t = k / perSegment, t2 = t * t, t3 = t2 * t;
      const f = (a, b, c, d) => 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p[i - 1][0], p[i][0], p[i + 1][0], p[i + 2][0]), f(p[i - 1][1], p[i][1], p[i + 1][1], p[i + 2][1])]);
    }
  }
  return `M${out.map((q) => `${n1(q[0])} ${n1(q[1])}`).join(' L')} Z`;
}

// Outline of a closed polygon through the points (sharp corners, for spiky hair).
function polyPath(points) {
  return `M${points.map((q) => `${n1(q[0])} ${n1(q[1])}`).join(' L')} Z`;
}

// A tapered limb between two joints (a capsule whose two ends have different radii).
function limbPath(p0, p1, r0, r1) {
  const ang = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]), pts = [], steps = 10;
  for (let i = 0; i <= steps; i++) { const a = ang - Math.PI / 2 + Math.PI * i / steps; pts.push([p1[0] + r1 * Math.cos(a), p1[1] + r1 * Math.sin(a)]); }
  for (let i = 0; i <= steps; i++) { const a = ang + Math.PI / 2 + Math.PI * i / steps; pts.push([p0[0] + r0 * Math.cos(a), p0[1] + r0 * Math.sin(a)]); }
  return `M${pts.map((q) => `${n1(q[0])} ${n1(q[1])}`).join(' L')} Z`;
}

function starPath(cx, cy, r, inner = 0.3, rot = 0, points = 4) {
  const out = [];
  for (let i = 0; i < points * 2; i++) { const a = rot + i * Math.PI / points - Math.PI / 2, rr = i % 2 ? r * inner : r; out.push(`${n1(cx + rr * Math.cos(a))} ${n1(cy + rr * Math.sin(a))}`); }
  return `M${out.join(' L')} Z`;
}

// -- colour, gradient and text helpers -----------------------------------------------------------------------------------------------

function linear(stops, x1 = 0, y1 = 0, x2 = 0, y2 = 1, units = 'objectBoundingBox') {
  const id = uid('g');
  def(`<linearGradient id="${id}" gradientUnits="${units}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}">${stops.map(([o, c, a]) => `<stop offset="${o}" stop-color="${c}"${a !== undefined ? ` stop-opacity="${a}"` : ''}/>`).join('')}</linearGradient>`);
  return `url(#${id})`;
}
function radial(stops, cx = 0.5, cy = 0.5, r = 0.5, units = 'objectBoundingBox') {
  const id = uid('r');
  def(`<radialGradient id="${id}" gradientUnits="${units}" cx="${cx}" cy="${cy}" r="${r}">${stops.map(([o, c, a]) => `<stop offset="${o}" stop-color="${c}"${a !== undefined ? ` stop-opacity="${a}"` : ''}/>`).join('')}</radialGradient>`);
  return `url(#${id})`;
}
function blur(std) {
  const id = uid('f');
  def(`<filter id="${id}" x="-50%" y="-50%" width="200%" height="200%" color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="${std}"/></filter>`);
  return `url(#${id})`;
}
function mixHex(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16), ch = (s) => Math.round(lerp((pa >> s) & 255, (pb >> s) & 255, t));
  return '#' + [16, 8, 0].map((s) => ch(s).toString(16).padStart(2, '0')).join('');
}
const shadeOf = (hex, t = 0.28, toward = '#3a2a6a') => mixHex(hex, toward, t);
const lightOf = (hex, t = 0.35) => mixHex(hex, '#ffffff', t);

const FONT_BODY = "'Avenir Next', 'Helvetica Neue', Arial, sans-serif";
const _measureContext = document.createElement('canvas').getContext('2d');

// The width in pixels that a string takes with the given font (measured by the browser, so that text can be fitted into boxes exactly).
function measure(str, size, o = {}) {
  _measureContext.font = `${o.italic ? 'italic ' : ''}${o.weight ?? 800} ${size}px ${o.family ?? FONT_BODY}`;
  return _measureContext.measureText(str).width + (o.spacing ?? 0) * str.length;
}

// Split a string into lines no wider than maxWidth (words are never broken; a single long word keeps its own line).
function wrap(str, maxWidth, size, o = {}) {
  const lines = [];
  for (const paragraph of String(str).split('\n')) {
    let line = '';
    for (const word of paragraph.split(' ')) {
      const trial = line ? line + ' ' + word : word;
      if (line && measure(trial, size, o) > maxWidth) { lines.push(line); line = word; } else line = trial;
    }
    lines.push(line);
  }
  return lines;
}

function text(s, x, y, size, o = {}) {
  const fill = o.fill ?? '#fff', stroke = o.stroke, anchor = o.anchor ?? 'middle', family = o.family ?? FONT_BODY, weight = o.weight ?? 800;
  if (o.maxWidth) {  // shrink the text, never the box: the string must fit in maxWidth
    const w = measure(s, size, { weight, family, spacing: o.spacing, italic: o.italic });
    if (w > o.maxWidth) size = size * o.maxWidth / w;
  }
  return `<text x="${n1(x)}" y="${n1(y)}" font-family="${family}" font-size="${n1(size)}" font-weight="${weight}" ` +
    `font-style="${o.italic ? 'italic' : 'normal'}" text-anchor="${anchor}" fill="${fill}" letter-spacing="${o.spacing ?? 0}"` +
    (stroke ? ` stroke="${stroke}" stroke-width="${o.strokeWidth ?? size / 8}" stroke-linejoin="round" paint-order="stroke"` : '') +
    (o.opacity !== undefined ? ` opacity="${o.opacity}"` : '') + (o.transform ? ` transform="${o.transform}"` : '') + `>${s}</text>`;
}

// Wrapped text: the lines of a paragraph inside a box of the given width, starting at (x, y) (the baseline of the first line); returns the markup and the height used.
function paragraph(str, x, y, size, o = {}) {
  const maxWidth = o.maxWidth ?? 800, lineHeight = o.lineHeight ?? size * 1.32;
  const lines = wrap(str, maxWidth, size, o);
  const markup = lines.map((line, i) => text(line, x, y + i * lineHeight, size, { ...o, maxWidth })).join('');
  return { markup, height: lines.length * lineHeight, lines: lines.length };
}

function sparkle(x, y, r, rot = 0, fill = '#fff', opacity = 1) {
  return `<path d="${starPath(x, y, r, 0.22, rot)}" fill="${fill}" opacity="${opacity}"/>`;
}

function speedLines(cx, cy, r0, r1, n, color, width, phase = 0, opacity = 0.7) {
  let out = '';
  for (let i = 0; i < n; i++) {
    const a = 2 * Math.PI * i / n + phase, w = width * (0.6 + 0.8 * ((i * 7919) % 13) / 13), c = Math.cos(a), s = Math.sin(a);
    out += `<path d="M${n1(cx + r0 * c - s * w)} ${n1(cy + r0 * s + c * w)} L${n1(cx + r1 * c)} ${n1(cy + r1 * s)} L${n1(cx + r0 * c + s * w)} ${n1(cy + r0 * s - c * w)} Z" fill="${color}" opacity="${opacity}"/>`;
  }
  return out;
}
