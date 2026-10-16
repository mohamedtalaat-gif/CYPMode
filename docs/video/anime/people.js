'use strict';
// The laboratory staff, drawn as young adult men in an anime style: about seven and a half heads tall, a small face with a defined jaw, narrow eyes with layered irises, hair built from
// separate locks, drawn hands, thin coloured line art, soft shading and a rim light. Coordinates: the origin is the middle of the feet, the y axis points down, the figure is about 940 high.
// The head is drawn in "face units" (a face is about 190 high) and scaled onto the body by HEAD.k.

const SKIN_DEFAULT = '#ffe2cf';
const HEAD = { y: -836, k: 0.72 };
const SHOULDER = 100, HIP_Y = -470, KNEE_Y = -246, ANKLE_Y = -40;

const POSES = {
  stand: { arms: [[7, 3], [7, 3]], hands: ['relaxed', 'relaxed'], lean: 0, head: 0 },
  present: { arms: [[7, 3], [70, 112]], hands: ['relaxed', 'open'], lean: 2, head: -3 },
  point: { arms: [[7, 3], [78, 92]], hands: ['relaxed', 'point'], lean: 3, head: -4 },
  wave: { arms: [[7, 3], [96, 160]], hands: ['relaxed', 'open'], lean: 0, head: 4 },
  cheer: { arms: [[112, 168], [112, 168]], hands: ['open', 'open'], lean: 0, head: 0 },
  hold: { arms: [[7, 3], [48, 112]], hands: ['relaxed', 'grip'], lean: 0, head: 3 },
  hips: { arms: [[40, -28], [40, -28]], hands: ['fist', 'fist'], lean: 0, head: -3 },
  think: { arms: [[7, 3], [34, 150]], hands: ['relaxed', 'fist'], lean: -1, head: 5 },
};

// -- face -----------------------------------------------------------------------------------------------------------------------------

// The face outline. `jaw` widens or narrows the lower face, `chin` lengthens it, `cheek` fills the cheeks; a character's `face` object holds these three numbers.
function facePath(jaw = 1, chin = 0, cheek = 1) {
  const j = (v) => n1(v * jaw), c = (v) => n1(v * cheek), cc = chin;
  return `M-64 -32 C-66 -72 -37 -92 0 -92 C37 -92 66 -72 64 -32 C${c(64)} 6 ${n1(61 * jaw * cheek)} 36 ${j(52)} 60 C${j(43)} 82 ${j(23)} ${98 + cc} 0 ${102 + cc} C${-j(23)} ${98 + cc} ${-j(43)} 82 ${-j(52)} 60 C${n1(-61 * jaw * cheek)} 36 ${c(-64)} 6 -64 -32 Z`;
}
const FACE = facePath();
const EAR = (side) => `M${side * 59} -6 C${side * 72} -12 ${side * 79} 12 ${side * 69} 30 C${side * 65} 36 ${side * 60} 33 ${side * 59} 27 Z`;
const EYE = 'M-20 4 C-14 -8 0 -14 12 -12 C20 -10 25 -5 26 0 C20 7 6 12 -6 11 C-14 10 -19 7 -20 4 Z';

function eyeMarkup(side, o) {
  const iris = o.eye, open = 1 - 0.95 * o.blink, look = o.look ?? [0, 0], ink = shadeOf(o.hair, 0.55, '#120a22');
  const clip = uid('c');
  def(`<clipPath id="${clip}"><path d="${EYE}"/></clipPath>`);
  const irisFill = linear([[0, shadeOf(iris, 0.72, '#06030f')], [0.4, shadeOf(iris, 0.1, '#06030f')], [0.8, lightOf(iris, 0.35)], [1, lightOf(iris, 0.7)]]);
  const lx = look[0] * 4, ly = look[1] * 3;
  let eye = '';
  if (open > 0.16) {
    eye += `<g transform="translate(0 -6) scale(1 ${n1(open * 1000) / 1000}) translate(0 6)">`;
    eye += `<path d="${EYE}" fill="#fbf7f4"/>`;
    eye += `<g clip-path="url(#${clip})">`;
    eye += `<ellipse cx="${n1(3 + lx)}" cy="${n1(-0.5 + ly)}" rx="11.6" ry="14" fill="${irisFill}"/>`;
    eye += `<ellipse cx="${n1(3 + lx)}" cy="${n1(-0.5 + ly)}" rx="11.6" ry="14" fill="none" stroke="${shadeOf(iris, 0.8, '#06030f')}" stroke-width="1.6"/>`;
    eye += `<ellipse cx="${n1(3 + lx)}" cy="${n1(0.5 + ly)}" rx="5.4" ry="7" fill="#0c0618"/>`;
    eye += `<path d="M${n1(-4 + lx)} ${n1(7 + ly)} C${n1(0 + lx)} ${n1(11 + ly)} ${n1(8 + lx)} ${n1(11 + ly)} ${n1(12 + lx)} ${n1(6 + ly)}" fill="none" stroke="${lightOf(iris, 0.85)}" stroke-width="2.2" opacity="0.7" stroke-linecap="round"/>`;
    eye += `<ellipse cx="${n1(7.5 + lx)}" cy="${n1(-6 + ly)}" rx="4" ry="4.8" fill="#fff"/>`;
    eye += `<circle cx="${n1(-2 + lx)}" cy="${n1(5 + ly)}" r="2" fill="#fff" opacity="0.92"/>`;
    eye += `<path d="M-22 -9 C-8 -17 12 -16 26 -5 L26 -16 L-22 -16 Z" fill="${shadeOf(o.skin, 0.25, '#7a3a4a')}" opacity="0.28"/>`;
    eye += `</g>`;
    eye += `<path d="M-19 9 C-6 14 8 13 19 7" fill="none" stroke="${shadeOf(o.skin, 0.55, '#6a2a3a')}" stroke-width="1.3" opacity="0.5"/>`;
    eye += `</g>`;
    const top = -13 + (1 - open) * 9;
    eye += taper([[-24, 5], [-15, top + 4], [0, top - 1], [14, top + 1], [25, -3], [31, -5]], 3.2, 1.2, ink, { bulge: 1.5 });
    eye += taper([[-12, top - 3], [2, top - 8], [17, top - 5]], 1.5, 0.6, shadeOf(o.skin, 0.5, '#6a2a3a'), { opacity: 0.55 });
  } else {
    eye += taper([[-22, 3], [-10, 9], [8, 10], [27, -1]], 3.6, 1.6, ink, { bulge: 0.6 });
  }
  const s = o.eyeScale ?? 1, spacing = o.eyeSpacing ?? 1, tilt = (o.eyeTilt ?? 0) * side;
  return `<g transform="translate(${n1(side * 29 * spacing)} ${n1(8 + (o.eyeY ?? 0))}) rotate(${n1(tilt)}) scale(${n1(side * s * 100) / 100} ${s})">${eye}</g>`;
}

// Thin brows that follow the hair colour; tilt > 0 raises the inner end (worry), tilt < 0 lowers it (anger).
function browMarkup(side, o) {
  const tilt = { smile: 2, proud: -1, happy: 3, worry: 8, shout: -4, angry: -10, calm: 0, surprised: 5, think: 4, sad: 9 }[o.mood] ?? 0;
  const lift = { surprised: -9, shout: -2, proud: -3, worry: 0 }[o.mood] ?? 0;
  const inner = [side * 11, -26 - tilt * 0.7 + lift], mid = [side * 33, -34 + lift - tilt * 0.1], outer = [side * 56, -28 + tilt * 0.4 + lift];
  const k = o.browLength ?? 1;
  return taper([inner, [mid[0] * k, mid[1]], [outer[0] * k, outer[1]]], o.browWidth ?? 6.4, 2.2, shadeOf(o.hair, 0.5, '#150c22'), { bulge: 0.2 });
}

// Facial hair kinds: stubble, mustache, goatee, beard. Drawn on top of the face, under the mouth.
function facialHair(kind, color, fp) {
  if (!kind) return '';
  const tone = shadeOf(color, 0.3, '#10081f'), edge = shadeOf(color, 0.6, '#0a0620');
  const mustache = cel('M-24 50 C-13 40 -3 46 0 50 C3 46 13 40 24 50 C15 57 5 54 0 53 C-5 54 -15 57 -24 50 Z', color, tone, { stroke: edge, sw: 1.4, shift: 4 });
  switch (kind) {
    case 'stubble': {
      const id = uid('c');
      def(`<clipPath id="${id}"><path d="${fp}"/></clipPath>`);
      return `<g clip-path="url(#${id})"><path d="M-66 22 C-66 58 -50 88 -24 102 C-8 108 8 108 24 102 C50 88 66 58 66 22 C60 46 44 62 22 66 C10 58 -10 58 -22 66 C-44 62 -60 46 -66 22 Z" fill="${tone}" opacity="0.34"/>` +
        `<path d="M-20 50 C-11 44 -3 47 0 50 C3 47 11 44 20 50 C11 54 4 52 0 52 C-4 52 -11 54 -20 50 Z" fill="${tone}" opacity="0.3"/></g>`;
    }
    case 'goatee':
      return cel('M-18 70 C-14 98 14 98 18 70 C10 77 -10 77 -18 70 Z', color, tone, { stroke: edge, sw: 1.4, shift: 4 }) + mustache;
    case 'beard':
      return cel('M-62 8 C-66 52 -52 90 -22 103 C-8 108 8 108 22 103 C52 90 66 52 62 8 L56 28 C50 48 40 58 26 60 C18 62 12 74 0 74 C-12 74 -18 62 -26 60 C-40 58 -50 48 -56 28 Z', color, tone, { stroke: edge, sw: 1.4, shift: 7 }) + mustache;
    case 'mustache':
      return mustache;
    default: return '';
  }
}

function faceMarkup(o) {
  const skin = o.skin, shade = shadeOf(skin, 0.3, '#b86a82'), deep = shadeOf(skin, 0.5, '#9a4a62'), line = shadeOf(skin, 0.6, '#7a3a48'), clip = uid('c');
  const f = o.face ?? {}, fp = facePath(f.jaw ?? 1, f.chin ?? 0, f.cheek ?? 1), earSize = f.ear ?? 1;
  def(`<clipPath id="${clip}"><path d="${fp}"/></clipPath>`);
  const ear = (side) => `<g transform="translate(${side * 59} 10) scale(${earSize}) translate(${-side * 59} -10)">${cel(EAR(side), skin, shade, { shift: 4, sw: 1.6, stroke: line })}</g>`;
  let out = ear(-1) + ear(1);
  out += `<path d="M-64 6 C-67 10 -67 20 -63 24 M64 6 C67 10 67 20 63 24" fill="none" stroke="${line}" stroke-width="1.4" opacity="0.6"/>`;
  out += cel(fp, skin, shade, { shift: 11, light: [-0.7, -0.9], sw: 2.1, stroke: shadeOf(skin, 0.7, '#4a2030') });
  // soft shading: the shadow of the hair on the forehead, the cheek planes, under the lower lip and along the jaw
  out += `<g clip-path="url(#${clip})">` +
    `<path d="M-70 -54 C-30 -38 30 -38 70 -54 L70 -100 L-70 -100 Z" fill="${deep}" opacity="0.45"/>` +
    `<ellipse cx="-38" cy="40" rx="20" ry="26" fill="${shade}" opacity="0.35" filter="${blur(6)}"/><ellipse cx="38" cy="40" rx="20" ry="26" fill="${shade}" opacity="0.35" filter="${blur(6)}"/>` +
    `<path d="M-62 40 C-52 78 -26 98 0 102 C26 98 52 78 62 40" fill="none" stroke="${shade}" stroke-width="7" opacity="0.55"/></g>`;
  if (o.blush) out += `<g clip-path="url(#${clip})"><ellipse cx="-38" cy="38" rx="17" ry="9" fill="#ff7f8f" opacity="${o.blush}" filter="${blur(4)}"/><ellipse cx="38" cy="38" rx="17" ry="9" fill="#ff7f8f" opacity="${o.blush}" filter="${blur(4)}"/></g>`;
  out += `<g transform="scale(${f.jaw ?? 1} ${1 + (f.chin ?? 0) / 110})">${facialHair(o.beard, o.hair, fp)}</g>`;
  out += eyeMarkup(-1, o) + eyeMarkup(1, o);
  out += browMarkup(-1, o) + browMarkup(1, o);
  const age = o.lines ?? 0;
  if (age > 0) {  // the lines of an older face: the forehead, the corners of the eyes and the folds from the nose to the mouth
    const a = Math.min(1, age) * 0.5, crease = shadeOf(skin, 0.5, '#7a3a48');
    out += `<g fill="none" stroke="${crease}" stroke-width="1.6" stroke-linecap="round" opacity="${n1(a * 100) / 100}">` +
      `<path d="M-34 -44 C-14 -50 14 -50 34 -44"/><path d="M-30 -52 C-10 -57 10 -57 30 -52"/>` +
      `<path d="M-60 6 C-56 10 -54 13 -52 18 M60 6 C56 10 54 13 52 18"/>` +
      `<path d="M-26 44 C-32 56 -33 66 -28 76 M26 44 C32 56 33 66 28 76"/></g>`;
  }
  const nose = o.nose ?? 1;
  out += `<g transform="translate(0 30) scale(${nose}) translate(0 -30)"><path d="M2 18 C3 26 5 32 8 37" fill="none" stroke="${line}" stroke-width="1.8" stroke-linecap="round"/><path d="M9 38 C5 41 -1 41 -5 38" fill="none" stroke="${line}" stroke-width="1.8" stroke-linecap="round"/></g>`;
  out += `<g transform="translate(0 ${68 + (o.mouthY ?? 0)}) scale(${0.9 * (o.mouthWidth ?? 1)} 0.9)">${mouthShape(o.mouth ?? 'smile', o.talk ?? 0)}</g>`;
  out += `<path d="M-7 90 C-2 93 2 93 7 90" fill="none" stroke="${line}" stroke-width="1.5" stroke-linecap="round" opacity="0.4"/>`;
  return out;
}

// -- hair: built from locks (tapered strands) -----------------------------------------------------------------------------------------

// A lock from its root to its tip, bent sideways by `bend` (a fraction of its length, positive = to the right of the direction of growth); w is the width near the root.
function lockPoints(rx, ry, tx, ty, bend = 0.12) {
  const mx = (rx + tx) / 2, my = (ry + ty) / 2, dx = tx - rx, dy = ty - ry;
  return [[rx, ry], [mx - dy * bend, my + dx * bend], [tx, ty]];
}

const lock = (rx, ry, tx, ty, w = 26, bend = 0.12) => ({ pts: lockPoints(rx, ry, tx, ty, bend), w });

const BACK_MASS = [[-76, -34], [-80, 4], [-72, 24], [-46, 34], [0, 36], [46, 34], [72, 24], [80, 4], [76, -34], [68, -90], [0, -112], [-68, -90]];
const HAIR = {
  short: {
    back: BACK_MASS,
    locks: [lock(-60, -78, -74, -20, 30, 0.05), lock(60, -78, 74, -20, 30, -0.05), lock(-40, -100, -64, -52, 34, 0.1), lock(40, -100, 64, -52, 34, -0.1), lock(-10, -104, -46, -56, 36, 0.12), lock(14, -104, 40, -58, 34, -0.12),
      lock(-26, -100, -10, -60, 30, 0.1), lock(0, -102, 18, -62, 28, -0.1), lock(-50, -96, -58, -64, 22, 0.05), lock(52, -96, 58, -64, 22, -0.05)],
    side: [[[-62, -60], [-70, -20], [-66, 14]], [[62, -60], [70, -20], [66, 14]]],
  },
  sidepart: {
    back: BACK_MASS,
    locks: [lock(-60, -78, -74, -14, 30, 0.05), lock(60, -78, 74, -14, 30, -0.05), lock(-48, -100, -68, -50, 34, 0.1), lock(-24, -108, -50, -44, 38, 0.14), lock(-4, -112, -22, -40, 36, 0.16), lock(18, -112, 6, -44, 34, 0.14),
      lock(36, -106, 38, -50, 30, 0.1), lock(52, -96, 66, -44, 28, -0.06), lock(-6, -118, 28, -134, 30, -0.2), lock(20, -114, 54, -118, 26, -0.15), lock(-30, -112, -4, -128, 28, -0.18)],
    side: [[[-62, -60], [-70, -20], [-66, 14]], [[62, -60], [70, -20], [66, 14]]],
  },
  crop: {
    back: BACK_MASS,
    locks: [lock(-60, -78, -74, -16, 30, 0.05), lock(60, -78, 74, -16, 30, -0.05), lock(-46, -100, -64, -52, 34, 0.1), lock(-22, -106, -44, -50, 36, 0.12), lock(2, -108, -8, -50, 34, 0.1), lock(26, -106, 40, -52, 34, -0.1),
      lock(46, -98, 62, -52, 32, -0.1), lock(-30, -108, -4, -122, 28, -0.2), lock(0, -112, 36, -118, 28, -0.16)],
    side: [[[-62, -60], [-70, -20], [-66, 14]], [[62, -60], [70, -20], [66, 14]]],
  },
  messy: {
    back: BACK_MASS,
    locks: [lock(-60, -78, -78, -22, 30, 0.05), lock(60, -78, 78, -22, 30, -0.05), lock(-50, -94, -88, -118, 30, -0.2), lock(-30, -104, -52, -146, 28, -0.16), lock(-8, -108, -14, -156, 28, 0.12), lock(14, -108, 26, -150, 28, -0.1),
      lock(36, -102, 62, -138, 28, -0.16), lock(54, -92, 88, -104, 26, -0.2), lock(-54, -92, -66, -52, 28, 0.08), lock(-28, -104, -36, -50, 30, 0.1), lock(-4, -108, 4, -48, 26, 0.08), lock(22, -106, 30, -50, 30, -0.1),
      lock(48, -96, 60, -52, 28, -0.08), lock(-14, -108, -22, -46, 24, 0.1)],
    side: [[[-62, -60], [-72, -16], [-66, 12]], [[62, -60], [72, -16], [66, 12]]],
  },
  undercut: {
    back: [[-74, -30], [-76, 2], [-66, 20], [0, 24], [66, 20], [76, 2], [74, -30], [66, -88], [0, -112], [-66, -88]],
    locks: [lock(-52, -98, -62, -62, 30, 0.04), lock(-28, -106, -52, -128, 34, -0.22), lock(-6, -112, -30, -140, 34, -0.26), lock(18, -112, -2, -146, 34, -0.3), lock(40, -104, 24, -138, 32, -0.3), lock(56, -92, 52, -118, 28, -0.24),
      lock(-40, -100, -22, -62, 30, 0.1), lock(-10, -108, 6, -70, 30, 0.1), lock(26, -106, 34, -66, 28, 0.08)],
    side: [[[-62, -52], [-68, -20], [-64, 6]], [[62, -52], [68, -20], [64, 6]]],
  },
  spiky: {
    back: BACK_MASS,
    locks: [lock(-60, -78, -82, -26, 30, 0.05), lock(60, -78, 82, -26, 30, -0.05), lock(-52, -92, -90, -128, 30, -0.2), lock(-34, -102, -64, -152, 30, -0.18), lock(-14, -108, -30, -162, 30, -0.12), lock(8, -110, 10, -166, 30, 0.04),
      lock(28, -106, 50, -158, 30, -0.12), lock(46, -98, 80, -140, 30, -0.18), lock(60, -90, 92, -108, 26, -0.2), lock(-40, -96, -52, -52, 28, 0.08), lock(-12, -106, -14, -54, 28, 0.06), lock(20, -106, 26, -54, 28, -0.06), lock(44, -98, 54, -54, 26, -0.08)],
    side: [[[-62, -60], [-72, -16], [-66, 12]], [[62, -60], [72, -16], [66, 12]]],
  },
};

function hairBack(style, color) {
  const h = HAIR[style], shade = shadeOf(color, 0.4, '#140a2c');
  return cel(blobPath(h.back), color, shade, { stroke: shadeOf(color, 0.7, '#0a0620'), sw: 1.8, shift: 18 });
}

function hairFront(style, color) {
  const h = HAIR[style], dark = shadeOf(color, 0.42, '#140a2c'), edge = shadeOf(color, 0.72, '#0a0620'), light = lightOf(color, 0.62);
  let out = '';
  for (const pts of h.side) out += taper(pts, 14, 3, color, {}) + `<path d="${taperD(pts, 14, 3)}" fill="none" stroke="${edge}" stroke-width="1.2"/>`;
  // the locks, back to front: shadow tone, colour, outline, a highlight streak along the side facing the light
  for (const { pts, w } of h.locks) {
    const d = taperD(pts, w, 1.4, { bulge: 0.3 });
    out += `<path d="${d}" fill="${dark}" stroke="${edge}" stroke-width="1.5" stroke-linejoin="round"/>`;
    const lit = pts.map(([x, y]) => [x - 3, y - 3]);
    out += `<path d="${taperD(lit, w * 0.78, 1, { bulge: 0.3 })}" fill="${color}"/>`;
    const streak = pts.map(([x, y], i) => [lerp(x, pts[pts.length - 1][0], 0.12 * i) - w * 0.16, lerp(y, pts[pts.length - 1][1], 0.12 * i) - 2]);
    out += `<path d="${taperD(streak, w * 0.2, 0.4)}" fill="${light}" opacity="0.7"/>`;
  }
  // the glossy band across the crown
  const band = linear([[0, light, 0], [0.5, light, 0.55], [1, light, 0]], 0, 0, 1, 0);
  out += `<path d="M-58 -84 C-26 -112 26 -112 58 -84 C32 -96 -32 -96 -58 -84 Z" fill="${band}"/>`;
  return out;
}

// -- accessories ----------------------------------------------------------------------------------------------------------------------

function headAccessory(name) {
  switch (name) {
    case 'goggles': {
      const strap = `<path d="M-76 -76 C-40 -96 40 -96 76 -76" fill="none" stroke="#2d8f7b" stroke-width="8" stroke-linecap="round"/>`;
      const lens = (x) => `<circle cx="${x}" cy="-100" r="25" fill="${radial([[0, '#d9fbff', 0.95], [1, '#7fd4e0', 0.9]])}" stroke="#2d8f7b" stroke-width="6"/><ellipse cx="${x + 7}" cy="-109" rx="6" ry="3.4" fill="#fff" opacity="0.9" transform="rotate(-30 ${x + 7} -109)"/>`;
      return strap + lens(-31) + lens(31) + `<rect x="-7" y="-105" width="14" height="10" rx="3" fill="#2d8f7b"/>`;
    }
    case 'glasses': {
      const lens = (x) => `<rect x="${x - 25}" y="-10" width="50" height="38" rx="12" fill="#cfe9ff" fill-opacity="0.16" stroke="#3b3b6d" stroke-width="3"/><path d="M${x - 14} -2 L${x + 2} -2" stroke="#fff" stroke-width="2.6" opacity="0.7" stroke-linecap="round"/>`;
      return `<path d="M-6 6 L6 6" stroke="#3b3b6d" stroke-width="3" stroke-linecap="round"/>` + `<path d="M-56 4 L-68 2 M56 4 L68 2" stroke="#3b3b6d" stroke-width="2.8" stroke-linecap="round"/>` + lens(-31) + lens(31);
    }
    case 'frames': {
      const lens = (x) => `<rect x="${x - 26}" y="-12" width="52" height="36" rx="5" fill="#cfe9ff" fill-opacity="0.14" stroke="#1d2238" stroke-width="4.6"/><path d="M${x - 16} -4 L${x - 2} -4" stroke="#fff" stroke-width="2.6" opacity="0.7" stroke-linecap="round"/>`;
      return `<path d="M-5 4 L5 4" stroke="#1d2238" stroke-width="4" stroke-linecap="round"/>` + `<path d="M-57 2 L-68 0 M57 2 L68 0" stroke="#1d2238" stroke-width="3.6" stroke-linecap="round"/>` + lens(-31) + lens(31);
    }
    case 'headband':
      return `<path d="M-74 -62 C-40 -88 40 -88 74 -62" fill="none" stroke="#e5484d" stroke-width="12" stroke-linecap="round"/><path d="M-74 -67 C-40 -93 40 -93 74 -67" fill="none" stroke="#ff8f93" stroke-width="2.6" opacity="0.7"/>` +
        `<path d="M72 -60 L106 -44 L90 -66 L112 -78 Z" fill="#e5484d" stroke="#6a1020" stroke-width="1.6"/>`;
    case 'visor':
      return `<path d="M-76 -66 C-40 -102 40 -102 76 -66 L124 -54 C94 -46 -2 -48 -76 -66 Z" fill="#2b5fa8" stroke="#10285a" stroke-width="1.8"/><path d="M-64 -72 C-30 -96 30 -96 64 -72" fill="none" stroke="#6fa0e6" stroke-width="3.6" opacity="0.7"/>`;
    case 'headset':
      return `<path d="M-72 -46 C-70 -100 70 -100 72 -46" fill="none" stroke="#3a3f4f" stroke-width="7" stroke-linecap="round"/>` +
        `<ellipse cx="-74" cy="-30" rx="12" ry="22" fill="#4a5064" stroke="#1d2030" stroke-width="2.4"/><ellipse cx="-76" cy="-30" rx="6" ry="14" fill="#8a93ad" opacity="0.7"/>` +
        `<path d="M-72 -10 C-72 36 -52 60 -26 66" fill="none" stroke="#3a3f4f" stroke-width="4" stroke-linecap="round"/><circle cx="-24" cy="66" r="6" fill="#4a5064" stroke="#1d2030" stroke-width="2"/>`;
    default: return '';
  }
}

// -- hands ----------------------------------------------------------------------------------------------------------------------------

// A finger from its base (x, y) in the hand frame: two segments, the second bent by `curl` degrees; length and width are in hand units.
function finger(x, y, angle, len, w, curl, skin, shade, line) {
  const a1 = rad(angle), a2 = rad(angle + curl), l1 = len * 0.55, l2 = len * 0.45;
  const p1 = [x + Math.sin(a1) * 0 + Math.sin(-a1) * -0, y];
  const j = [x - Math.sin(a1) * l1, y + Math.cos(a1) * l1], t = [j[0] - Math.sin(a2) * l2, j[1] + Math.cos(a2) * l2];
  const d = taperD([[x, y], j, t], w, w * 0.72, { bulge: 0.08 });
  return `<path d="${d}" fill="${skin}" stroke="${line}" stroke-width="1.3" stroke-linejoin="round"/>` + `<path d="${taperD([[x + w * 0.2, y], j, t], w * 0.3, w * 0.2)}" fill="${shade}" opacity="0.35"/>`;
}

// The hand in a frame whose origin is the wrist and whose +y runs along the forearm; the thumb is on the side of `side`.
function handMarkup(side, gesture, skin) {
  const shade = shadeOf(skin, 0.3, '#b86a82'), line = shadeOf(skin, 0.65, '#5a2a38');
  const palm = cel('M-17 -2 C-20 14 -19 30 -14 42 C-6 48 8 48 15 42 C20 30 20 14 17 -2 Z', skin, shade, { shift: 6, sw: 1.4, stroke: line });
  const thumbAngle = { relaxed: -8, open: -42, point: -22, fist: -14, grip: -20 }[gesture] ?? -10;
  const thumb = `<g transform="scale(${-side} 1)">` + finger(13, 14, thumbAngle + 18, 36, 11, -22, skin, shade, line) + `</g>`;
  const f = { x: [-12.5, -4.2, 4.2, 12.5], len: [30, 36, 34, 27], w: 8.4 };
  let fingers = '';
  const spread = { relaxed: [5, 1, -2, -6], open: [16, 6, -5, -16], point: [4, 0, -3, -7], fist: [0, 0, 0, 0], grip: [0, 0, 0, 0] }[gesture] ?? [0, 0, 0, 0];
  const curl = { relaxed: [26, 30, 32, 34], open: [4, 3, 3, 5], point: [-4, 70, 74, 78], fist: [82, 86, 88, 90], grip: [70, 74, 76, 78] }[gesture] ?? [20, 20, 20, 20];
  const len = { relaxed: 1, open: 1.02, point: 1.05, fist: 0.78, grip: 0.82 }[gesture] ?? 1;
  for (let i = 0; i < 4; i++) fingers += finger(f.x[i], 42, spread[i], f.len[i] * len, f.w, curl[i], skin, shade, line);
  return fingers + palm + thumb;
}

// -- the body ---------------------------------------------------------------------------------------------------------------------------

const ARM_UPPER = 140, ARM_FORE = 126;

function armGeometry(side, upper, fore, build) {
  const shoulder = [side * SHOULDER * build, -732], dir = (a) => [side * Math.sin(rad(a)), Math.cos(rad(a))];
  const u = dir(upper), f = dir(fore);
  const elbow = [shoulder[0] + u[0] * ARM_UPPER, shoulder[1] + u[1] * ARM_UPPER];
  const wrist = [elbow[0] + f[0] * ARM_FORE, elbow[1] + f[1] * ARM_FORE];
  return { shoulder, elbow, wrist, frame: Math.atan2(-f[0], f[1]) * 180 / Math.PI };
}

// A band across a limb between two points (the fold of a rolled sleeve): a rectangle perpendicular to the limb, a little wider than it.
function bandPath(p0, p1, r) {
  const dx = p1[0] - p0[0], dy = p1[1] - p0[1], len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
  return `M${n1(p0[0] + nx * r)} ${n1(p0[1] + ny * r)} L${n1(p1[0] + nx * r)} ${n1(p1[1] + ny * r)} L${n1(p1[0] - nx * r)} ${n1(p1[1] - ny * r)} L${n1(p0[0] - nx * r)} ${n1(p0[1] - ny * r)} Z`;
}

// An arm: the shoulder cap, the upper arm and the forearm in the sleeve (a rolled sleeve ends below the elbow in a thick fold, with the bare forearm under it), the cuff, the hand.
function armMarkup(side, upper, fore, gesture, o) {
  const a = armGeometry(side, upper, fore, o.build * (o.shoulders ?? 1)), sleeve = o.sleeveColor, shade = shadeOf(sleeve, 0.2, '#7a8fcf'), line = o.sleeveLine ?? (o.outfit === 'jacket' || o.outfit === 'blazer' ? shadeOf(sleeve, 0.65, '#08061a') : '#5a6590');
  const rolled = o.sleeve === 'rolled', skin = o.skin, skinShade = shadeOf(skin, 0.3, '#b86a82'), skinLine = shadeOf(skin, 0.65, '#5a2a38'), w = o.armWidth ?? 1;
  const at = (k) => [lerp(a.elbow[0], a.wrist[0], k), lerp(a.elbow[1], a.wrist[1], k)];
  const upperArm = limbPath(a.shoulder, a.elbow, 26 * w, 21 * w);
  const cut = rolled ? 0.4 : 1, mid = at(cut);
  const foreSleeve = limbPath(a.elbow, mid, 21 * w, (rolled ? 19 : 15.5) * w);
  let out = `<circle cx="${n1(a.shoulder[0])}" cy="${n1(a.shoulder[1] + 4)}" r="${n1(28 * w)}" fill="${line}"/>`;
  out += `<path d="${upperArm}" fill="${line}" stroke="${line}" stroke-width="3.4" stroke-linejoin="round"/>`;
  if (rolled) {
    const bare = limbPath(mid, a.wrist, 17 * w, 14 * w);
    out += `<path d="${bare}" fill="${skinLine}" stroke="${skinLine}" stroke-width="3" stroke-linejoin="round"/>` + cel(bare, skin, skinShade, { sw: 0, shift: 7 });
  }
  out += `<path d="${foreSleeve}" fill="${line}" stroke="${line}" stroke-width="3.4" stroke-linejoin="round"/>`;
  out += `<circle cx="${n1(a.shoulder[0])}" cy="${n1(a.shoulder[1] + 4)}" r="${n1(26 * w)}" fill="${sleeve}"/>` + cel(upperArm, sleeve, shade, { sw: 0, shift: 9 }) + cel(foreSleeve, sleeve, shade, { sw: 0, shift: 8 });
  const bend = Math.abs(upper - fore);
  if (bend > 14) {  // creases at the inside of a bent elbow
    const e = a.elbow;
    out += `<path d="M${n1(e[0] - side * 6)} ${n1(e[1] - 8)} C${n1(e[0])} ${n1(e[1] - 2)} ${n1(e[0] + side * 2)} ${n1(e[1] + 4)} ${n1(e[0] + side * 8)} ${n1(e[1] + 10)} M${n1(e[0] - side * 12)} ${n1(e[1] + 2)} C${n1(e[0] - side * 6)} ${n1(e[1] + 8)} ${n1(e[0] - side * 4)} ${n1(e[1] + 12)} ${n1(e[0] - side * 2)} ${n1(e[1] + 18)}" fill="none" stroke="${shade}" stroke-width="2" stroke-linecap="round" opacity="0.8"/>`;
  }
  if (rolled) {
    const fold = bandPath(at(cut - 0.1), at(cut + 0.03), 21.5 * w);
    out += cel(fold, shadeOf(sleeve, 0.08, '#ffffff'), shadeOf(sleeve, 0.3, '#3a3a5a'), { sw: 1.6, stroke: line, shift: 6 });
  } else {
    out += `<path d="${limbPath(at(0.84), a.wrist, 17.5 * w, 15.5 * w)}" fill="${o.cuff}" stroke="${line}" stroke-width="1.4"/>`;
  }
  const handSkin = o.gloves ?? skin;
  out += `<g transform="translate(${n1(a.wrist[0])} ${n1(a.wrist[1])}) rotate(${n1(a.frame)}) scale(${o.handScale ?? 1.16})">${handMarkup(side, gesture, handSkin)}${o.holds?.[side] ? `<g transform="scale(${n1(100 / (o.handScale ?? 1.16)) / 100})">${o.holds[side]}</g>` : ''}</g>`;
  return out;
}

// Boots, sneakers and loafers, drawn on the ground at the foot that stands at `ankleX`; the toe points outwards.
function shoeMarkup(side, kind, colour) {
  const dark = shadeOf(colour, 0.35, '#0a0618'), edge = shadeOf(colour, 0.7, '#08061a');
  const sole = (c) => `<path d="M${side * 22} -4 C${side * 50} 2 ${side * 78} 0 ${side * 85} -6" fill="none" stroke="${c}" stroke-width="3" opacity="0.85" stroke-linecap="round"/>`;
  if (kind === 'boot') {
    const shaft = cel(`M${side * 18} -128 L${side * 56} -128 L${side * 54} -48 L${side * 18} -46 Z`, colour, dark, { sw: 1.8, stroke: edge, shift: 8 });
    const foot = cel(`M${side * 20} -50 C${side * 16} -26 ${side * 20} -6 ${side * 28} 0 C${side * 58} 4 ${side * 88} 2 ${side * 90} -8 C${side * 86} -24 ${side * 60} -32 ${side * 54} -52 Z`, colour, dark, { sw: 1.8, stroke: edge, shift: 8 });
    const laces = [-112, -98, -84, -70].map((y) => `<path d="M${side * 26} ${y} L${side * 48} ${y - 4}" stroke="#e8d9bd" stroke-width="2.6" stroke-linecap="round"/>`).join('');
    return shaft + foot + laces + `<path d="M${side * 18} -128 L${side * 56} -128" stroke="${lightOf(colour, 0.35)}" stroke-width="8" stroke-linecap="round"/>` +
      `<path d="M${side * 24} -2 C${side * 54} 4 ${side * 84} 2 ${side * 90} -6" fill="none" stroke="#2a2018" stroke-width="5" stroke-linecap="round"/>`;
  }
  if (kind === 'loafer') {
    return cel(`M${side * 20} -40 C${side * 18} -22 ${side * 22} -6 ${side * 30} 0 C${side * 54} 4 ${side * 80} 2 ${side * 84} -6 C${side * 80} -17 ${side * 62} -22 ${side * 54} -42 Z`, colour, dark, { sw: 1.8, stroke: edge, shift: 7 }) +
      `<path d="M${side * 34} -14 C${side * 52} -20 ${side * 68} -16 ${side * 76} -9" fill="none" stroke="${lightOf(colour, 0.45)}" stroke-width="2.4" opacity="0.75" stroke-linecap="round"/>` +
      `<path d="M${side * 24} -2 C${side * 54} 4 ${side * 80} 2 ${side * 84} -6" fill="none" stroke="#120e18" stroke-width="3.6" stroke-linecap="round"/>`;
  }
  return cel(`M${side * 20} -48 C${side * 16} -26 ${side * 20} -6 ${side * 28} 0 C${side * 56} 4 ${side * 84} 2 ${side * 86} -8 C${side * 82} -22 ${side * 58} -30 ${side * 52} -50 Z`, colour, dark, { sw: 1.8, stroke: edge, shift: 8 }) + sole('#fff') +
    `<path d="M${side * 30} -26 L${side * 50} -34 M${side * 28} -18 L${side * 52} -26" stroke="#fff" stroke-width="2.4" stroke-linecap="round" opacity="0.8"/>`;
}

// A leg. `stance` 'left' or 'right' puts the weight on that leg: the other knee relaxes and its foot turns out; 'wide' stands with the feet apart.
function legMarkup(side, o) {
  const leg = o.trousers, shade = shadeOf(leg, 0.28, '#7a5a9a'), line = shadeOf(leg, 0.7, '#08061a'), k = Math.pow(o.build, 0.7), stance = o.stance ?? 'even';
  const free = (stance === 'left' && side === 1) || (stance === 'right' && side === -1), wide = stance === 'wide' ? 16 : 0;
  const hip = [side * 34 * k, HIP_Y + (free ? 5 : 0)], knee = [side * (36 * k + (free ? 13 : 0) + wide * 0.5), KNEE_Y + (free ? 3 : 0)], ankle = [side * (37 * k + (free ? 22 : 0) + wide), ANKLE_Y];
  const thigh = limbPath(hip, knee, 33, 24), shin = limbPath(knee, ankle, 24, 17);
  let out = `<path d="${thigh}" fill="${line}" stroke="${line}" stroke-width="3.4" stroke-linejoin="round"/><path d="${shin}" fill="${line}" stroke="${line}" stroke-width="3.4" stroke-linejoin="round"/>`;
  out += cel(thigh, leg, shade, { sw: 0, shift: 9 }) + cel(shin, leg, shade, { sw: 0, shift: 8 });
  out += `<g transform="translate(${n1(ankle[0] - side * 37)} 0)">${shoeMarkup(side, o.shoeKind ?? 'sneaker', o.shoe ?? '#3b82f6')}</g>`;
  return out;
}

// -- outfits ---------------------------------------------------------------------------------------------------------------------------

// The garment worn over the shirt: the open lab coat, a work jacket or a blazer; each returns the markup of the two front panels with their details.
function outerLayer(o) {
  const kind = o.outfit ?? 'coat', c = o.outer ?? o.coat, shade = shadeOf(c, 0.22, kind === 'coat' ? '#7a8fcf' : '#10101a'), line = kind === 'coat' ? '#5a6590' : shadeOf(c, 0.65, '#08061a');
  const fold = shadeOf(c, 0.28, kind === 'coat' ? '#7a8fcf' : '#10101a');
  if (kind === 'jacket') {
    const left = 'M-100 -726 C-80 -754 -42 -752 -22 -740 L-26 -436 C-54 -424 -90 -430 -104 -446 C-98 -530 -78 -604 -76 -654 C-76 -692 -88 -706 -100 -726 Z';
    const right = 'M100 -726 C80 -754 42 -752 22 -740 L26 -436 C54 -424 90 -430 104 -446 C98 -530 78 -604 76 -654 C76 -692 88 -706 100 -726 Z';
    let out = cel(left, c, shade, { sw: 1.8, shift: 14, stroke: line }) + cel(right, c, shade, { sw: 1.8, shift: 14, stroke: line });
    const hem = (s) => cel(`M${s * 26} -462 L${s * 27} -436 C${s * 54} -424 ${s * 90} -430 ${s * 104} -446 L${s * 101} -474 C${s * 80} -462 ${s * 50} -458 ${s * 26} -462 Z`, shadeOf(c, 0.2, '#10101a'), shade, { sw: 1.4, stroke: line, shift: 4 });
    out += hem(-1) + hem(1);
    const pocket = (s) => `<path d="M${s * 36} -690 L${s * 74} -688 L${s * 72} -634 L${s * 36} -636 Z" fill="${shadeOf(c, 0.1, '#ffffff')}" stroke="${line}" stroke-width="1.8"/><path d="M${s * 36} -690 L${s * 74} -688 L${s * 73} -668 L${s * 36} -670 Z" fill="${shade}" stroke="${line}" stroke-width="1.8"/><circle cx="${s * 55}" cy="-669" r="3.4" fill="#d9b84a" stroke="#5a4a1a" stroke-width="1"/>`;
    out += pocket(-1) + pocket(1);
    out += `<path d="M-24 -736 L-24 -450 M24 -736 L24 -450" fill="none" stroke="${shadeOf(c, 0.4, '#05030a')}" stroke-width="2.2" stroke-dasharray="7 5" opacity="0.6"/>`;
    out += `<path d="M-86 -716 L-52 -742 M86 -716 L52 -742" stroke="${shade}" stroke-width="9" stroke-linecap="round"/>`;
    out += cel('M-20 -742 L-62 -768 L-82 -724 L-34 -682 Z', shadeOf(c, 0.12, '#ffffff'), shade, { sw: 1.6, shift: 6, stroke: line }) + cel('M20 -742 L62 -768 L82 -724 L34 -682 Z', shadeOf(c, 0.12, '#ffffff'), shade, { sw: 1.6, shift: 6, stroke: line });
    return out;
  }
  if (kind === 'blazer') {
    const left = 'M-98 -726 C-78 -752 -40 -750 -20 -738 L-32 -418 C-56 -410 -84 -414 -92 -432 C-84 -480 -64 -540 -66 -590 C-68 -650 -84 -690 -98 -726 Z';
    const right = 'M98 -726 C78 -752 40 -750 20 -738 L32 -418 C56 -410 84 -414 92 -432 C84 -480 64 -540 66 -590 C68 -650 84 -690 98 -726 Z';
    let out = cel(left, c, shade, { sw: 1.8, shift: 14, stroke: line }) + cel(right, c, shade, { sw: 1.8, shift: 14, stroke: line });
    out += cel('M-20 -738 L-58 -762 L-82 -726 L-36 -650 L-30 -690 Z', shadeOf(c, 0.12, '#ffffff'), shade, { sw: 1.6, shift: 6, stroke: line }) + cel('M20 -738 L58 -762 L82 -726 L36 -650 L30 -690 Z', shadeOf(c, 0.12, '#ffffff'), shade, { sw: 1.6, shift: 6, stroke: line });
    out += `<circle cx="22" cy="-600" r="5" fill="${shadeOf(c, 0.5, '#000')}" stroke="${line}" stroke-width="1.2"/><circle cx="24" cy="-520" r="5" fill="${shadeOf(c, 0.5, '#000')}" stroke="${line}" stroke-width="1.2"/>`;
    out += `<path d="M-84 -520 L-48 -516 M84 -520 L48 -516" stroke="${fold}" stroke-width="2.6" stroke-linecap="round"/>`;
    out += `<path d="M-66 -684 L-44 -682 L-46 -664 Z" fill="#ffffff" stroke="${line}" stroke-width="1.2"/>`;  // the pocket square
    return out;
  }
  // the open lab coat
  const left = 'M-96 -724 C-76 -750 -38 -748 -20 -736 L-30 -304 C-58 -290 -88 -296 -100 -318 C-92 -420 -66 -520 -67 -600 C-68 -650 -84 -690 -96 -724 Z';
  const right = 'M96 -724 C76 -750 38 -748 20 -736 L30 -304 C58 -290 88 -296 100 -318 C92 -420 66 -520 67 -600 C68 -650 84 -690 96 -724 Z';
  let out = cel(left, c, shade, { sw: 1.8, shift: 14, stroke: line }) + cel(right, c, shade, { sw: 1.8, shift: 14, stroke: line });
  out += `<path d="M-60 -640 C-60 -560 -66 -480 -82 -330 M60 -640 C60 -560 66 -480 82 -330 M-46 -470 C-50 -420 -56 -360 -62 -312 M46 -470 C50 -420 56 -360 62 -312" fill="none" stroke="${fold}" stroke-width="1.8" opacity="0.8"/>`;
  out += `<path d="M-84 -440 L-50 -438 L-48 -400 L-82 -400 Z M84 -440 L50 -438 L48 -400 L82 -400 Z" fill="none" stroke="${fold}" stroke-width="1.8"/>`;
  out += cel('M-20 -736 L-56 -760 L-78 -728 L-34 -672 Z', lightOf(c, 0.25), shade, { sw: 1.6, shift: 6, stroke: line }) + cel('M20 -736 L56 -760 L78 -728 L34 -672 Z', lightOf(c, 0.25), shade, { sw: 1.6, shift: 6, stroke: line });
  return out;
}

// Small things in the breast pocket.
function pocketProp(kind, o) {
  switch (kind) {
    case 'pipette': return `<path d="M-52 -664 L-60 -738" stroke="#f2c14e" stroke-width="9" stroke-linecap="round"/><path d="M-52 -664 L-60 -738" stroke="#fff3c4" stroke-width="2.6" stroke-linecap="round" opacity="0.8"/><rect x="-66" y="-752" width="12" height="16" rx="4" fill="#e5484d" stroke="#6a1020" stroke-width="1.4" transform="rotate(-8 -60 -744)"/>`;
    case 'pens': return `<path d="M-58 -664 L-64 -722 M-50 -664 L-52 -726 M-42 -664 L-40 -718" stroke-width="5" stroke-linecap="round" stroke="#3b82f6"/><path d="M-50 -664 L-52 -726" stroke="#e5484d" stroke-width="5" stroke-linecap="round"/><path d="M-42 -664 L-40 -718" stroke="#2d8f7b" stroke-width="5" stroke-linecap="round"/>`;
    case 'pen': return `<path d="M-50 -664 L-54 -724" stroke="#1d2238" stroke-width="5" stroke-linecap="round"/><path d="M-54 -724 L-55 -732" stroke="#d9b84a" stroke-width="3" stroke-linecap="round"/>`;
    case 'stamp': return `<rect x="-60" y="-700" width="22" height="14" rx="3" fill="#2e9e57" stroke="#14502a" stroke-width="1.4"/><rect x="-54" y="-714" width="10" height="16" rx="3" fill="#1f7a42" stroke="#14502a" stroke-width="1.4"/>`;
    default: return '';
  }
}

// Things worn at the neck: headphones around it, a lanyard with a card.
function neckProp(kind, o) {
  if (kind === 'headphones') {
    return `<path d="M-30 -766 C-52 -716 -40 -688 0 -682 C40 -688 52 -716 30 -766" fill="none" stroke="#2b2f40" stroke-width="9" stroke-linecap="round"/>` +
      `<rect x="-52" y="-726" width="24" height="42" rx="11" fill="#7a52d6" stroke="#2b2f40" stroke-width="3"/><rect x="28" y="-726" width="24" height="42" rx="11" fill="#7a52d6" stroke="#2b2f40" stroke-width="3"/>`;
  }
  if (kind === 'lanyard') {
    return `<path d="M-30 -758 L-4 -610 M30 -758 L4 -610" stroke="#2e9e57" stroke-width="7" stroke-linecap="round"/><rect x="-24" y="-624" width="48" height="62" rx="6" fill="#fff" stroke="#14502a" stroke-width="2.4"/><rect x="-16" y="-614" width="32" height="22" rx="3" fill="#cfe9d8"/><rect x="-16" y="-584" width="32" height="5" fill="#2e9e57"/><rect x="-16" y="-575" width="22" height="4" fill="#8ac6a0"/>`;
  }
  return '';
}

function torsoMarkup(o) {
  const top = o.top, trousers = o.trousers, line = '#5a6590', kind = o.outfit ?? 'coat', hem = kind === 'jacket' ? -436 : kind === 'blazer' ? -418 : -304;
  let out = '';
  out += cel('M-62 -586 L62 -586 L70 -440 L-70 -440 Z', trousers, shadeOf(trousers, 0.3, '#10102a'), { sw: 1.6, stroke: shadeOf(trousers, 0.7, '#08061a') });
  out += `<path d="M-62 -570 L62 -570" stroke="${shadeOf(trousers, 0.55, '#000')}" stroke-width="5"/><rect x="-9" y="-576" width="18" height="12" rx="2" fill="#d9b84a" stroke="#5a4a1a" stroke-width="1.2"/>`;
  // the shirt (or the hoodie): narrow at the waist, broad at the chest
  out += cel('M-48 -748 C-24 -736 24 -736 48 -748 L62 -716 L54 -582 L-54 -582 L-62 -716 Z', top, shadeOf(top, 0.3, '#101040'), { sw: 1.6, stroke: shadeOf(top, 0.75, '#08061a') });
  if (o.under === 'hoodie') {  // the hood lies behind the neck; two drawstrings hang from the collar; a pouch pocket crosses the belly
    out += cel('M-52 -756 C-30 -720 30 -720 52 -756 L60 -742 C30 -690 -30 -690 -60 -742 Z', shadeOf(top, 0.12, '#ffffff'), shadeOf(top, 0.3, '#101040'), { sw: 1.6, stroke: shadeOf(top, 0.75, '#08061a'), shift: 6 });
    out += `<path d="M-12 -706 L-14 -648 M12 -706 L14 -648" stroke="#f3f0ff" stroke-width="3.4" stroke-linecap="round"/><circle cx="-14" cy="-644" r="3.6" fill="#f3f0ff"/><circle cx="14" cy="-644" r="3.6" fill="#f3f0ff"/>`;
    out += `<path d="M-40 -610 L-30 -566 M40 -610 L30 -566 M-30 -566 L30 -566" fill="none" stroke="${shadeOf(top, 0.4, '#101040')}" stroke-width="2.4" stroke-linecap="round"/>`;
  } else if (kind !== 'jacket') {  // a collar for the shirt
    out += cel('M-30 -748 L-6 -738 L-20 -716 L-44 -734 Z', '#fbfbff', '#c9d2ea', { sw: 1.4, stroke: '#7a86b0', shift: 4 }) + cel('M30 -748 L6 -738 L20 -716 L44 -734 Z', '#fbfbff', '#c9d2ea', { sw: 1.4, stroke: '#7a86b0', shift: 4 });
  }
  if (o.neckwear === 'tie') out += cel('M-9 -742 L9 -742 L12 -650 L0 -626 L-12 -650 Z', o.tieColor, shadeOf(o.tieColor, 0.3, '#10102a'), { sw: 1.4, stroke: shadeOf(o.tieColor, 0.75, '#08061a') });
  out += outerLayer(o);
  if (kind === 'coat' && o.badge) out += `<rect x="-62" y="-664" width="26" height="17" rx="3" fill="#fff" stroke="${line}" stroke-width="1.3"/><rect x="-58" y="-659" width="11" height="3" fill="${o.badge}"/><rect x="-58" y="-653" width="17" height="2.2" fill="#9aa5c8"/>`;
  out += pocketProp(o.pocket, o);
  if (o.neckwear === 'scarf') out += cel('M-54 -758 C-20 -732 20 -732 54 -758 L60 -722 C20 -700 -20 -700 -60 -722 Z', o.tieColor, shadeOf(o.tieColor, 0.3, '#3a0a20'), { sw: 1.6, stroke: '#4a1020' }) +
    cel('M30 -724 L60 -726 L68 -650 L42 -656 Z', o.tieColor, shadeOf(o.tieColor, 0.3, '#3a0a20'), { sw: 1.6, stroke: '#4a1020' });
  out += neckProp(o.neckProp, o);
  return out;
}

function neckMarkup(skin) {
  return cel('M-22 -790 L22 -790 L24 -742 C10 -733 -10 -733 -24 -742 Z', skin, shadeOf(skin, 0.45, '#b86a82'), { sw: 1.6, shift: 10, stroke: shadeOf(skin, 0.7, '#4a2030') });
}

// A complete figure. o: skin, hair, hairStyle, beard, eye, outfit ('coat', 'jacket', 'blazer'), coat / outer (the garment), top (the shirt), trousers, shoe, shoeKind, sleeve ('long' or 'rolled'), gloves,
// acc[] (head accessories), neckwear, neckProp, pocket, tieColor, badge, cuff, build (body width), shoulders, height, stance, tilt, face {jaw, chin, cheek, ear}, eyeScale, eyeTilt, eyeSpacing, browWidth,
// nose, lines (age), blush, pose (a POSES key or {arms, hands, lean, head}), mood, mouth, talk, blink, look, holds {left, right} (markup in the hand frame), sway (hair motion in degrees), breathe (pixels),
// t (time in seconds, for the idle motion), phase, bust (draw only the upper body)
function person(o) {
  const pose = typeof o.pose === 'string' ? POSES[o.pose] : (o.pose ?? POSES.stand);
  o = { build: 1, blink: 0, skin: SKIN_DEFAULT, hair: '#4a3a6a', ...o };
  o.sleeveColor = o.sleeveColor ?? ((o.outfit === 'jacket' || o.outfit === 'blazer') ? o.outer : o.coat);
  const acc = o.acc ?? [], breathe = o.breathe ?? 0, sway = o.sway ?? 0, style = o.hairStyle ?? 'short', time = o.t ?? 0, ph = o.phase ?? 0;
  const idle = (k, f, p = 0) => k * Math.sin(time * f + ph + p);
  const head = `<g transform="translate(0 ${n1(HEAD.y + breathe * 0.6)}) rotate(${n1((o.headTilt ?? pose.head ?? 0) + (o.headExtra ?? 0) + idle(1.4, 0.7, 1))} 0 60) scale(${HEAD.k})">` +
    `<g transform="rotate(${n1(sway)} 0 -100)">${hairBack(style, o.hair)}</g>` + faceMarkup(o) +
    `<g transform="rotate(${n1(sway * 0.6)} 0 -100)">${hairFront(style, o.hair)}</g>` + acc.map((a) => headAccessory(a)).join('') + `</g>`;
  const arms = pose.arms, hands = pose.hands ?? ['relaxed', 'relaxed'], holds = o.holds ?? {};
  const swing = (i) => idle(1.6, 1.1, i * 1.7);
  const legs = o.bust ? '' : legMarkup(-1, o) + legMarkup(1, o);
  const upper = `<g transform="rotate(${n1((o.tilt ?? 0) + idle(0.5, 0.6))} 0 ${HIP_Y})"><g transform="scale(${o.build} 1)">${torsoMarkup(o)}</g>` +
    `<g transform="translate(0 ${n1(breathe * 0.4)})">${neckMarkup(o.skin)}</g>` +
    armMarkup(-1, arms[0][0] + swing(0), arms[0][1] + swing(0) * 1.3, hands[0], { ...o, holds: { [-1]: holds.left } }) + armMarkup(1, arms[1][0] + swing(1), arms[1][1] + swing(1) * 1.3, hands[1], { ...o, holds: { [1]: holds.right } }) + head + `</g>`;
  const body = `<g transform="translate(${n1(pose.lean ?? 0)} ${n1(breathe)})">` + legs + upper + `</g>`;
  const shadow = `<ellipse cx="0" cy="2" rx="${n1(120 * (o.shadowScale ?? 1))}" ry="14" fill="#1a1040" opacity="0.28" filter="${blur(7)}"/>`;
  const height = o.bust ? 1 : (o.height ?? 1);
  return `<g transform="${place(o.x ?? 0, o.y ?? 0, 0, (o.scale ?? 1) * (o.flip ?? 1) * height, (o.scale ?? 1) * height)}">${shadow}${body}</g>`;
}

// The cast: every member has his own build, stance, face, outfit and things he carries.
const CAST = {
  kai: { name: 'Kai', skin: '#ffe0cc', hair: '#1f4a55', hairStyle: 'messy', eye: '#2d8f7b', top: '#2d8f7b', coat: '#ffffff', trousers: '#2b3552', shoe: '#2d8f7b', shoeKind: 'sneaker', acc: ['goggles'], neckwear: null, cuff: '#7fd0be',
    badge: '#2d8f7b', build: 0.92, height: 0.95, stance: 'left', tilt: 2, sleeve: 'rolled', gloves: '#8fd0ff', pocket: 'pipette', blush: 0.35, eyeScale: 1.1, eyeSpacing: 1.04, nose: 0.85, face: { jaw: 0.9, chin: -4, cheek: 1.04 } },
  bits: { name: 'Bits', skin: '#f6d3b8', hair: '#c9772f', hairStyle: 'sidepart', eye: '#7a52d6', top: '#7a52d6', under: 'hoodie', coat: '#ffffff', trousers: '#3b2f6b', shoe: '#7a52d6', shoeKind: 'sneaker', acc: ['glasses'], neckProp: 'headphones',
    cuff: '#b9a6ef', badge: '#7a52d6', build: 0.86, height: 1.06, stance: 'right', tilt: -2.5, pocket: 'pens', eyeScale: 0.96, nose: 1.05, face: { jaw: 0.88, chin: 8, cheek: 0.97 }, browWidth: 5.4 },
  stacker: { name: 'Stacker', skin: '#e9b98f', hair: '#5a3b28', hairStyle: 'spiky', beard: 'stubble', eye: '#b0472e', top: '#e5484d', outfit: 'jacket', outer: '#3d4358', coat: '#3d4358', trousers: '#2f3a52', shoe: '#e5484d',
    shoeKind: 'sneaker', acc: ['headband'], neckwear: 'scarf', tieColor: '#e5484d', cuff: '#f08a8d', sleeve: 'rolled', build: 1.1, shoulders: 1.04, armWidth: 1.12, height: 1.0, stance: 'wide', tilt: 1.5, browWidth: 8.4, face: { jaw: 1.1, chin: 0, cheek: 1.02 },
    eyeScale: 0.95 },
  gatekeeper: { name: 'Gatekeeper', skin: '#fbe4d6', hair: '#c8d3e6', hairStyle: 'undercut', beard: 'goatee', eye: '#3b6fc7', top: '#f4f7ff', outfit: 'blazer', outer: '#243a63', coat: '#243a63', trousers: '#1f2a44', shoe: '#1b1722', shoeKind: 'loafer', acc: [],
    neckwear: 'tie', tieColor: '#2a5ab8', cuff: '#e8eefc', badge: '#3b82f6', build: 0.95, height: 1.07, stance: 'even', tilt: 0, lines: 0.9, pocket: 'pen', browWidth: 5.6, face: { jaw: 0.94, chin: 9, cheek: 0.96 }, eyeScale: 0.94, eyeTilt: 2 },
  dock: { name: 'Dock', skin: '#a8704c', hair: '#1c1620', hairStyle: 'short', beard: 'beard', eye: '#c7660a', top: '#f2994a', outfit: 'jacket', outer: '#7a7448', coat: '#7a7448', trousers: '#4a3554', shoe: '#6a4a2a', shoeKind: 'boot', acc: ['visor'],
    cuff: '#f6b878', sleeve: 'rolled', build: 1.14, shoulders: 1.05, armWidth: 1.14, height: 1.03, stance: 'wide', tilt: 1, browWidth: 8, face: { jaw: 1.12, chin: 2, cheek: 1.04 }, neckProp: null },
  auditor: { name: 'Auditor', skin: '#d8a27a', hair: '#26355d', hairStyle: 'crop', eye: '#2e9e57', top: '#f4f7ff', outfit: 'blazer', outer: '#2f4a45', coat: '#2f4a45', trousers: '#1f2a44', shoe: '#2a2230', shoeKind: 'loafer', acc: ['frames'],
    neckwear: 'tie', tieColor: '#1f7a42', cuff: '#e8eefc', neckProp: 'lanyard', build: 0.97, height: 0.97, stance: 'left', tilt: -1, pocket: 'stamp', blush: 0.15, face: { jaw: 1.0, chin: 0, cheek: 1.07 }, eyeScale: 1.0 },
  scout: { name: 'Scout', skin: '#f6d9c2', hair: '#e8c35a', hairStyle: 'messy', eye: '#8a5a2b', top: '#d6439a', coat: '#ffffff', trousers: '#2f3a52', shoe: '#d6439a', shoeKind: 'sneaker', acc: [], neckProp: null,
    cuff: '#efa6cf', badge: '#d6439a', build: 0.9, height: 0.98, stance: 'right', tilt: 2, blush: 0.2, eyeScale: 1.04, nose: 0.9, face: { jaw: 0.92, chin: 2, cheek: 1.0 } },
  foundry: { name: 'Foundry', skin: '#e7b894', hair: '#6b4a32', hairStyle: 'crop', beard: 'mustache', eye: '#6a4a2a', top: '#0891b2', coat: '#ffffff', trousers: '#2b3a46', shoe: '#5b4630', shoeKind: 'boot', acc: ['goggles'],
    cuff: '#7fd0e0', sleeve: 'rolled', badge: '#0891b2', build: 1.16, shoulders: 1.06, armWidth: 1.14, height: 0.98, stance: 'wide', tilt: -1, browWidth: 8, face: { jaw: 1.12, chin: 2, cheek: 1.05 } },
};
