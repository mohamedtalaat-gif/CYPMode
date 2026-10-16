'use strict';
// Sets, props and screen furniture: the laboratory, name plates, speech bubbles, stamps, boards, transitions and the final grading.

const FONT_TITLE = "'Avenir Next Condensed', 'Avenir Next', 'Helvetica Neue', Arial, sans-serif";

function lab(t, o = {}) {
  const tint = o.tint ?? '#7fa6e8', cam = o.cam ?? 0, warm = o.warm ?? 0.5;
  let s = `<rect width="${W}" height="${H}" fill="${linear([[0, mixHex('#f4f7ff', tint, 0.12)], [0.55, mixHex('#dbe6f8', tint, 0.22)], [1, mixHex('#bccbe6', tint, 0.32)]])}"/>`;
  // far layer: windows with sky, shelves with bottles, softly out of focus
  let far = '';
  for (let i = 0; i < 4; i++) {
    const x = 150 + i * 520 - cam * 0.25, w = 300;
    const sky = linear([[0, mixHex('#ffe1b5', '#bfe3ff', 1 - warm)], [0.6, '#cfe8ff'], [1, '#eaf4ff']]);
    far += `<rect x="${x}" y="70" width="${w}" height="520" rx="22" fill="${sky}" stroke="#ffffff" stroke-width="12"/>`;
    far += `<path d="M${x} 330 L${x + w} 330 M${x + w / 2} 70 L${x + w / 2} 590" stroke="#fff" stroke-width="10"/>`;
    far += `<path d="M${x + 20} 560 L${x + 90} 430 L${x + 140} 500 L${x + 210} 380 L${x + 280} 560 Z" fill="#9fb8e0" opacity="0.35"/>`;
    far += `<ellipse cx="${x + 90}" cy="190" rx="70" ry="22" fill="#fff" opacity="0.8"/><ellipse cx="${x + 140}" cy="176" rx="46" ry="20" fill="#fff" opacity="0.8"/>`;
  }
  s += `<g filter="${blur(2.5)}">${far}</g>`;
  // light shafts from the windows
  let shafts = '';
  for (let i = 0; i < 4; i++) {
    const x = 150 + i * 520 - cam * 0.25, drift = 14 * Math.sin(t * 0.5 + i);
    shafts += `<path d="M${x + 20} 560 L${x + 280} 560 L${x + 520 + drift} 1000 L${x + 180 + drift} 1000 Z" fill="${linear([[0, '#fff7da', 0.55], [1, '#fff7da', 0]])}"/>`;
  }
  s += `<g filter="${blur(14)}" opacity="${0.5 + 0.2 * warm}">${shafts}</g>`;
  // the bench and the shelves
  let bench = '';
  for (let i = -1; i < 9; i++) {
    const x = i * 260 - ((cam * 0.6) % 260);
    bench += `<rect x="${x}" y="760" width="250" height="190" rx="8" fill="${linear([[0, '#d4e2f7'], [1, '#aebfdc']])}" stroke="#8ea4cc" stroke-width="3"/><circle cx="${x + 108}" cy="850" r="6" fill="#8ea4cc"/><circle cx="${x + 142}" cy="850" r="6" fill="#8ea4cc"/>`;
    const kind = ((i % 5) + 5) % 5, bx = x + 125;
    if (kind === 0) bench += `<path d="M${bx - 22} 700 L${bx + 22} 700 L${bx + 18} 760 L${bx - 18} 760 Z" fill="#dff3ff" opacity="0.85" stroke="#8ea4cc" stroke-width="3"/><path d="M${bx - 20} 735 L${bx + 20} 735 L${bx + 18} 760 L${bx - 18} 760 Z" fill="#7fd1ae"/>`;
    if (kind === 1) bench += `<path d="M${bx - 10} 650 L${bx + 10} 650 L${bx + 10} 690 L${bx + 40} 760 L${bx - 40} 760 L${bx - 10} 690 Z" fill="#ffe3ec" stroke="#8ea4cc" stroke-width="3"/><path d="M${bx - 30} 735 L${bx + 30} 735 L${bx + 40} 760 L${bx - 40} 760 Z" fill="#f59aa6"/>`;
    if (kind === 2) bench += [-30, 0, 30].map((d) => `<rect x="${bx + d - 8}" y="690" width="16" height="70" rx="8" fill="#fff" opacity="0.85" stroke="#8ea4cc" stroke-width="3"/><rect x="${bx + d - 7}" y="725" width="14" height="33" rx="7" fill="${['#ffc857', '#9ec5ff', '#c9a7ff'][d / 30 + 1]}"/>`).join('');
    if (kind === 3) bench += `<rect x="${bx - 26}" y="728" width="52" height="32" rx="6" fill="#c9a27a" stroke="#8a6a44" stroke-width="3"/>` + [-0.7, -0.2, 0.3, 0.8].map((a) => `<ellipse cx="${bx + 44 * Math.sin(a)}" cy="${700 - 6 * Math.cos(a)}" rx="12" ry="34" fill="#4fb477" stroke="#2f7a4f" stroke-width="2" transform="rotate(${a * 40} ${bx} 728)"/>`).join('');
    if (kind === 4) bench += `<rect x="${bx - 42}" y="705" width="84" height="55" rx="6" fill="#e8eefb" stroke="#8ea4cc" stroke-width="3"/><rect x="${bx - 32}" y="714" width="64" height="30" rx="3" fill="#7fb7ff"/><path d="M${bx - 24} 738 L${bx - 8} 726 L${bx + 6} 734 L${bx + 24} 720" stroke="#fff" stroke-width="3" fill="none"/>`;
  }
  s += `<g filter="${blur(1.2)}">${bench}</g><rect x="-20" y="742" width="${W + 40}" height="26" rx="6" fill="${linear([[0, '#9bb4de'], [1, '#7b95c6']])}" stroke="#6b84b8" stroke-width="3"/>`;
  // the floor and its reflections
  s += `<rect x="0" y="948" width="${W}" height="${H - 948}" fill="${linear([[0, '#c3d1ea'], [1, '#9eb0d2']])}"/>`;
  for (let i = -6; i < 16; i++) s += `<path d="M${960 + (i * 220 - (cam * 0.9) % 220) - 960 * 0} 948 L${960 + (i * 220 - (cam * 0.9) % 220 - 960) * 1.8} ${H}" stroke="#fff" stroke-width="3" opacity="0.28"/>`;
  s += `<rect x="0" y="946" width="${W}" height="6" fill="#7b95c6" opacity="0.6"/>`;
  // dust in the light
  const r = rng(11);
  let dust = '';
  for (let i = 0; i < 34; i++) {
    const x = (r() * W + t * (6 + 20 * r())) % W, y = 120 + ((r() * 820 + t * (4 + 8 * r())) % 820), rr = 3 + 9 * r();
    dust += `<circle cx="${n1(x)}" cy="${n1(y)}" r="${n1(rr)}" fill="#fff" opacity="${n1(0.12 + 0.35 * r())}"/>`;
  }
  s += `<g filter="${blur(1.6)}">${dust}</g>`;
  return s;
}

// An anime-style backdrop for the introduction cards: a gradient, halftone dots and diagonal light bars in the character's colour.
function cardBackdrop(t, color, o = {}) {
  const dark = shadeOf(color, 0.78, '#0a0a22'), mid = shadeOf(color, 0.45, '#0a0a22');
  let s = `<rect width="${W}" height="${H}" fill="${linear([[0, dark], [0.6, mid], [1, shadeOf(color, 0.15, '#0a0a22')]], 0, 0, 1, 1)}"/>`;
  const dots = uid('p');
  def(`<pattern id="${dots}" width="26" height="26" patternUnits="userSpaceOnUse" patternTransform="rotate(30)"><circle cx="13" cy="13" r="5" fill="#fff" opacity="0.1"/></pattern>`);
  s += `<rect width="${W}" height="${H}" fill="url(#${dots})"/>`;
  for (let i = 0; i < 5; i++) {
    const x = -300 + i * 520 + ((t * 36) % 520), w = 130 + 60 * (i % 2);
    s += `<path d="M${x} ${H} L${x + w} ${H} L${x + w + 500} 0 L${x + 500} 0 Z" fill="${color}" opacity="0.12"/>`;
  }
  s += `<circle cx="${o.cx ?? 1400}" cy="${o.cy ?? 560}" r="520" fill="${radial([[0, lightOf(color, 0.4), 0.6], [1, color, 0]])}"/>`;
  return s;
}

function vignette() {
  return `<rect width="${W}" height="${H}" fill="${radial([[0.55, '#0a0a28', 0], [1, '#0a0a28', 0.5]], 0.5, 0.5, 0.75)}"/>`;
}




function stamp(label, x, y, color, p, size = 70, tilt = -8, sub = '') {
  if (p <= 0) return '';
  const drop = outBack(seg(p, 0, 0.3), 2.2), scale = lerp(2.6, 1, drop), a = clamp(p * 6);
  const wd = measure(label, size, { weight: 900, family: FONT_TITLE, spacing: 4 }) + 110;
  let s = `<g transform="translate(${x} ${y}) rotate(${tilt}) scale(${scale})" opacity="${a}">`;
  s += `<rect x="${-wd / 2}" y="${-size * 0.85}" width="${wd}" height="${size * 1.5 + (sub ? 34 : 0)}" rx="14" fill="#fff" fill-opacity="0.85" stroke="${color}" stroke-width="10"/>`;
  s += `<rect x="${-wd / 2 + 12}" y="${-size * 0.85 + 12}" width="${wd - 24}" height="${size * 1.5 + (sub ? 34 : 0) - 24}" rx="8" fill="none" stroke="${color}" stroke-width="3"/>`;
  s += text(label, 0, size * 0.05, size, { fill: color, weight: 900, family: FONT_TITLE, spacing: 4 });
  if (sub) s += text(sub, 0, size * 0.62, size * 0.38, { fill: color, weight: 800, spacing: 3 });
  return s + `</g>` + (seg(p, 0.28, 0.7) > 0 ? burstAt(x, y, seg(p, 0.28, 0.7), '#ffe58a', 190) : '');
}

function burstAt(x, y, u, color = '#fff', radius = 150) {
  if (u <= 0 || u >= 1) return '';
  let s = '';
  for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4 + 0.3, rr = radius * outCubic(u); s += sparkle(x + rr * Math.cos(a), y + rr * Math.sin(a), 26 * (1 - u) + 6, u * 2, color, 1 - u * 0.4); }
  return s + `<circle cx="${x}" cy="${y}" r="${n1(radius * 0.7 * outCubic(u))}" fill="none" stroke="${color}" stroke-width="${n1(8 * (1 - u))}" opacity="${n1(1 - u)}"/>`;
}

// A board with a title, drawn on the wall (a dark green chalkboard frame).
function board(x, y, w, h, o = {}) {
  let s = `<rect x="${x - 18}" y="${y - 18}" width="${w + 36}" height="${h + 36}" rx="22" fill="${linear([[0, '#d8b48a'], [1, '#a98355']])}" stroke="${INK}" stroke-width="4"/>`;
  s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${linear([[0, '#38705f'], [1, '#2a5a4c']])}" stroke="${INK}" stroke-width="3"/>`;
  s += `<path d="M${x + 30} ${y + h - 20} L${x + w - 30} ${y + h - 20}" stroke="#fff" stroke-width="3" opacity="0.18"/>`;
  return s;
}

function clipboard(x, y, s = 1, rotate = 0) {
  return `<g transform="translate(${x} ${y}) rotate(${rotate}) scale(${s})"><rect x="-46" y="-62" width="92" height="124" rx="8" fill="#c9a27a" stroke="${INK}" stroke-width="3.4"/><rect x="-36" y="-48" width="72" height="104" rx="4" fill="#fff" stroke="${INK}" stroke-width="2.4"/>` +
    [0, 1, 2, 3].map((i) => `<path d="M-26 ${-28 + i * 22} L26 ${-28 + i * 22}" stroke="#8a8fa3" stroke-width="3.4" stroke-linecap="round"/>`).join('') + `<rect x="-20" y="-72" width="40" height="18" rx="5" fill="#8a8fa3" stroke="${INK}" stroke-width="3"/></g>`;
}

function tabletIcon(x, y, w, h, color, label, size = 30) {
  return `<g><rect x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" rx="${h / 2}" fill="${linear([[0, lightOf(color, 0.35)], [1, color]])}" stroke="${INK}" stroke-width="3.4"/>` +
    `<path d="M${x - w / 2 + h * 0.4} ${y - h * 0.28} L${x + w / 2 - h * 0.4} ${y - h * 0.28}" stroke="#fff" stroke-width="5" opacity="0.5" stroke-linecap="round"/>` + (label ? text(label, x, y + size * 0.34, size, { fill: INK, weight: 800 }) : '') + `</g>`;
}

// The wipe between two scenes: diagonal colour blocks that cross the screen.
function wipe(p, colors = ['#e5484d', '#ffffff', '#3b82f6']) {
  if (p <= 0 || p >= 1) return '';
  let s = '';
  colors.forEach((c, i) => {
    const q = clamp((p - i * 0.12) / 0.7), x = lerp(-W * 1.6, W * 1.6, inOutCubic(q));
    s += `<path d="M${x - 700} ${H} L${x} ${H} L${x + 560} 0 L${x - 140} 0 Z" fill="${c}"/>`;
  });
  return s;
}

function grading(t, strength = 1) {
  return vignette() + `<rect width="${W}" height="${H}" fill="${linear([[0, '#ffd9a8', 0.1 * strength], [1, '#7a8fff', 0.08 * strength]])}"/>`;
}

// -- text that always fits ----------------------------------------------------------------------------------------------------------------

// The subtitle bar: the line is wrapped to at most two lines of at most 1500 px, on a translucent dark band sized to the text.
function subtitle(str, opacity = 1, o = {}) {
  if (!str || opacity <= 0) return '';
  const left = typeof CHARACTERS !== 'undefined' && !o.centered, cx = left ? 700 : W / 2, wide = left ? 1240 : 1500;  // the painted staff stands at the right edge: the line keeps to the left of him
  let size = 36, lines = wrap(str, wide, size, { weight: 700 });  // the bar sits under the feet of the vector people (they stand on y = 1012)
  if (lines.length > 2) { size = 31; lines = wrap(str, wide + 60, size, { weight: 700 }); }
  const lh = size * 1.28, width = Math.max(...lines.map((l) => measure(l, size, { weight: 700 }))) + 80, height = lines.length * lh + 20, y0 = 1072 - height;
  let s = `<g opacity="${n1(opacity * 1000) / 1000}"><rect x="${n1(cx - width / 2)}" y="${n1(y0)}" width="${n1(width)}" height="${n1(height)}" rx="${n1(height / 2.4)}" fill="#0c0a1e" fill-opacity="0.62"/>`;
  lines.forEach((l, i) => { s += text(l, cx, y0 + 10 + size * 0.95 + i * lh, size, { fill: '#fff', weight: 700 }); });
  return s + '</g>';
}

// A speech bubble: wraps its text to the given width and grows downward to hold it; (x, y) is the centre of the top edge of the body.
function bubble(str, x, y, w, tail = 'down', opacity = 1, size = 34) {
  const inner = w - 64, lines = wrap(str, inner, size, { weight: 700 }), lh = size * 1.3, h = lines.length * lh + 48, top = y;
  let s = `<g opacity="${opacity}"><rect data-box="1" x="${n1(x - w / 2)}" y="${n1(top)}" width="${w}" height="${n1(h)}" rx="34" fill="#fff" stroke="${INK}" stroke-width="5"/>`;
  if (tail === 'down') s += `<path d="M${x - 20} ${n1(top + h - 2)} L${x + 6} ${n1(top + h + 46)} L${x + 28} ${n1(top + h - 2)} Z" fill="#fff" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/><path d="M${x - 14} ${n1(top + h - 4)} L${x + 22} ${n1(top + h - 4)}" stroke="#fff" stroke-width="8"/>`;
  if (tail === 'left') s += `<path d="M${x - w / 2 + 2} ${n1(top + h / 2 - 16)} L${x - w / 2 - 44} ${n1(top + h / 2 + 8)} L${x - w / 2 + 2} ${n1(top + h / 2 + 24)} Z" fill="#fff" stroke="${INK}" stroke-width="5" stroke-linejoin="round"/><path d="M${x - w / 2 + 4} ${n1(top + h / 2 - 10)} L${x - w / 2 + 4} ${n1(top + h / 2 + 18)}" stroke="#fff" stroke-width="8"/>`;
  lines.forEach((l, i) => { s += text(l, x, top + 24 + size * 0.95 + i * lh, size, { fill: INK, weight: 700, maxWidth: inner }); });
  return { markup: s + '</g>', height: h };
}

// A panel (white card with a coloured title bar); the title is fitted into the bar. Returns the markup; the content area starts at y + 76.
function panel(x, y, w, h, title, color, o = {}) {
  let s = `<g opacity="${o.opacity ?? 1}"><rect data-box="1" x="${x}" y="${y}" width="${w}" height="${h}" rx="30" fill="${o.fill ?? '#fff'}" fill-opacity="${o.fillOpacity ?? 0.96}" stroke="${color}" stroke-width="8"/>`;
  s += `<path d="M${x} ${y + 76} L${x} ${y + 30} Q${x} ${y} ${x + 30} ${y} L${x + w - 30} ${y} Q${x + w} ${y} ${x + w} ${y + 30} L${x + w} ${y + 76} Z" fill="${color}"/>`;
  s += text(title, x + w / 2, y + 52, 38, { fill: '#fff', weight: 900, spacing: 2, maxWidth: w - 60 });
  return s + '</g>';
}

// A pill with a label, sized by measuring the label.
function pill(label, x, y, size, color, o = {}) {
  const w = measure(label, size, { weight: o.weight ?? 800 }) + size * 1.2, h = size * 1.55;
  if (o.anchor === 'start') x += w / 2;  // x is then the left edge
  return `<g opacity="${o.opacity ?? 1}"><rect data-box="1" x="${n1(x - w / 2)}" y="${n1(y - h / 2)}" width="${n1(w)}" height="${n1(h)}" rx="${n1(h / 2)}" fill="${o.fill ?? '#0f1b2d'}" fill-opacity="${o.fillOpacity ?? 0.9}" stroke="${color}" stroke-width="3"/>` +
    text(label, x, y + size * 0.34, size, { fill: o.color ?? '#fff', weight: o.weight ?? 800 }) + '</g>';
}

// A real picture (a PNG rendered from coordinates), cropped to a rounded frame, with a slow push-in (zoom z0 to z1 over progress p, panning towards (fx, fy) in 0..1).
function realImage(href, x, y, w, h, p = 0, o = {}) {
  const z = lerp(o.z0 ?? 1.0, o.z1 ?? 1.12, smooth(p)), id = uid('c'), img = ASSET_SIZES[href] ?? [1800, 1100];
  const base = Math.max(w / img[0], h / img[1]), dw = img[0] * base * z, dh = img[1] * base * z;
  const fx = o.fx ?? 0.5, fy = o.fy ?? 0.5, ix = x + (w - dw) * fx, iy = y + (h - dh) * fy;
  def(`<clipPath id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.rx ?? 26}"/></clipPath>`);
  return `<g opacity="${o.opacity ?? 1}"><g clip-path="url(#${id})"><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#0d1117"/><image href="${href}" x="${n1(ix)}" y="${n1(iy)}" width="${n1(dw)}" height="${n1(dh)}" preserveAspectRatio="none"/></g>` +
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.rx ?? 26}" fill="none" stroke="${o.stroke ?? '#fff'}" stroke-width="${o.strokeWidth ?? 6}"/></g>`;
}


// Animated pictures. A clip is a sequence of frames rendered from coordinates (blender_clips.py writes assets_clips.js); clipFrame picks the frame of the local time s and registers it so that
// renderAsync loads it before the frame is drawn. A clip that does not exist gives null and the callers fall back to the still picture.
const WANTED_FRAMES = new Set();
function clipFrame(key, s) {
  const c = typeof CLIPS === 'undefined' ? null : CLIPS[key];
  if (!c) return null;
  const i = Math.max(0, Math.min(c.frames - 1, Math.floor(s * c.fps))), href = c.pattern.replace('%04d', String(i + 1).padStart(4, '0'));
  ASSET_SIZES[href] = c.size;
  WANTED_FRAMES.add(href);
  return { href, index: i, clip: c };
}

// realImage of the frame of a clip at local time s (the still picture `fallback` when the clip is not there).
function clipImage(key, fallback, s, x, y, w, h, p = 0, o = {}) {
  const f = clipFrame(key, s);
  return realImage(f ? f.href : fallback, x, y, w, h, p, f ? { z0: 1.0, z1: 1.0, ...o } : o);
}

// Where a picture drawn by realImage puts the pixel (px, py) of the original image (so labels can follow the structure).
function imagePoint(href, px, py, x, y, w, h, p = 0, o = {}) {
  const z = lerp(o.z0 ?? 1.0, o.z1 ?? 1.12, smooth(p)), img = ASSET_SIZES[href] ?? [1800, 1100], base = Math.max(w / img[0], h / img[1]), dw = img[0] * base * z, dh = img[1] * base * z;
  return [x + (w - dw) * (o.fx ?? 0.5) + px / img[0] * dw, y + (h - dh) * (o.fy ?? 0.5) + py / img[1] * dh];
}

// The character introduction card: a coloured ribbon with the member number, the name, the role, what the member does (wrapped) and what the member stands for in the project.
function plate(o, p) {
  const e = outCubic(seg(p, 0, 0.12)), color = o.color;
  let s = `<g transform="translate(${n1(-1000 * (1 - e))} 0)"><path d="M0 150 L1180 150 L1060 330 L0 330 Z" fill="${color}" opacity="0.96"/><path d="M0 120 L1220 120 L1180 150 L0 150 Z" fill="#fff" opacity="0.95"/>`;
  s += text(o.index, 100, 100, 32, { fill: '#fff', anchor: 'start', weight: 700, spacing: 8, opacity: 0.9 });
  s += text(o.name.toUpperCase(), 90, 305, 190, { fill: '#fff', stroke: shadeOf(color, 0.7, '#0a0a22'), strokeWidth: 16, anchor: 'start', italic: true, weight: 900, family: FONT_TITLE, maxWidth: 1000 }) + '</g>';
  const e2 = outCubic(seg(p, 0.03, 0.15));
  s += `<g transform="translate(${n1(-1000 * (1 - e2))} 0)"><path d="M0 350 L980 350 L930 410 L0 410 Z" fill="#fff"/>` + text(o.role.toUpperCase(), 100, 396, 40, { fill: shadeOf(color, 0.35, '#0a0a22'), anchor: 'start', weight: 900, spacing: 5, maxWidth: 790 }) + '</g>';
  let y = 500;
  o.lines.forEach((line, i) => {
    const a = seg(p, 0.12 + i * 0.1, 0.2 + i * 0.1), para = paragraph(line, 100 + 40 * (1 - outCubic(a)), y, 42, { fill: '#fff', anchor: 'start', weight: 600, stroke: shadeOf(color, 0.8, '#0a0a22'), strokeWidth: 6, opacity: a, maxWidth: 940, lineHeight: 54 });
    s += para.markup; y += para.height + 14;
  });
  if (o.represents) {
    const a = seg(p, 0.52, 0.62), label = o.represents.toUpperCase();
    s += `<g opacity="${a}">` + text('IN THE PROJECT', 100, y + 70, 26, { fill: lightOf(color, 0.55), anchor: 'start', weight: 800, spacing: 6 }) + pill(label, 100 + (measure(label, 34) + 44) / 2, y + 128, 34, color, { fill: '#ffffff', color: shadeOf(color, 0.4, '#0a0a22'), fillOpacity: 0.96 }) + '</g>';
  }
  return s;
}
