'use strict';
// The non-human characters: Cap (the capsule), Quanta (an orbital spirit), the enzyme and the two co-folding assistants.

// A pair of anime eyes and a mouth for a round character; (x, y) is the centre of the face, size scales the eyes.
function roundBrows(mood = 'smile') {
  const tilt = { smile: 1, happy: 3, proud: -1, worry: 11, shout: -9, angry: -13, calm: 0, surprised: 4, think: 5, sad: 12 }[mood] ?? 0;
  const lift = { surprised: -12, shout: -4, happy: -3, worry: 2, sad: 3 }[mood] ?? 0;
  const brow = (s) => taper([[s * 24, -46 - tilt * 0.9 + lift], [s * 42, -53 + lift - tilt * 0.15], [s * 62, -47 + tilt * 0.5 + lift]], 7, 3, INK, { bulge: 0.25 });
  return brow(-1) + brow(1);
}

function roundFace(x, y, size, o) {
  const eyes = eyeShape(-1, { ...o, eyeScale: 1, spacing: 1.32 }) + eyeShape(1, { ...o, eyeScale: 1, spacing: 1.32 }) + roundBrows(o.mood);
  return `<g transform="translate(${n1(x)} ${n1(y)}) scale(${size})">${eyes}` +
    `<ellipse cx="-44" cy="36" rx="15" ry="8.5" fill="#ff6f86" opacity="0.22" filter="${blur(2)}"/><ellipse cx="44" cy="36" rx="15" ry="8.5" fill="#ff6f86" opacity="0.22" filter="${blur(2)}"/>` +
    `<g transform="translate(0 56)">${mouthShape(o.mouth ?? 'smile', o.talk ?? 0)}</g></g>`;
}

// Cap: o = {x, y (centre of the body), scale, angle (roll, degrees), squash, mood, mouth, talk, blink, look, arms [[deg, deg], [deg, deg]], legs (true/false), glow}
function cap(o) {
  const L = 360, R = 118, half = L / 2 - R, sq = o.squash ?? 0;
  const bodyPath = `M${-half} ${-R} L${half} ${-R} C${half + R * 1.1} ${-R} ${half + R * 1.1} ${R} ${half} ${R} L${-half} ${R} C${-half - R * 1.1} ${R} ${-half - R * 1.1} ${-R} ${-half} ${-R} Z`;
  const leftHalf = `M0 ${-R} L${-half} ${-R} C${-half - R * 1.1} ${-R} ${-half - R * 1.1} ${R} ${-half} ${R} L0 ${R} Z`;
  const rightHalf = `M0 ${-R} L${half} ${-R} C${half + R * 1.1} ${-R} ${half + R * 1.1} ${R} ${half} ${R} L0 ${R} Z`;
  const red = linear([[0, '#ff7b7f'], [0.5, '#e5484d'], [1, '#b82f3a']]), white = linear([[0, '#ffffff'], [0.55, '#eef1fb'], [1, '#c2c9e6']]);
  let out = '';
  if (o.legs !== false) {
    for (const s of [-1, 1]) {
      out += `<g transform="translate(${s * 62} ${R - 8})"><path d="M-14 0 L14 0 L12 56 L-12 56 Z" fill="#fff" stroke="${INK}" stroke-width="2.4"/>` +
        `<path d="M-13.5 22 L13.5 22 L13 30 L-13 30 Z" fill="#3b82f6" opacity="0.9"/><path d="M-12.5 36 L12.5 36 L12 42 L-12 42 Z" fill="#3b82f6" opacity="0.9"/>` +
        `<path d="M-22 56 C-22 44 22 44 22 56 L28 70 C12 78 -12 78 -28 70 Z" fill="#3b82f6" stroke="${INK}" stroke-width="2.4"/>` +
        `<path d="M-27 68 C-12 76 12 76 27 68 L28 74 C12 82 -12 82 -28 74 Z" fill="#fff" stroke="${INK}" stroke-width="2"/>` +
        `<path d="M-10 52 L10 52 M-11 58 L11 58" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/><path d="M-4 49 C0 45 4 45 8 49" fill="none" stroke="#fff" stroke-width="2.2"/></g>`;
    }
  }
  const arms = o.arms ?? [[40, 70], [40, 70]];
  for (const s of [-1, 1]) {
    const a = arms[s < 0 ? 0 : 1], sx = s * (L / 2 - 20), sy = 8, e = [sx + s * Math.sin(rad(a[0])) * 52, sy + Math.cos(rad(a[0])) * 52], h = [e[0] + s * Math.sin(rad(a[1])) * 46, e[1] + Math.cos(rad(a[1])) * 46];
    out += `<path d="${limbPath([sx, sy], e, 14, 12)}" fill="#fff" stroke="${INK}" stroke-width="2.4"/><path d="${limbPath(e, h, 12, 11)}" fill="#fff" stroke="${INK}" stroke-width="2.4"/>` +
      `<g transform="translate(${n1(h[0])} ${n1(h[1])}) rotate(${n1(s * (a[0] + a[1]) * 0.5)})"><path d="M-14 -4 L14 -4 L13 6 L-13 6 Z" fill="#3b82f6" stroke="${INK}" stroke-width="2.2" transform="translate(0 -12)"/>` +
      `<circle cx="0" cy="6" r="17" fill="#fff" stroke="${INK}" stroke-width="2.4"/><ellipse cx="${s * 14}" cy="2" rx="7" ry="9" fill="#fff" stroke="${INK}" stroke-width="2.2" transform="rotate(${s * 24} ${s * 14} 2)"/>` +
      `<path d="M-9 8 C-3 13 5 13 9 8" fill="none" stroke="${shadeOf('#c2c9e6', 0.2)}" stroke-width="2" opacity="0.7"/></g>`;
  }
  out += `<g clip-path="url(#${(() => { const id = uid('c'); def(`<clipPath id="${id}"><path d="${bodyPath}"/></clipPath>`); return id; })()})">` +
    `<path d="${leftHalf}" fill="${red}"/><path d="${rightHalf}" fill="${white}"/>` +
    `<ellipse cx="0" cy="${R * 0.78}" rx="${L / 2}" ry="${R * 0.5}" fill="#1a1040" opacity="0.18"/>` +
    `<rect x="-6" y="${-R}" width="12" height="${2 * R}" fill="#1a1040" opacity="0.22"/>` +
    `<path d="M${-half - 70} ${-R * 0.55} C${-half} ${-R * 0.95} ${half} ${-R * 0.95} ${half + 70} ${-R * 0.55}" fill="none" stroke="#fff" stroke-width="14" opacity="0.55" stroke-linecap="round"/></g>`;
  out += `<path d="${bodyPath}" fill="none" stroke="${INK}" stroke-width="3.4" stroke-linejoin="round"/>`;
  out += roundFace(8, -4, 1.0, { eye: '#4c7bd9', blink: o.blink ?? 0, look: o.look, mood: o.mood ?? 'smile', mouth: o.mouth ?? 'smile', talk: o.talk ?? 0 });
  const glow = o.glow ? `<ellipse cx="0" cy="0" rx="${L * 0.7}" ry="${R * 1.9}" fill="${radial([[0, '#fff3b0', 0.85], [1, '#fff3b0', 0]])}" />` : '';
  const k = o.scale ?? 1;
  return `<g transform="translate(${n1(o.x)} ${n1(o.y)}) scale(${k * (1 + 0.12 * sq)} ${k * (1 - 0.14 * sq)}) rotate(${n1(o.angle ?? 0)})">${glow}${out}</g>`;
}

// Quanta: a glowing nucleus with three orbits and electrons.
function quanta(o) {
  const t = o.t ?? 0, k = o.scale ?? 1, r = 74;
  let back = '', front = '';
  const colors = ['#35b6e8', '#e86cc9', '#f2b230'];
  colors.forEach((c, i) => {
    const tilt = i * 60 + 20, speed = 1.6 + i * 0.5, ph = t * speed + i * 2.1;
    const ring = `<ellipse cx="0" cy="0" rx="190" ry="62" fill="none" stroke="${c}" stroke-width="7" opacity="0.9" transform="rotate(${tilt})"/>`;
    const ex = 190 * Math.cos(ph), ey = 62 * Math.sin(ph);
    const electron = `<g transform="rotate(${tilt})"><circle cx="${n1(ex)}" cy="${n1(ey)}" r="15" fill="${c}" stroke="#fff" stroke-width="3"/><circle cx="${n1(ex)}" cy="${n1(ey)}" r="30" fill="${c}" opacity="0.25"/></g>`;
    (Math.sin(ph) > 0 ? front : back);
    if (Math.sin(ph) > 0) front += electron; else back += electron;
    back += ring;
  });
  const halo = `<circle cx="0" cy="0" r="${r * 3.2}" fill="${radial([[0, '#9fe8ff', 0.8], [0.5, '#9fe8ff', 0.25], [1, '#9fe8ff', 0]], 0.5, 0.5, 0.5)}"/>`;
  const core = `<circle cx="0" cy="0" r="${r}" fill="${radial([[0, '#ffffff'], [0.6, '#fff3b0'], [1, '#f6c445']], 0.4, 0.35, 0.75)}" stroke="${INK}" stroke-width="3.4"/>` +
    roundFace(0, 4, 0.9, { eye: '#e86cc9', blink: o.blink ?? 0, mood: o.mood ?? 'smile', mouth: o.mouth ?? 'smile', talk: o.talk ?? 0 });
  let sparkles = '';
  for (let i = 0; i < 7; i++) {  // a few stars that twinkle around the orbits
    const a = i * 2.4 + 0.6, rr = 150 + (i % 3) * 50, tw = 0.5 + 0.5 * Math.sin(t * 3 + i * 1.7);
    sparkles += `<path d="${starPath(rr * Math.cos(a) * 1.25, rr * Math.sin(a) * 0.8, 9 + 8 * tw, 0.28, 0, 4)}" fill="#fff" opacity="${n1(0.35 + 0.55 * tw)}"/>`;
  }
  return `<g transform="translate(${n1(o.x)} ${n1(o.y)}) scale(${k})">${halo}${sparkles}${back}${core}${front}</g>`;
}

// The enzyme: a purple body with the heme in its pocket. o = {x, y, scale, t, mood, mouth, alert, glow}
function enzyme(o) {
  const t = o.t ?? 0, k = o.scale ?? 1, pts = [];
  for (let i = 0; i < 14; i++) { const a = 2 * Math.PI * i / 14, w = 1 + 0.06 * Math.sin(3 * a + t * 1.6) + 0.04 * Math.sin(5 * a - t * 2.1); pts.push([300 * w * Math.cos(a), 250 * w * Math.sin(a)]); }
  const bodyD = blobPath(pts), fill = linear([[0, '#b79cf0'], [0.55, '#8e6fd8'], [1, '#6a4fb8']]);
  let out = `<ellipse cx="0" cy="${n1(250 * k > 0 ? 270 : 270)}" rx="300" ry="30" fill="#1a1040" opacity="0.25" filter="${blur(8)}"/>`;
  for (const s of [-1, 1]) {  // stubby feet and arms, behind the body
    out += `<ellipse cx="${s * 110}" cy="250" rx="62" ry="30" fill="#5b43a6" stroke="${INK}" stroke-width="4.4"/>`;
    const wave = (o.alert ? 12 * Math.sin(t * 7 + s) : 5 * Math.sin(t * 1.7 + s)), sh = [s * 268, 40], hand = [s * (352 + wave), 70 + wave * 1.5];
    out += `<path d="${limbPath(sh, hand, 30, 22)}" fill="#8e6fd8" stroke="${INK}" stroke-width="4.4"/><circle cx="${n1(hand[0])}" cy="${n1(hand[1])}" r="30" fill="#a98ae8" stroke="${INK}" stroke-width="4.4"/>`;
  }
  out += cel(bodyD, fill, '#4c3590', { shift: 26, sw: 5, shadeOpacity: 0.55 });
  out += `<ellipse cx="-110" cy="-120" rx="80" ry="40" fill="#fff" opacity="0.28" transform="rotate(-24 -110 -120)"/>`;
  for (const [hx, hy, hr] of [[-196, 10, 1], [200, -20, -1]]) {  // two alpha helices on the sides of the body: the protein's most recognisable shape
    let d = '';
    for (let k = 0; k <= 36; k++) { const u = k / 36, yy = hy - 90 + u * 180, xx = hx + hr * 20 * Math.sin(u * Math.PI * 6); d += `${k ? 'L' : 'M'}${n1(xx)} ${n1(yy)} `; }
    out += `<path d="${d}" fill="none" stroke="#d9c8ff" stroke-width="9" stroke-linecap="round" opacity="0.55"/><path d="${d}" fill="none" stroke="#6a4fb8" stroke-width="3" stroke-linecap="round" opacity="0.5"/>`;
  }
  out += `<ellipse cx="0" cy="118" rx="170" ry="84" fill="#46317f" stroke="${INK}" stroke-width="4"/>`;
  const glow = (o.glow ?? 0.6);
  out += `<ellipse cx="0" cy="118" rx="150" ry="70" fill="${radial([[0, '#ffb04a', 0.55 * glow], [1, '#ffb04a', 0]])}"/>`;
  out += hemeRing(0, 118, 1.0, o.t ?? 0);
  out += roundFace(0, -58, 1.5, { eye: '#4b2f9c', blink: o.blink ?? 0, mood: o.mood ?? 'worry', mouth: o.mouth ?? 'o', talk: o.talk ?? 0 });
  return `<g transform="translate(${n1(o.x)} ${n1(o.y)}) scale(${k})">${out}</g>`;
}

// The heme seen from above at an angle: a ring with four nitrogens and the iron at its centre.
function hemeRing(x, y, s, t = 0) {
  const ring = 'M-98 0 L-6 -46 L98 -6 L10 46 Z';
  return `<g transform="translate(${n1(x)} ${n1(y)}) scale(${s})"><path d="${ring}" fill="#d9485b" stroke="${INK}" stroke-width="4" stroke-linejoin="round"/>` +
    `<path d="M-52 0 L-4 -24 L52 -3 L6 24 Z" fill="#f59aa6" stroke="${INK}" stroke-width="3"/>` +
    [[-60, 0], [-4, -30], [56, -4], [6, 30]].map(([px, py]) => `<circle cx="${px}" cy="${py}" r="10" fill="#3b82f6" stroke="${INK}" stroke-width="2.6"/>`).join('') +
    `<circle cx="0" cy="0" r="23" fill="${radial([[0, '#ffd27a'], [1, '#ff9f1c']])}" stroke="${INK}" stroke-width="3.4"/>` +
    text('Fe', 0, 8, 22, { fill: INK, weight: 900 }) + `</g>`;
}

// A round co-folding assistant: Boltz-2 is bright and bouncy, OpenFold3 is grey and asleep.
function jelly(o) {
  const k = o.scale ?? 1, awake = o.awake !== false, c = awake ? '#7fb7ff' : '#b9c0d2';
  const w = 1 + 0.05 * Math.sin((o.t ?? 0) * 5), h = 1 - 0.05 * Math.sin((o.t ?? 0) * 5);
  const d = 'M-90 40 C-100 -50 -50 -92 0 -92 C50 -92 100 -50 90 40 C90 66 50 72 0 72 C-50 72 -90 66 -90 40 Z';
  let out = cel(d, linear([[0, lightOf(c, 0.4)], [1, c]]), shadeOf(c, 0.35, '#1a2a80'), { sw: 4, shift: 14 });
  out += `<ellipse cx="-34" cy="-52" rx="22" ry="11" fill="#fff" opacity="0.55" transform="rotate(-24 -34 -52)"/>`;
  const wave = awake ? 18 * Math.sin((o.t ?? 0) * 6) : 0;
  for (const s of [-1, 1]) out += `<ellipse cx="${s * 96}" cy="${n1(14 - (s > 0 ? wave : 0))}" rx="15" ry="22" fill="${lightOf(c, 0.25)}" stroke="${INK}" stroke-width="3.4" transform="rotate(${n1(s * (28 + (s > 0 ? wave : 0)))} ${s * 96} 14)"/>`;
  if (!awake) out += `<g opacity="${n1(0.55 + 0.4 * Math.sin((o.t ?? 0) * 1.6)) }">` + text('z', 72, -92 - 6 * Math.sin((o.t ?? 0) * 1.6), 34, { fill: '#6a7396', weight: 900 }) + text('z', 100, -122, 26, { fill: '#6a7396', weight: 900 }) + `</g>`;
  out += awake ? roundFace(0, -6, 0.9, { eye: '#2a52be', mood: 'happy', mouth: 'grin', blink: o.blink ?? 0 }) :
    `<path d="M-46 -8 C-34 2 -22 2 -10 -8 M10 -8 C22 2 34 2 46 -8" fill="none" stroke="${INK}" stroke-width="5" stroke-linecap="round"/><ellipse cx="0" cy="30" rx="8" ry="5" fill="#7a2e3a"/>`;
  return `<g transform="translate(${n1(o.x)} ${n1(o.y)}) scale(${k * w} ${k * h})">${out}</g>`;
}
