'use strict';
// The film: scenes on a timeline (seconds), each a function of its own local time s and the global time t. Every number on screen comes from FACTS or REAL (the project's own files),
// and every picture of a molecule or a pocket is drawn from coordinates (REAL.molecule, the PyMOL pictures of ASSETS).

const GROUND = 1012;
const CARD = 14;  // seconds of the introduction card of a member: the voice reads its lines
// The timeline in seconds: [scene id, length]. A member's scene is his introduction card (CARD seconds) followed by the story that he tells.
const TIMELINE = [['open', 30], ['kai', CARD + 24], ['scout', CARD + 24], ['bits', CARD + 24], ['stacker', CARD + 24], ['gate', 2 * CARD + 32], ['foundry', CARD + 26], ['dock', CARD + 38], ['auditor', CARD + 18], ['finale', 28]];
const SCENES = TIMELINE.reduce((list, [id, length]) => [...list, { id, start: list.length ? list[list.length - 1].end : 0, end: (list.length ? list[list.length - 1].end : 0) + length }], []);
const MEMBERS = 9;  // Kai, Scout, Bits, Stacker, Gatekeeper, Quanta, Foundry, Dock and Auditor
const MEMBER_TOTAL = String(MEMBERS).padStart(2, '0');
const DURATION = SCENES[SCENES.length - 1].end;

const ACCENT = { kai: '#2d8f7b', bits: '#7a52d6', stacker: '#e5484d', gatekeeper: '#3b82f6', quanta: '#e8a317', scout: '#d6439a', foundry: '#0891b2', dock: '#f2994a', auditor: '#2e9e57', cap: '#e5484d' };
const NAVY = '#16233a';

function blinkAt(t, phase = 0) { const b = (t + phase) % 3.4; return b < 0.15 ? Math.sin(b / 0.15 * Math.PI) : 0; }
const speakingAt = (t) => typeof SPEECH_ON === 'undefined' || SPEECH_ON.some(([a, b]) => t >= a && t < b);  // without a voice the staff talk whenever the scene says so
const mouthOpenAt = (t) => (typeof SPEECH_ENV === 'undefined' ? null : SPEECH_ENV.charAt(Math.floor(t * 24)) === '1');  // from the loudness of the voice, once per frame of the film
const talking = (t, on = true) => {
  if (!on) return 0;
  const open = mouthOpenAt(t);
  return open === null ? (speakingAt(t) ? 0.5 + 0.5 * Math.sin(t * 13) : 0) : (open ? 1 : 0);
};

// A member of the staff at the edge of the screen: the painted portrait when its pictures exist, the vector figure on the floor otherwise.
function staff(name, x, t, o = {}) {
  const phase = o.phase ?? (name.length * 0.7);
  const painted = paintedStaff(name, x + (o.paintedShift ?? 0), t, { blink: blinkAt(t, phase), talk: o.talk ?? 0, mood: o.mood, phase, scale: o.paintedScale ?? 1, flip: o.flip ?? 1, y: o.bust ? undefined : H });
  if (painted !== null) return painted;
  return person({ ...CAST[name], x, y: o.y ?? GROUND, scale: o.scale ?? 1.0, flip: o.flip ?? 1, pose: o.pose ?? 'stand', mood: o.mood ?? 'smile', mouth: o.mouth ?? 'smile', talk: o.talk ?? 0,
    blink: blinkAt(t, phase), sway: 2.2 * Math.sin(t * 1.5 + phase), breathe: 3.2 * Math.sin(t * 2.0 + phase), holds: o.holds, headExtra: o.headExtra ?? 0, look: o.look, bust: o.bust });
}

// A member drawn large from the waist up, for the introduction cards.
function bustOf(name, x, t, o = {}) {
  const scale = o.scale ?? 2.15, headY = o.headY ?? 400;
  const phase = o.phase ?? (name.length * 0.7);
  const painted = paintedStaff(name, x, t, { blink: blinkAt(t, phase), talk: o.talk ?? 0, mood: o.mood, phase, scale: 1.32, y: H + 120 });
  if (painted !== null) return painted;
  return staff(name, x, t, { ...o, scale, y: headY + 836 * scale, bust: true });
}

// The lines of subtitles of a scene: [[start, end, text], ...] in the scene's own seconds; each fades in and out over 0.4 s.
let CURRENT_TIME = 0;  // the global time of the frame being drawn (set by render), so that a line of speech knows where it sits in the film
function subs(s, list, o = {}) {
  if (typeof SPEECH_LOG !== 'undefined') for (const [a, b, str] of list) SPEECH_LOG.push({ kind: 'subtitle', a: CURRENT_TIME - s + a, b: CURRENT_TIME - s + b, text: str });  // the harvest of voices.py
  const origin = CURRENT_TIME - s;  // film time of this scene's (or segment's) local zero
  for (const [a, b, str] of list) {
    const spoken = typeof SPEECH_LINES === 'undefined' || !SPEECH_LINES[str] ? null : SPEECH_LINES[str].reduce((best, x) => (Math.abs(x[0] - (origin + a)) < Math.abs(best[0] - (origin + a)) ? x : best));  // [start, end] in film seconds when the line is spoken: the subtitle stays up for exactly that
    const from = spoken ? spoken[0] - 0.2 - origin : a, to = spoken ? spoken[1] + 0.45 - origin : b;
    if (s >= from && s < to) return subtitle(str, Math.min(seg(s, from, from + 0.4), 1 - seg(s, to - 0.4, to)), o);
  }
  return '';
}

// The card that introduces a member: the plate on the left, the member's portrait on the right.
function introCard(def, s, t, who) {
  if (typeof SPEECH_LOG !== 'undefined') SPEECH_LOG.push({ kind: 'card', name: def.name, role: def.role, start: t - s, lines: def.lines });
  const p = seg(s, 0, CARD);
  let body = cardBackdrop(t, def.color, { cx: 1420, cy: 560 });
  body += speedLines(1420, 620, 330, 1300, 36, lightOf(def.color, 0.5), 18, t * 0.05, 0.22);
  const slide = outBack(seg(s, 0.3, 1.1), 1.3);
  body += `<g transform="translate(${n1(760 * (1 - slide))} 0)">${who(1440, t)}</g>`;
  body += plate(def, p);
  const flash = seg(s, 0.5, 1.4);
  if (flash > 0 && flash < 1) body += burstAt(1380, 520, flash, '#fff', 260);
  return body;
}

// ---- opening ----------------------------------------------------------------------------------------------------------------------------

function sceneOpen(s, t) {
  const push = lerp(1.1, 1, smooth(seg(s, 0, 9)));
  let body = `<g transform="translate(960 540) scale(${n1(push * 1000) / 1000}) translate(-960 -540)">${lab(t, { warm: 1, tint: '#ffb48a', cam: 30 * s })}</g>`;
  const roll = seg(s, 1.4, 4.6), x = lerp(-420, 560, outCubic(roll)), land = seg(s, 4.5, 5.4), standing = roll >= 1;
  body += cap({ x, y: GROUND - 150, scale: 0.9, angle: standing ? 0 : (x - 560) * 0.5, squash: hump(land) * 0.8, mood: s < 12 ? 'happy' : 'worry', mouth: s < 12 ? 'grin' : 'worry', blink: blinkAt(t, 0.3),
    arms: [[30 + 20 * Math.sin(t * 3), 70], [30 - 20 * Math.sin(t * 3), 70]], legs: standing });
  const ep = outBack(seg(s, 6, 8), 1.3), settle = seg(s, 8, 9.5), ex = lerp(1450, 1560, settle), ey = lerp(600, 770, settle), es = lerp(0.78, 0.6, settle);
  if (ep > 0.02) body += enzyme({ x: ex, y: ey + (1 - ep) * 500, scale: es * ep, t, mood: s > 12 ? 'shout' : 'worry', mouth: s > 12 ? 'open' : 'o', talk: s > 12 ? talking(t) : 0, blink: blinkAt(t, 1.1), glow: 0.6 + 0.4 * Math.sin(t * 5) });
  const lens = seg(s, 8, 9.2) * (1 - seg(s, 15.5, 16.5));
  if (lens > 0.01) {
    body += `<g opacity="${lens}">` + clipImage('enzyme', ASSETS.enzyme.file, s - 8, 1010, 70, 820, 470, seg(s, 8, 16), { z0: 1.0, z1: 1.3, stroke: '#ffe58a', strokeWidth: 8, fx: 0.52, fy: 0.55 }) +
      pill('the real CYP3A4, drawn from its crystal structure', 1420, 574, 28, '#ffe58a', { opacity: lens }) + '</g>';
  }
  const tp = outBack(seg(s, 6.5, 7.8), 1.8);
  if (tp > 0) {
    body += `<g transform="translate(560 220) scale(${n1(tp * 1000) / 1000})">` + text('CYPMode', 0, 0, 200, { fill: '#fff', stroke: '#e5484d', strokeWidth: 28, weight: 900, italic: true, family: FONT_TITLE, spacing: 6 }) +
      text('CYPMode', 0, 0, 200, { fill: '#fff', stroke: INK, strokeWidth: 8, weight: 900, italic: true, family: FONT_TITLE, spacing: 6 }) + '</g>';
  }
  const tag = seg(s, 7.4, 8.4) * (1 - seg(s, 15.5, 16.5));
  body += text('predicting how drugs affect liver enzymes', 560, 310, 38, { fill: '#fff', stroke: INK, strokeWidth: 8, weight: 700, opacity: tag, maxWidth: 900 });
  const q = outBack(seg(s, 11, 12), 1.4) * (1 - seg(s, 17, 17.6));
  if (q > 0.01) { const b = bubble('Will I switch off a liver enzyme, and keep it off?', 640, 470 - 60 * (1 - q), 620, 'down', q); body += b.markup; }
  const c = outBack(seg(s, 17.6, 18.8), 1.3);
  if (c > 0.01) {
    const card = (x, color, title, big, small) => `<g transform="translate(${x} 640) scale(${n1(c * 1000) / 1000})"><rect data-box="1" x="-400" y="-190" width="800" height="380" rx="36" fill="#fff" stroke="${color}" stroke-width="10"/><path d="M-400 -104 L-400 -154 Q-400 -190 -364 -190 L364 -190 Q400 -190 400 -154 L400 -104 Z" fill="${color}"/>` +
      text(title, 0, -122, 42, { fill: '#fff', weight: 900, spacing: 3, maxWidth: 700 }) + text(big, 0, 28, 124, { fill: color, weight: 900, family: FONT_TITLE }) + paragraph(small, 0, 100, 38, { fill: INK, weight: 700, maxWidth: 700, lineHeight: 46 }).markup + '</g>';
    body += card(520, '#2d8f7b', 'TDI TRACK', String(FACTS.n_test), 'hidden compounds: does each one switch off the enzyme for good?');
    body += card(1400, '#f2994a', 'STRUCTURE TRACK', String(FACTS.poses), 'ligands: where does each one sit inside the enzyme?');
  }
  body += subs(s, [[3, 6.5, 'Meet Cap, a drug molecule.'], [8.5, 12.8, 'CYP3A4 is a liver enzyme that clears many medicines from the body.'], [12.8, 17.2, 'Will Cap switch it off, and keep it off?'],
    [18.2, 22.4, 'CYPMode answers this for two real tasks of the OpenADMET challenge.'], [22.4, 26, `${FACTS.n_test} hidden molecules to classify, and ${FACTS.poses} ligands to place in the enzyme.`]]);
  body += `<rect width="${W}" height="${H}" fill="#000" opacity="${n1(1 - seg(s, 0, 2))}"/>`;
  return body;
}

// ---- Kai: the labels --------------------------------------------------------------------------------------------------------------------

// Fraction of the enzyme's activity left at a dose with the given pIC50 (a Hill curve of slope 1); the dose axis is -log10(concentration), so a larger pIC50 lies further right.
const doseResponse = (x, pIC50) => 1 / (1 + Math.pow(10, x - pIC50));

function sceneKai(s, t) {
  const def = { color: ACCENT.kai, index: `MEMBER 01 / ${MEMBER_TOTAL}`, name: 'Kai', role: 'The assay analyst', lines: ['I run the assay. Every compound is measured twice: at once, and after a head start with the enzyme.', 'The shift between the two is the label.'], represents: 'the data and the label rule' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('kai', x, t, { pose: 'wave', mood: 'happy', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5) }));
  const u = s - CARD, m = REAL.molecule;
  let body = lab(t, { tint: '#6fc4b0', cam: 20 });
  body += panel(90, 120, 1230, 800, 'TWO MEASUREMENTS, ONE LABEL', ACCENT.kai);
  const fadeA = 1 - seg(u, 9.4, 10.4), fadeB = seg(u, 10, 11);
  if (fadeA > 0.01) {
    const left = 190, right = 800, top = 310, base = 700;
    let g = `<g opacity="${fadeA}">` + text('Cap is a real compound of the challenge: ' + m.name, 705, 232, 30, { fill: NAVY, weight: 700, maxWidth: 1130 });
    g += `<path d="M${left} ${top - 10} L${left} ${base} L${right} ${base}" stroke="${NAVY}" stroke-width="4" fill="none"/>` + text('dose of the compound (more to the right)', (left + right) / 2, base + 46, 26, { fill: '#5a6477', weight: 600, maxWidth: 600 }) +
      text('activity left', left + 10, top - 22, 26, { fill: '#5a6477', weight: 600, anchor: 'start', maxWidth: 220 });
    const curve = (pic, color, a) => {
      const pts = []; for (let i = 0; i <= 80 * a; i++) { const x = 3 + 4.4 * i / 80; pts.push([left + (right - left) * (x - 3) / 4.4, base - (base - top - 20) * doseResponse(x, pic)]); }
      return pts.length > 1 ? `<path d="M${pts.map((q) => `${n1(q[0])} ${n1(q[1])}`).join(' L')}" fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round"/>` : '';
    };
    g += curve(m.direct, '#2f80ff', seg(u, 1.5, 4.5)) + curve(m.tdi, '#e5484d', seg(u, 4.5, 7.5));
    g += pill('measured at once: pIC50 ' + m.direct.toFixed(2), 520, 330, 26, '#2f80ff', { opacity: seg(u, 1.8, 2.6), color: '#cfe3ff' });
    g += pill('after a head start: pIC50 ' + m.tdi.toFixed(2), 520, 392, 26, '#e5484d', { opacity: seg(u, 4.8, 5.6), color: '#ffd5d7' });
    const sx = 850, shift = m.tdi - Math.max(m.direct, 4);
    g += `<g opacity="${seg(u, 7, 8)}">` + paragraph('The two curves lie on top of each other. The shift is', sx, 340, 32, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: 420, lineHeight: 42 }).markup +
      text(shift.toFixed(3), sx + 210, 520, 96, { fill: ACCENT.kai, weight: 900, family: FONT_TITLE }) + paragraph('far below the line of 0.301 (the logarithm of 2).', sx, 590, 32, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: 420, lineHeight: 42 }).markup + '</g>';
    g += stamp('NOT TIME-DEPENDENT', 1010, 810, '#2f80ff', seg(u, 8, 9.4), 40, -6);
    body += g + '</g>';
  }
  if (fadeB > 0.01) {
    const left = 190, right = 930, top = 310, base = 770, x0 = 4, x1 = 7.4, y0 = 2.8, y1 = 7.8, X = (v) => left + (right - left) * (v - x0) / (x1 - x0), Y = (v) => base - (base - top) * (v - y0) / (y1 - y0);
    let g = `<g opacity="${fadeB}">` + text(`${REAL.scatter.n.toLocaleString('en-US')} real CYP3A4 compounds, each with both measurements`, 705, 232, 30, { fill: NAVY, weight: 700, maxWidth: 1130 });
    g += `<path d="M${left} ${top - 10} L${left} ${base} L${right} ${base}" stroke="${NAVY}" stroke-width="4" fill="none"/>` + text('measured at once (pIC50, not below 4)', (left + right) / 2, base + 46, 26, { fill: '#5a6477', weight: 600, maxWidth: 700 }) +
      text('after the head start', left + 10, top - 22, 26, { fill: '#5a6477', weight: 600, anchor: 'start', maxWidth: 260 });
    const n = Math.floor(REAL.scatter.points.length * outCubic(seg(u, 11.2, 17)));
    REAL.scatter.points.slice(0, n).filter(([d, tt]) => tt >= y0 && tt <= y1).forEach(([d, tt, lab]) => { g += `<circle cx="${n1(X(d))}" cy="${n1(Y(tt))}" r="5.5" fill="${lab ? '#e5484d' : '#2f80ff'}" fill-opacity="0.78" stroke="#fff" stroke-width="1.2"/>`; });
    const rule = seg(u, 17.2, 18.6);
    if (rule > 0) g += `<path d="M${n1(X(x0))} ${n1(Y(x0 + 0.301))} L${n1(lerp(X(x0), X(7.4), rule))} ${n1(lerp(Y(x0 + 0.301), Y(7.4 + 0.301), rule))}" stroke="${NAVY}" stroke-width="5" stroke-dasharray="14 9" fill="none"/>`;
    g += pill('time-dependent: above the line  t - max(d, 4) > log10 2', 640, 350, 24, '#e5484d', { opacity: seg(u, 17.8, 18.6), color: '#ffd5d7', fill: NAVY });
    g += `<circle cx="${n1(X(m.direct))}" cy="${n1(Y(m.tdi))}" r="${n1(13 + 3 * Math.sin(t * 5))}" fill="none" stroke="#1db954" stroke-width="5" opacity="${seg(u, 17, 18)}"/>` +
      text('Cap', X(m.direct) + 40, Y(m.tdi) + 40, 30, { fill: '#1db954', weight: 900, opacity: seg(u, 17, 18), stroke: '#fff', strokeWidth: 7 });
    const cx = 980, share = REAL.scatter.positives / REAL.scatter.n;
    g += `<g opacity="${seg(u, 18.4, 19.4)}">` + text(REAL.scatter.positives.toLocaleString('en-US'), cx + 150, 420, 100, { fill: '#e5484d', weight: 900, family: FONT_TITLE, maxWidth: 300 }) +
      paragraph(`of ${REAL.scatter.n.toLocaleString('en-US')} compounds are time-dependent (${(100 * share).toFixed(1)} %)`, cx, 480, 30, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: 300, lineHeight: 40 }).markup + '</g>';
    body += g + '</g>';
  }
  body += staff('kai', 1560, t, { pose: u < 9.4 ? 'point' : 'present', mood: 'happy', mouth: 'open', talk: talking(t, (u > 0.5 && u < 9) || (u > 11 && u < 19.5)), scale: 1.0 });
  body += subs(u, [[0.6, 4.4, 'Kai measures every compound twice.'], [4.6, 9.2, "Cap's two measurements agree: the compound is not time-dependent."], [10.6, 16.8, 'Real data: every point is one compound of the challenge.'],
    [17, 23.6, 'Above the line, the head start made the compound clearly stronger: that is the label.']]);
  return body;
}
