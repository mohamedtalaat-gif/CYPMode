'use strict';
// The Gatekeeper and Quanta (the pre-declared gates, the physics that did not pass) and Dock (the structure track on real pictures).

const NOT_MET = '#e5484d', MET = '#1db954';

function sceneGate(s, t) {
  const gk = { color: ACCENT.gatekeeper, index: `MEMBER 05 / ${MEMBER_TOTAL}`, name: 'Gatekeeper', role: 'The statistician', lines: ['Before an experiment runs, I write down the rule it must pass. Nothing is tuned until it passes.', 'Every idea is judged alone and inside the final blend.'], represents: 'the pre-declared gates' };
  const qa = { color: ACCENT.quanta, index: `MEMBER 06 / ${MEMBER_TOTAL}`, name: 'Quanta', role: 'The quantum chemist', lines: ['I computed orbitals and reactive sites with GFN2-xTB, atom by atom, for thousands of molecules.'], represents: 'the physics features' };
  if (s < CARD) return introCard(gk, s, t, (x) => bustOf('gatekeeper', x, t, { pose: 'hold', mood: 'calm', mouth: 'smile', talk: talking(t, s > 1.2 && s < 6.5), holds: { right: clipboardHeld() } }));
  const G1 = CARD + 16, G2 = G1 + CARD;
  if (s >= G1 && s < G2) return introCard(qa, s - G1, t, (x) => quanta({ x, y: 560, scale: 2.0, t, mood: 'smile', mouth: 'smile' }));
  if (s >= G2) return sceneQuanta(s - G2, t);
  const u = s - CARD;
  let body = lab(t, { tint: '#8fb4f5', cam: 40 });
  const color = ACCENT.gatekeeper, rule = seg(u, 0.4, 1.2) * (1 - seg(u, 3.4, 4.2));
  if (rule > 0.01) {
    body += `<g opacity="${rule}">` + panel(160, 220, 1160, 500, 'THE RULE, WRITTEN FIRST', color);
    const lines = ['A new idea must beat the reference with a corrected p value below 0.05,', 'at both confidence cut-offs, alone and inside the final blend.', 'The rule is fixed before the result exists.'];
    lines.forEach((l, i) => { body += text(l, 740, 400 + i * 78, 40, { fill: NAVY, weight: 700, maxWidth: 1060, opacity: seg(u, 0.8 + i * 0.5, 1.6 + i * 0.5) }); });
    body += '</g>';
  }
  const rows = REAL.gates, shown = seg(u, 3.8, 13.4), colW = 590, perCol = Math.ceil(rows.length / 2), rowH = Math.min(46, 520 / perCol), size = Math.max(19, Math.min(25, rowH * 0.58));
  if (shown > 0) {
    body += panel(90, 120, 1250, 700, 'THE LEDGER: EVERY IDEA AND ITS VERDICT', color);
    rows.forEach((r, i) => {  // two columns that fill the panel, whatever the number of ideas; the row height and the type size follow the count
      const col = Math.floor(i / perCol), row = i % perCol, x = 120 + col * (colW + 20), y = 232 + row * rowH, a = seg(shown, i / rows.length, i / rows.length + 0.06);
      if (a <= 0) return;
      const ok = r.met > 0 && r.not_met === 0, mixed = r.met > 0 && r.not_met > 0, c = ok ? MET : mixed ? '#f2994a' : NOT_MET, word = ok ? 'MET' : mixed ? `${r.met} MET` : 'NOT MET';
      body += `<g opacity="${a}">` + text(r.label, x, y + 10, size, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: colW - 190 }) + pill(word, x + colW - 70, y, Math.max(16, size - 4), c, { fill: c, color: '#fff', fillOpacity: 1 }) + '</g>';
    });
  }
  const tally = seg(u, 13.6, 14.6);
  if (tally > 0) body += `<g opacity="${tally}">` + pill(`${FACTS.gates} verdicts: ${FACTS.gates - FACTS.gates_not_met} met, ${FACTS.gates_not_met} not met`, 715, 780, 34, color, { fill: '#fff', color: NAVY, fillOpacity: 0.97 }) + '</g>';
  body += staff('gatekeeper', 1560, t, { pose: 'hold', mood: u > 10 ? 'proud' : 'calm', mouth: 'open', talk: talking(t, u > 0.4 && u < 15), scale: 1.0, holds: { right: clipboardHeld() } });
  body += subs(u, [[0.4, 4, 'Gatekeeper fixes the rule before the result exists.'], [4.2, 9.6, 'Every idea the project tried is in the ledger.'], [9.8, 15.8, 'Almost all of them ended "not met": the data did not support them.']]);
  return body;
}

// A clipboard in the hand frame of a staff member (the origin is the wrist, +y runs along the forearm).
function clipboardHeld() {
  return `<g transform="translate(-6 46) rotate(-8)"><rect x="-44" y="-52" width="88" height="112" rx="8" fill="#c9a27a" stroke="${INK}" stroke-width="3.4"/><rect x="-34" y="-40" width="68" height="92" rx="4" fill="#fff" stroke="${INK}" stroke-width="2.4"/>` +
    [0, 1, 2, 3].map((i) => `<path d="M-24 ${-22 + i * 20} L24 ${-22 + i * 20}" stroke="#8a8fa3" stroke-width="3.4" stroke-linecap="round"/>`).join('') + `<rect x="-18" y="-62" width="36" height="16" rx="5" fill="#8a8fa3" stroke="${INK}" stroke-width="3"/></g>`;
}

function sceneQuanta(u, t) {
  const color = ACCENT.quanta;
  let body = lab(t, { tint: '#f0c470', cam: 30 });
  body += panel(90, 120, 1000, 700, 'PHYSICS FEATURES: SIGNAL OF EACH COLUMN ALONE', color);
  const x0 = 700, scale = 1500;  // AUC 0.5 is the centre line; the bar length is (AUC - 0.5)
  body += `<path d="M${x0} 220 L${x0} 770" stroke="${NAVY}" stroke-width="4"/>` + text('no signal (AUC 0.5)', x0, 800, 24, { fill: '#5a6477', weight: 600, maxWidth: 360 });
  REAL.physics.forEach((r, i) => {
    const y = 270 + i * 74, a = outCubic(seg(u, 0.6 + i * 0.5, 1.6 + i * 0.5)), w = (r.auc - 0.5) * scale * a, positive = w >= 0;
    body += text(r.column, 120, y + 9, 26, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: 300 }) +
      `<rect x="${n1(positive ? x0 : x0 + w)}" y="${y - 20}" width="${n1(Math.max(Math.abs(w), 0.5))}" height="40" rx="10" fill="${positive ? '#35b6e8' : '#f2994a'}"/>` +
      text(r.auc.toFixed(3), positive ? x0 + w + 14 : x0 + w - 14, y + 9, 24, { fill: NAVY, anchor: positive ? 'start' : 'end', weight: 800, opacity: a });
  });
  const nm = REAL.gates.filter((g) => /physics/.test(g.label)).reduce((a, g) => a + g.not_met, 0), st = stamp('NOT MET', 1380, 520, NOT_MET, seg(u, 5.4, 7.6), 60, -7, `${nm} physics verdicts`);
  body += st;
  body += quanta({ x: 1380, y: 300 + 14 * Math.sin(t * 2), scale: 0.9, t, mood: u > 5.4 ? 'worry' : 'happy', mouth: u > 5.4 ? 'worry' : 'grin' });
  body += subs(u, [[0.6, 5.2, 'Quanta computed physics for every compound with GFN2-xTB.'], [5.4, 9.6, 'Some columns carry signal alone, but no physics idea passed its gate.'], [9.8, 13.6, 'Beautiful physics: honestly reported as not used.']]);
  return body;
}

// ---- Dock: the structure track ----------------------------------------------------------------------------------------------------------

function callout(file, key, label, x, y, w, h, p, dx, dy, color, opts = {}, alpha = 1) {
  const a = ASSETS[key] ?? ASSETS[file], name = opts.anchor ?? 'fe', f = opts.clip ? clipFrame(opts.clip, opts.clipTime) : null;
  const anchor = f ? [f.clip.anchors[name][f.index][0] * f.clip.size[0], f.clip.anchors[name][f.index][1] * f.clip.size[1]] : a.anchors[name];
  const [px, py] = imagePoint(f ? f.href : a.file, anchor[0], anchor[1], x, y, w, h, p, f ? { z0: 1.0, z1: 1.0 } : (opts.image ?? {}));
  return `<g opacity="${alpha}"><path d="M${n1(px)} ${n1(py)} L${n1(px + dx)} ${n1(py + dy)}" stroke="${color}" stroke-width="4"/><circle cx="${n1(px)}" cy="${n1(py)}" r="8" fill="none" stroke="${color}" stroke-width="4"/>` + pill(label, px + dx, py + dy, 26, color, { color: '#fff' }) + '</g>';
}

function sceneDock(s, t) {
  const def = { color: ACCENT.dock, index: `MEMBER 08 / ${MEMBER_TOTAL}`, name: 'Dock', role: 'The structural biologist', lines: ['I fold each ligand into the enzyme with Boltz-2, restrain the nitrogen that can bind the iron, and check every pose.', 'Two co-folding models work with me.'], represents: 'the structure track' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('dock', x, t, { pose: 'present', mood: 'happy', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5) }));
  const u = s - CARD, color = ACCENT.dock, A = ASSETS, D = A.distances;
  let body = lab(t, { tint: '#f6b27a', cam: 20 });
  const fx = 90, fy = 130, fw = 1150, fh = 660;
  const part = (a, b) => seg(u, a, a + 0.8) * (1 - seg(u, b - 0.8, b));
  // 1. the enzyme, 2. the real Type II pocket, 3. a donor ligand without and with the restraint
  if (part(0, 8) > 0.01) body += `<g opacity="${part(0, 8)}">` + clipImage('enzyme', A.enzyme.file, u, fx, fy, fw, fh, seg(u, 0, 8), { z0: 1.5, z1: 1.9, stroke: color }) + pill('CYP3A4 with its heme (orange) and a ligand (teal): real coordinates', fx + fw / 2, fy + fh + 40, 28, color) + '</g>';
  if (part(7.4, 16) > 0.01) {
    const p = seg(u, 7.4, 16), alpha = part(7.4, 16);
    body += `<g opacity="${alpha}">` + clipImage('pocket', A.pocket_type_ii.file, u - 7.4, fx, fy, fw, fh, p, { z0: 1.35, z1: 1.55, stroke: color }) +
      callout(null, 'pocket_type_ii', 'iron of the heme', fx, fy, fw, fh, p, -180, 110, '#ffb347', { anchor: 'fe', image: { z0: 1.35, z1: 1.55 }, clip: 'pocket', clipTime: u - 7.4 }, seg(u, 9, 10)) +
      callout(null, 'pocket_type_ii', 'nitrogen of the ligand', fx, fy, fw, fh, p, 150, -150, '#2ee6c8', { anchor: 'coordinating_n', image: { z0: 1.35, z1: 1.55 }, clip: 'pocket', clipTime: u - 7.4 }, seg(u, 10.6, 11.6)) +
      pill('Type II: the nitrogen binds the iron (crystal structure 6MA7)', fx + fw / 2, fy + fh + 40, 28, color, { opacity: seg(u, 12.4, 13.4) }) + '</g>';
  }
  if (part(15.4, 27.4) > 0.01) {
    const p = seg(u, 15.4, 27.4), alpha = part(15.4, 27.4);
    body += `<g opacity="${alpha}">` + realImage(A.example_compare.file, fx, fy, fw, fh, p, { z0: 1.4, z1: 1.8, stroke: color, fx: 0.45, fy: 0.5 }) +
      pill(`${D.ligand}: ligand pose without the restraint`, fx + 24, fy + 40, 25, '#ff6b6b', { opacity: seg(u, 17, 18), color: '#ffd5d5', anchor: 'start' }) + pill(`and with the iron-nitrogen restraint`, fx + 24, fy + 100, 25, '#2ee6c8', { opacity: seg(u, 19, 20), anchor: 'start' }) +
      pill(`nitrogen to iron: ${D.unguided} A without, ${D.restrained} A with the restraint`, fx + fw / 2, fy + fh + 40, 28, color, { opacity: seg(u, 21, 22) }) + '</g>';
    body += stamp('RESTRAINT: MET', 1020, 330, MET, seg(u, 23.2, 25.4) * (1 - seg(u, 26.6, 27.4)), 46, -7, 'on the crystal benchmark');
  }
  // 4. PoseBusters
  const pb = seg(u, 27.4, 28.4);
  if (pb > 0) {
    const chk = REAL.structure.checks, n = REAL.structure.n;
    body += `<g opacity="${pb}">` + panel(fx, 120, fw, 700, `POSEBUSTERS: ${chk.length} PHYSICAL CHECKS ON EVERY POSE`, color);
    chk.forEach((c, i) => {
      const col = Math.floor(i / 11), row = i % 11, x = fx + 40 + col * 560, y = 236 + row * 46, a = seg(u, 28.2 + i * 0.12, 28.7 + i * 0.12), ok = REAL.structure.passed[i] === n;
      body += `<g opacity="${a}">` + text(c, x + 50, y + 9, 25, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: 480 }) + `<circle cx="${x + 20}" cy="${y}" r="15" fill="${ok ? MET : NOT_MET}"/>` + text(ok ? 'ok' : 'x', x + 20, y + 8, 20, { fill: '#fff', weight: 900 }) + '</g>';
    });
    const done = seg(u, 31.4, 32.6);
    body += `<g opacity="${done}">` + pill(`${FACTS.poses - FACTS.pose_failures} of ${FACTS.poses} submitted poses pass every check`, fx + fw / 2, 772, 32, MET, { fill: '#fff', color: NAVY, fillOpacity: 0.97 }) + '</g></g>';
  }
  body += jelly({ x: 1280, y: 870, scale: 0.5, t }) + text('Boltz-2', 1280, 922, 22, { fill: '#fff', stroke: INK, strokeWidth: 6, weight: 800 });
  body += jelly({ x: 1425, y: 870, scale: 0.5, t, awake: false }) + text('OpenFold3', 1425, 922, 22, { fill: '#fff', stroke: INK, strokeWidth: 6, weight: 800 });
  body += staff('dock', 1590, t, { pose: u < 27 ? 'present' : 'hold', mood: 'happy', mouth: 'open', talk: talking(t, u > 0.5 && u < 34), scale: 0.98, flip: 1 });
  body += subs(u, [[0.6, 7.4, 'Dock folds the molecule into the enzyme with Boltz-2.'], [8, 15.6, 'Some molecules bind the heme iron through a nitrogen: a Type II interaction.'], [16, 22.4, 'For some ligands the nitrogen is placed far from the iron, so Dock restrains it.'],
    [22.6, 27, 'On the crystal benchmark the restraint passed its pre-declared gate.'], [28, 35.6, 'Every pose is checked with PoseBusters before it is handed in.']]);
  return body;
}
