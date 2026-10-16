'use strict';
// Bits (molecules as numbers) and Stacker (the committee of models).

const ELEMENT_COLOUR = { N: '#2f6fe4', O: '#e5484d', S: '#e0a800', Br: '#a8552a', F: '#1db954', Cl: '#1db954' };

// A flat drawing of a molecule from REAL.molecule (atoms with 2D coordinates, bonds with their order); `p` reveals the bonds one after the other.
function moleculeDrawing(m, cx, cy, size, p, o = {}) {
  const P = (a) => [cx + a[0] * size, cy + a[1] * size];
  let s = '';
  const shown = Math.floor(m.bonds.length * p) + 1;
  m.bonds.slice(0, shown).forEach(([i, j, order]) => {
    const a = P(m.atoms[i]), b = P(m.atoms[j]), dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1, nx = -dy / len, ny = dx / len;
    const offsets = order === 2 ? [-4, 4] : order === 3 ? [-7, 0, 7] : [0];
    for (const k of offsets) s += `<path d="M${n1(a[0] + nx * k)} ${n1(a[1] + ny * k)} L${n1(b[0] + nx * k)} ${n1(b[1] + ny * k)}" stroke="${o.bond ?? '#2b2140'}" stroke-width="5" stroke-linecap="round"/>`;
  });
  m.atoms.forEach(([x, y, el], i) => {
    if (el === 'C' || i >= shown + 1) return;
    const c = ELEMENT_COLOUR[el] ?? '#555', [px, py] = P([x, y]);
    s += `<circle cx="${n1(px)}" cy="${n1(py)}" r="16" fill="#fff" stroke="${c}" stroke-width="4"/>` + text(el, px, py + 7, 19, { fill: c, weight: 900 });
  });
  return s;
}

const FAMILIES = { lgbm: ['boosted trees (LightGBM)', '#2f80ff'], knn: ['nearest neighbours', '#1db954'], svm: ['support vector machines', '#f2994a'], extratrees: ['extremely randomised trees', '#a86bff'], rf: ['random forest', '#e5484d'] };

// ---- Bits: molecules as numbers -----------------------------------------------------------------------------------------------------------

function sceneBits(s, t) {
  const def = { color: ACCENT.bits, index: `MEMBER 03 / ${MEMBER_TOTAL}`, name: 'Bits', role: 'The chemoinformatician', lines: ['I turn molecules into numbers a model can read: which small groups of atoms are present, how big, how polar, how reactive.', `${FACTS.n_columns.toLocaleString('en-US')} numbers for every compound.`], represents: 'the features' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('bits', x, t, { pose: 'present', mood: 'happy', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5) }));
  const u = s - CARD, m = REAL.molecule, color = ACCENT.bits;
  let body = lab(t, { tint: '#a58bf0', cam: 60 });
  // the molecule
  body += panel(90, 120, 560, 700, 'A REAL MOLECULE', color);
  const scan = seg(u, 0.6, 4.0);
  body += moleculeDrawing(m, 370, 470, 470, scan);
  if (scan > 0 && scan < 1) body += `<rect x="116" y="${n1(250 + 440 * scan)}" width="508" height="10" fill="#35b6e8" opacity="0.85"/>`;
  body += text(m.name, 370, 760, 34, { fill: NAVY, weight: 800, maxWidth: 500 }) + text(`${m.descriptors['heavy atoms']} heavy atoms`, 370, 798, 24, { fill: '#5a6477', weight: 600 });
  // the fingerprint
  body += panel(680, 120, 740, 340, 'FINGERPRINT: 2,048 YES/NO QUESTIONS', color);
  const cols = 64, rows = 32, cw = 10.4, ch = 6.2, gx = 700, gy = 215, visible = Math.floor(2048 * outCubic(seg(u, 2.5, 9)));
  const on = new Set(m.bits);
  let cells = '';
  for (let i = 0; i < 2048; i++) {
    const x = gx + (i % cols) * cw, y = gy + Math.floor(i / cols) * ch, lit = on.has(i) && i < visible;
    cells += lit ? `<rect x="${n1(x - 1)}" y="${n1(y - 1)}" width="${cw}" height="${ch + 0.4}" fill="#ff9f1a" stroke="#7a3d00" stroke-width="1"/>` : `<rect x="${n1(x)}" y="${n1(y)}" width="${cw - 1.6}" height="${ch - 1.4}" fill="#e6e9f5"/>`;
  }
  body += cells + `<rect x="${n1(gx - 3)}" y="${gy - 4}" width="${n1(cols * cw + 2)}" height="${n1(rows * ch + 6)}" fill="none" stroke="${color}" stroke-width="2" opacity="0.5"/>`;
  body += text(`${m.bits.length} of 2,048 questions answered yes`, 1050, 440, 28, { fill: NAVY, weight: 800, opacity: seg(u, 9, 10), maxWidth: 680 });
  // the descriptors
  body += panel(680, 490, 740, 330, 'DESCRIPTORS: SIZE, SHAPE, POLARITY', color);
  const limits = { 'molecular weight': 600, 'logP': 6, 'polar surface area': 150, 'H-bond donors': 5, 'H-bond acceptors': 10, 'rotatable bonds': 10 };
  Object.entries(limits).forEach(([name, max], i) => {
    const v = m.descriptors[name], y = 606 + i * 34, a = seg(u, 10 + i * 0.5, 12 + i * 0.5), w = Math.min(1, v / max) * 290 * outCubic(a);
    body += text(name, 710, y + 8, 26, { fill: NAVY, anchor: 'start', weight: 700, maxWidth: 270 }) + `<rect x="980" y="${y - 14}" width="290" height="24" rx="12" fill="#e6e9f5"/><rect x="980" y="${y - 14}" width="${n1(Math.max(w, 0.1))}" height="24" rx="12" fill="${['#f2b230', color, '#35b6e8'][i % 3]}"/>` +
      text(String(v), 1284, y + 7, 25, { fill: NAVY, anchor: 'start', weight: 800, opacity: a, maxWidth: 120 });
  });
  // all the blocks together
  const band = seg(u, 14, 16);
  if (band > 0) {
    body += `<g opacity="${band}"><rect data-box="1" x="90" y="838" width="1330" height="100" rx="24" fill="#fff" fill-opacity="0.96" stroke="${color}" stroke-width="6"/>`;
    const total = REAL.blocks.reduce((a, b) => a + Math.sqrt(b.columns), 0);
    let x = 110; const reveal = seg(u, 14.5, 19);
    REAL.blocks.forEach((b, i) => { const w = Math.sqrt(b.columns) / total * 1290 * reveal; body += `<rect x="${n1(x)}" y="854" width="${n1(Math.max(w - 2, 0))}" height="30" rx="6" fill="${['#7a52d6', '#35b6e8', '#f2b230', '#1db954', '#e5484d'][i % 5]}"/>`; x += w; });
    body += '</g>' + text(`${Math.round(FACTS.n_columns * outCubic(seg(u, 14.5, 19))).toLocaleString('en-US')} numbers per compound, in ${REAL.blocks.length} blocks`, 755, 920, 29, { fill: NAVY, weight: 800, opacity: seg(u, 17.5, 19), maxWidth: 1240 });
  }
  body += staff('bits', 1560, t, { pose: u < 9 ? 'present' : 'point', mood: 'happy', mouth: 'open', talk: talking(t, u > 0.5 && u < 21), scale: 0.98 });
  body += subs(u, [[0.6, 5.6, 'Bits turns a real molecule into numbers.'], [6, 11.6, 'A fingerprint records which small groups of atoms the molecule contains.'], [12, 17.8, 'Descriptors record its size, its shape and how polar it is.'],
    [18, 23.6, `Together: ${FACTS.n_columns.toLocaleString('en-US')} numbers describe every compound.`]]);
  return body;
}

// ---- Stacker: the committee ---------------------------------------------------------------------------------------------------------------

function rocCurve(c, x, y, size, p, color) {
  const n = Math.max(2, Math.floor(c.fpr.length * p)), pts = [];
  for (let i = 0; i < n; i++) pts.push(`${n1(x + c.fpr[i] * size)} ${n1(y + size - c.tpr[i] * size)}`);
  return `<path d="M${pts.join(' L')}" fill="none" stroke="${color}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>`;
}

function sceneStacker(s, t) {
  const def = { color: ACCENT.stacker, index: `MEMBER 04 / ${MEMBER_TOTAL}`, name: 'Stacker', role: 'The modeller', lines: [`I train ${FACTS.n_models} different models. Each votes on whether a compound switches an enzyme off for good.`, 'Then a logistic stack learns whom to trust and blends the votes into one score.'], represents: 'the models' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('stacker', x, t, { pose: 'present', mood: 'happy', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5), headY: 420 }));
  const u = s - CARD, color = ACCENT.stacker;
  let body = lab(t, { tint: '#f09a9c', cam: 30 });
  body += panel(90, 120, 940, 700, `${FACTS.n_models} MODELS, FIVE KINDS`, color);
  let y = 230, count = 0;
  Object.entries(REAL.families).sort((a, b) => b[1] - a[1]).forEach(([fam, n]) => {
    const [label, c] = FAMILIES[fam] ?? [fam, '#888'];
    body += text(`${label}: ${n}`, 120, y, 28, { fill: NAVY, anchor: 'start', weight: 800, maxWidth: 860 });
    for (let k = 0; k < n; k++) {
      const col = k % 8, row = Math.floor(k / 8), drop = outBack(seg(u, 0.5 + 0.2 * (count + k), 1.0 + 0.2 * (count + k)), 1.5), x = 170 + col * 112, yy = y + 42 + row * 52;
      if (drop > 0) body += `<g transform="translate(0 ${n1(-260 * (1 - drop))})" opacity="${Math.min(1, drop * 2)}">${tabletIcon(x, yy, 96, 40, c, '', 24)}</g>`;
    }
    count += n; y += 96 + (n > 8 ? 52 : 0);
  });
  // the real ROC curves
  body += panel(1060, 120, 520, 700, 'ONE BLENDED SCORE', color);
  const rx = 1110, ry = 240, rs = 420;
  body += `<rect x="${rx}" y="${ry}" width="${rs}" height="${rs}" fill="#f4f6fb" stroke="${NAVY}" stroke-width="3"/><path d="M${rx} ${ry + rs} L${rx + rs} ${ry}" stroke="#9aa3b8" stroke-width="3" stroke-dasharray="10 8"/>` +
    text('false positives', rx + rs / 2, ry + rs + 38, 24, { fill: '#5a6477', weight: 600 }) + text('true positives', rx + 8, ry - 14, 24, { fill: '#5a6477', weight: 600, anchor: 'start' });
  body += rocCurve(REAL.curves.CYP3A4, rx, ry, rs, seg(u, 10, 15), '#2f80ff') + rocCurve(REAL.curves.CYP2D6, rx, ry, rs, seg(u, 14.5, 18.5), '#f2994a');
  body += pill(`CYP3A4  AUC ${REAL.curves.CYP3A4.auc.toFixed(3)}`, rx + 290, ry + rs - 100, 26, '#2f80ff', { opacity: seg(u, 14, 15), color: '#cfe3ff' }) + pill(`CYP2D6  AUC ${REAL.curves.CYP2D6.auc.toFixed(3)}`, rx + 290, ry + rs - 44, 26, '#f2994a', { opacity: seg(u, 18, 19), color: '#ffe7cf' });
  body += text('measured on folds the models never saw', 1320, 790, 24, { fill: '#5a6477', weight: 600, maxWidth: 480, opacity: seg(u, 14, 15) });
  // the pipeline
  const steps = ['every model votes', 'a logistic stack weighs the votes', 'one score per compound', 'the top fraction is called positive'];
  let sx = 110;
  steps.forEach((label, i) => {
    const a = seg(u, 19 + i * 0.7, 20 + i * 0.7), w = measure(label, 24, { weight: 800 }) + 24 * 1.2;
    body += pill(label, sx, 880, 24, color, { opacity: a, fill: '#ffffff', color: NAVY, fillOpacity: 0.95, anchor: 'start' });
    if (i < steps.length - 1) body += `<path d="M${n1(sx + w + 6)} 880 L${n1(sx + w + 30)} 880" stroke="#fff" stroke-width="7" stroke-linecap="round" opacity="${a}"/>`;
    sx += w + 38;
  });
  body += staff('stacker', 1690, t, { pose: 'present', mood: 'proud', mouth: 'open', talk: talking(t, u > 0.5 && u < 21), scale: 0.98 });
  body += subs(u, [[0.6, 7.4, `Stacker's ${FACTS.n_models} models each cast a vote.`], [8, 13.6, 'A logistic stack learns whom to trust.'], [14, 19.4, 'Its ranking is checked on folds that no choice has seen.'], [19.6, 23.6, 'One score per compound: the highest are called positive.']]);
  return body;
}
