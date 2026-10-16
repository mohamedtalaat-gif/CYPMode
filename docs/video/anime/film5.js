'use strict';
// Scout (the files the organisers released and the guard that keeps the blinded test set out) and Foundry (the pretrained models, each judged by the same rule as every other idea).

const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
const word = (n) => WORDS[n] ?? String(n);
const sentence = (str) => str.charAt(0).toUpperCase() + str.slice(1);
const thousands = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

function sceneScout(s, t) {
  const def = { color: ACCENT.scout, index: `MEMBER 02 / ${MEMBER_TOTAL}`, name: 'Scout', role: 'The data scout', lines: ['I go through every file the organisers released and compare it, by compound name, with the blinded test set.', 'A compound of the test set is taken out before anything is computed.'], represents: 'the official extra files and the leak guard' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('scout', x, t, { pose: 'present', mood: 'happy', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5) }));
  const u = s - CARD, color = ACCENT.scout, R = REAL.scout ?? { sources: [], excluded: [] };
  const hits = R.sources.filter((r) => r.in_test > 0).length, taken = R.excluded.length;
  let body = lab(t, { tint: '#f1b4d8', cam: 30 });
  body += panel(90, 110, 1250, 740, 'FILES RELEASED BY THE ORGANISERS, CHECKED BY NAME', color);
  const rowH = Math.min(86, 540 / Math.max(R.sources.length, 1));
  body += text('compounds in the file', 640, 224, 24, { fill: '#5a6477', weight: 700, maxWidth: 280 }) + text('also in the blinded test set', 1010, 224, 24, { fill: '#5a6477', weight: 700, maxWidth: 360 });
  R.sources.forEach((r, i) => {
    const y = 290 + i * rowH, a = seg(u, 1.2 + i * 0.9, 2 + i * 0.9), bad = r.in_test > 0, c = bad ? NOT_MET : MET;
    body += `<g opacity="${a}">` + text(r.label, 130, y + 10, 30, { fill: NAVY, anchor: 'start', weight: 800, maxWidth: 400 }) + text(thousands(r.compounds), 640, y + 10, 30, { fill: NAVY, weight: 800, maxWidth: 220 }) +
      pill(bad ? `${word(r.in_test)} taken out` : 'none', 1010, y, 26, c, { fill: c, color: '#fff', fillOpacity: 1 }) + '</g>';
  });
  const check = seg(u, 1 + R.sources.length * 0.9 + 0.4, 2 + R.sources.length * 0.9 + 1.4);
  if (check > 0) body += `<g opacity="${check}">` + pill(`${taken} compound${taken === 1 ? '' : 's'} of the blinded test set taken out before use`, 715, 800, 28, color, { fill: '#fff', color: NAVY, fillOpacity: 0.97 }) + '</g>';
  body += staff('scout', 1590, t, { pose: 'present', mood: 'happy', mouth: 'open', talk: talking(t, u > 0.4 && u < 22), scale: 1.0 });
  body += subs(u, [[0.6, 5.4, 'Scout goes through every file the organisers released.'], [5.6, 11.4, 'Only names are compared: the measurement of a test compound is never read.'],
    [11.6, 17.4, sentence(`${hits === 0 ? 'no file' : `${word(hits)} of the ${word(R.sources.length)} files`} hold${hits === 1 ? 's' : ''} test compounds, ${word(taken)} in all. ${taken === 1 ? 'It is' : 'They are'} taken out first.`)], [17.6, 23.6, 'The organisers confirmed the overlap of the Octant release.']]);
  return body;
}

// The verdict of one pretrained model over both enzymes: pending while any enzyme is still running; "not run" when it was dropped; "screened" when only the first stage of a staged design exists;
// otherwise MET if any enzyme passed and NOT MET if none did.
function foundryVerdict(row) {
  const v = Object.values(row.enzymes).map((e) => e.verdict);
  return v.includes('pending') ? 'pending' : v.every((x) => x === 'not run') ? 'not run' : v.includes('screened') ? 'screened' : v.includes('MET') ? 'MET' : 'NOT MET';
}

function sceneFoundry(s, t) {
  const def = { color: ACCENT.foundry, index: `MEMBER 07 / ${MEMBER_TOTAL}`, name: 'Foundry', role: 'The foundation-model engineer', lines: ['I start from models pretrained on millions of molecules and ask whether any of them improves the blend.', 'Each one is added to the stack and judged by the same rule.'], represents: 'the pretrained models' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('foundry', x, t, { pose: 'hold', mood: 'proud', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5) }));
  const u = s - CARD, color = ACCENT.foundry, rows = REAL.foundry ?? [];
  const verdicts = rows.map(foundryVerdict), judged = verdicts.filter((v) => v === 'MET' || v === 'NOT MET').length, notMet = verdicts.filter((v) => v === 'NOT MET').length, met = verdicts.filter((v) => v === 'MET').length;
  const screened = verdicts.filter((v) => v === 'screened').length, notRun = verdicts.filter((v) => v === 'not run').length;
  let body = lab(t, { tint: '#8fd8e8', cam: 30 });
  body += panel(60, 110, 1320, 740, 'PRETRAINED MODELS ADDED TO THE STACK', color);
  const rowH = Math.min(56, 480 / Math.max(rows.length, 1)), size = Math.max(20, Math.min(26, rowH * 0.5));
  body += text('change in AUC when added to the stack (p value)', 790, 214, 22, { fill: '#5a6477', weight: 700, maxWidth: 600 }) + text('CYP3A4', 640, 244, 22, { fill: NAVY, weight: 800, maxWidth: 150 }) + text('CYP2D6', 940, 244, 22, { fill: NAVY, weight: 800, maxWidth: 150 });
  const change = (e) => (e.verdict === 'pending' ? 'still running' : e.verdict === 'not run' ? 'not run' : `${e.delta_auc >= 0 ? '+' : '-'}${Math.abs(e.delta_auc).toFixed(4)} (p ${e.p.toFixed(2)})`);
  rows.forEach((r, i) => {
    const y = 296 + i * rowH, a = seg(u, 1 + i * 0.8, 1.8 + i * 0.8), v = verdicts[i], c = v === 'MET' ? MET : v === 'NOT MET' ? NOT_MET : v === 'screened' ? '#c98a1b' : '#7a8499';
    body += `<g opacity="${a}">` + text(r.label, 90, y + 8, size, { fill: NAVY, anchor: 'start', weight: 800, maxWidth: 360 }) + text(change(r.enzymes.CYP3A4), 640, y + 8, size - 2, { fill: NAVY, weight: 700, maxWidth: 290 }) +
      text(change(r.enzymes.CYP2D6), 940, y + 8, size - 2, { fill: NAVY, weight: 700, maxWidth: 290 }) + pill(v === 'pending' ? 'RUNNING' : v.toUpperCase(), 1240, y, Math.max(16, size - 6), c, { fill: c, color: '#fff', fillOpacity: 1 }) + '</g>';
  });
  const tally = seg(u, 1.4 + rows.length * 0.8, 2.4 + rows.length * 0.8);
  if (tally > 0) body += `<g opacity="${tally}">` + pill(`${judged} judged: ${met} met, ${notMet} not met` + (screened ? `; ${screened} screened` : '') + (notRun ? `; ${notRun} not run` : ''), 720, 800, 28, color, { fill: '#fff', color: NAVY, fillOpacity: 0.97 }) + '</g>';
  body += staff('foundry', 1620, t, { pose: 'hold', mood: 'proud', mouth: 'open', talk: talking(t, u > 0.4 && u < 24), scale: 1.0 });
  body += subs(u, [[0.6, 5.4, 'Foundry starts from models pretrained on millions of molecules.'], [5.6, 11.4, 'Each one is added to the stack and judged by the same rule.'],
    [11.6, 17.4, judged === 0 ? 'Their verdicts arrive as the queue finishes.' : `${notMet} of the ${judged} judged so far did not pass.`], [17.6, 23.6, 'A model that adds nothing to the blend is reported as nothing.']]);
  return body;
}

Object.assign(SCENE_FUNCTIONS, { scout: sceneScout, foundry: sceneFoundry });
