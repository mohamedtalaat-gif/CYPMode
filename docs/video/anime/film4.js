'use strict';
// The Auditor (everything runs twice and must match) and the finale (the results, the team, the credits).

function sceneAuditor(s, t) {
  const def = { color: ACCENT.auditor, index: `MEMBER 09 / ${MEMBER_TOTAL}`, name: 'Auditor', role: 'The reproducibility officer', lines: ['I run the whole project twice and compare the two reports byte for byte.', 'If one byte differs, the work is not reproducible and I say so.'], represents: 'the audit and the tests' };
  if (s < CARD) return introCard(def, s, t, (x) => bustOf('auditor', x, t, { pose: 'hold', mood: 'proud', mouth: 'grin', talk: talking(t, s > 1.2 && s < 6.5), holds: { right: clipboardHeld() } }));
  const u = s - CARD, color = ACCENT.auditor, R = REAL.repo;
  let body = lab(t, { tint: '#8fd3a8', cam: 20 });
  const doc = (x, title, a) => {
    let g = `<g opacity="${a}"><rect data-box="1" x="${x}" y="200" width="430" height="480" rx="26" fill="#fff" stroke="${NAVY}" stroke-width="6"/>` + text(title, x + 215, 262, 36, { fill: NAVY, weight: 900, maxWidth: 380 });
    ['tests', 'determinism', 'data integrity', 'leakage', 'baselines', 'submission files'].forEach((l, i) => {
      g += `<rect x="${x + 40}" y="${300 + i * 56}" width="${n1(350 * seg(u, 1.2 + i * 0.35, 2 + i * 0.35))}" height="14" rx="7" fill="#c3cbe0"/>` + text(l, x + 40, 290 + i * 56, 22, { fill: '#5a6477', anchor: 'start', weight: 700, opacity: seg(u, 1 + i * 0.35, 1.6 + i * 0.35), maxWidth: 340 });
    });
    return g + `</g>`;
  };
  body += doc(150, 'AUDIT RUN 1', seg(u, 0.4, 1.2)) + doc(760, 'AUDIT RUN 2', seg(u, 0.8, 1.6));
  const eq = seg(u, 7, 8);
  body += `<g opacity="${eq}"><rect x="595" y="400" width="150" height="26" rx="13" fill="${color}"/><rect x="595" y="452" width="150" height="26" rx="13" fill="${color}"/></g>`;
  const hashShown = seg(u, 5.5, 8);
  body += `<g opacity="${hashShown}">` + pill(`sha256 ${R.audit_hash}...`, 365, 730, 28, color, { fill: '#fff', color: NAVY, fillOpacity: 0.97 }) + pill(`sha256 ${R.audit_hash}...`, 975, 730, 28, color, { fill: '#fff', color: NAVY, fillOpacity: 0.97 }) + '</g>';
  body += stamp(FACTS.audit_reproducible ? 'IDENTICAL' : 'DIFFERENT', 670, 560, FACTS.audit_reproducible ? '#1db954' : NOT_MET, seg(u, 8.4, 10.6), 66, -6, FACTS.audit_reproducible ? 'byte for byte' : '');
  let px = 150;
  [[`${R.test_functions} tests`, 11.2], [`${R.modules} modules`, 12], [`${R.jobs} queue jobs`, 12.8], [`final zip ${FACTS.zip_hash}...`, 13.6]].forEach(([l, a]) => { body += pill(l, px, 830, 30, color, { opacity: seg(u, a, a + 0.8), fill: '#fff', color: NAVY, fillOpacity: 0.97, anchor: 'start' }); px += measure(l, 30) + 36 + 24; });
  body += staff('auditor', 1590, t, { pose: 'hold', mood: 'proud', mouth: 'grin', talk: talking(t, u > 0.4 && u < 15), scale: 1.0, holds: { right: clipboardHeld() } });
  body += subs(u, [[0.6, 6.4, 'The Auditor runs everything twice.'], [6.6, 11.6, 'The two audit reports are identical, byte for byte.'], [11.8, 17.6, 'Tests, queue jobs and the final zip are all accounted for.']]);
  return body;
}

function sceneFinale(s, t) {
  let body = lab(t, { tint: '#ffb48a', warm: 1, cam: 20 });
  const pop = outBack(seg(s, 0.4, 2.0), 1.6), end = seg(s, 21, 24);
  body += `<g transform="translate(960 ${n1(-420 * (1 - pop) + 275)})" opacity="${1 - end}">`;
  body += `<rect x="-720" y="-215" width="1440" height="470" rx="28" fill="${linear([[0, '#38705f'], [1, '#2a5a4c']])}" stroke="#d8b48a" stroke-width="20"/><rect x="-720" y="-215" width="1440" height="470" rx="28" fill="none" stroke="${INK}" stroke-width="4"/>`;
  const rows = [['CYP3A4', `AUC ${FACTS.auc.CYP3A4.toFixed(3)}`, `MCC ${FACTS.mcc.CYP3A4.toFixed(2)}`], ['CYP2D6', `AUC ${FACTS.auc.CYP2D6.toFixed(3)}`, `MCC ${FACTS.mcc.CYP2D6.toFixed(2)}`],
    ['structures', `${FACTS.poses} poses`, `${FACTS.pose_failures} failed checks`], ['ideas judged', `${FACTS.gates} gates`, `${FACTS.gates - FACTS.gates_not_met} met`]];
  rows.forEach((r, i) => {
    const a = seg(s, 2 + 1.4 * i, 3 + 1.4 * i), y = -110 + i * 84;
    body += text(r[0], -680, y + 14, 50, { fill: '#ffe08a', anchor: 'start', weight: 900, opacity: a, maxWidth: 380 }) + text(r[1], -240, y + 14, 50, { fill: '#fff', anchor: 'start', weight: 800, opacity: a, maxWidth: 400 }) + text(r[2], 200, y + 14, 50, { fill: '#fff', anchor: 'start', weight: 800, opacity: a, maxWidth: 460 });
  });
  for (const [k, line] of ['nested cross-validation on the training data;', 'the challenge score stays hidden until it closes'].entries()) body += text(line, 0, 200 + k * 32, 25, { fill: '#cfe9dd', weight: 600, maxWidth: 640, opacity: seg(s, 7, 8.4) });  // two short lines in the middle, clear of the raised hands
  body += '</g>';
  const row = [['kai', 150], ['scout', 345], ['bits', 540], ['stacker', 740], ['gatekeeper', 1175], ['foundry', 1370], ['dock', 1570], ['auditor', 1770]];
  row.forEach(([n, x], i) => { const e = outCubic(seg(s, 0.6 + 0.14 * i, 1.8 + 0.14 * i)); body += `<g transform="translate(0 ${n1(500 * (1 - e))})" opacity="${1 - end}">${staff(n, x, t, { pose: i % 2 ? 'wave' : 'cheer', mood: 'happy', mouth: 'grin', scale: 0.52, flip: x < 960 ? 1 : -1, phase: i * 0.9, paintedScale: 0.58, bottom: 1000, fade: true })}</g>`; });
  body += `<g opacity="${1 - end}">` + quanta({ x: 960, y: 700, scale: 0.5, t, mood: 'happy', mouth: 'grin' }) + cap({ x: 960, y: GROUND - 130, scale: 0.78, mood: 'proud', mouth: 'grin', blink: blinkAt(t, 0.3), arms: [[150, 170], [150, 170]], legs: true }) + '</g>';
  const r = rng(21);
  for (let i = 0; i < 60; i++) {
    const x = r() * W, y = ((r() * 1300 + t * (120 + 160 * r())) % 1300) - 100, c = ['#e5484d', '#ffc857', '#7fd1ae', '#9ec5ff', '#c9a7ff'][i % 5];
    if (s > 2) body += `<rect x="${n1(x)}" y="${n1(y)}" width="${10 + 8 * r()}" height="${18 + 8 * r()}" fill="${c}" opacity="${1 - end}" transform="rotate(${n1(t * 120 * r() + i * 30)} ${n1(x)} ${n1(y)})"/>`;
  }
  body += subs(s, [[2.5, 8, 'The results, estimated on the training data with nested cross-validation.'], [8.4, 14, `${word(MEMBERS).charAt(0).toUpperCase()}${word(MEMBERS).slice(1)} specialists, one molecule, and every number read from the project\'s own files.`], [14.4, 20.4, 'The challenge closes on 3 November 2026.']], { centered: true });
  const outro = seg(s, 21, 22.4);
  if (outro > 0) {
    body += `<rect width="${W}" height="${H}" fill="#0d1117" opacity="${n1(outro * 0.94 * 100) / 100}"/>`;
    body += text('CYPMode', 960, 430, 190, { fill: '#fff', stroke: '#e5484d', strokeWidth: 26, weight: 900, italic: true, family: FONT_TITLE, opacity: outro }) + text('CYPMode', 960, 430, 190, { fill: '#fff', stroke: INK, strokeWidth: 7, weight: 900, italic: true, family: FONT_TITLE, opacity: outro });
    body += text('OpenADMET CYP Inhibition Blind Challenge', 960, 520, 44, { fill: '#c9d1d9', weight: 600, opacity: outro, maxWidth: 1500 });
    const credits = ['Co-folding: Boltz-2 and OpenFold3', 'Pose validity: PoseBusters', 'Chemistry: RDKit, GFN2-xTB (tblite)', 'Models: LightGBM, scikit-learn', 'Structures: PyMOL, Blender with Molecular Nodes', 'Characters painted with the Anima model (CircleStone Labs) in ComfyUI', 'Voices: Kokoro-82M (hexgrad, Apache-2.0)'];
    credits.forEach((c, i) => { body += text(c, 960, 620 + i * 48, 30, { fill: '#9fb0c7', weight: 600, opacity: seg(s, 22.6 + 0.35 * i, 23.4 + 0.35 * i), maxWidth: 1400 }); });
  }
  return body;
}

const SCENE_FUNCTIONS = { open: sceneOpen, kai: sceneKai, bits: sceneBits, stacker: sceneStacker, gate: sceneGate, dock: sceneDock, auditor: sceneAuditor, finale: sceneFinale };
