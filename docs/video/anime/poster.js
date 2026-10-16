'use strict';
// The poster: the title, the question the film answers, four real numbers and the whole team. It is its own picture (poster() draws it, posterAsync() waits until it is painted), not a frame of the film.

function poster() {
  begin();
  const t = 2.0;
  let body = lab(t, { tint: '#ffb48a', warm: 1, cam: 20 });
  body += `<rect width="${W}" height="${H}" fill="${linear([[0, '#fff3dc', 0.0], [0.55, '#ffd9b0', 0.0], [1, '#3a2a6a', 0.38]])}"/>`;
  body += `<g transform="translate(960 262)">` + text('CYPMode', 0, 0, 250, { fill: '#fff', stroke: '#e5484d', strokeWidth: 40, weight: 900, italic: true, family: FONT_TITLE, spacing: 8 }) +
    text('CYPMode', 0, 0, 250, { fill: '#fff', stroke: INK, strokeWidth: 10, weight: 900, italic: true, family: FONT_TITLE, spacing: 8 }) + '</g>';
  body += pill('OpenADMET CYP Inhibition Blind Challenge', 960, 345, 44, '#e5484d', { fill: '#fff', color: NAVY, fillOpacity: 0.97 });
  body += text('Will I switch off a liver enzyme, and keep it off?', 960, 432, 54, { fill: NAVY, stroke: '#fff', strokeWidth: 10, weight: 800, italic: true, maxWidth: 1500 }) +
    text('Will I switch off a liver enzyme, and keep it off?', 960, 432, 54, { fill: NAVY, weight: 800, italic: true, maxWidth: 1500 });
  const numbers = [`CYP3A4 AUC ${FACTS.auc.CYP3A4.toFixed(3)}`, `CYP2D6 AUC ${FACTS.auc.CYP2D6.toFixed(3)}`, `${FACTS.poses} poses, ${FACTS.pose_failures} failed checks`, `${FACTS.gates} ideas judged`];
  const sizes = numbers.map((n) => measure(n, 30, { weight: 800 }) + 36 + 30), total = sizes.reduce((a, b) => a + b, 0) - 30;
  let x = 960 - total / 2;
  numbers.forEach((n, i) => { body += pill(n, x, 505, 30, ACCENT.kai, { fill: '#fff', color: NAVY, fillOpacity: 0.97, anchor: 'start' }); x += sizes[i]; });
  const row = [['kai', 150], ['scout', 345], ['bits', 540], ['stacker', 740], ['gatekeeper', 1175], ['foundry', 1370], ['dock', 1570], ['auditor', 1770]];
  row.forEach(([n, px], i) => { body += staff(n, px, t, { mood: 'happy', phase: i * 0.9, paintedScale: 0.64, bottom: 1110, flip: 1 }); });
  body += quanta({ x: 960, y: 700, scale: 0.55, t, mood: 'happy', mouth: 'grin' }) + cap({ x: 960, y: 905, scale: 0.92, mood: 'proud', mouth: 'grin', blink: 0, arms: [[150, 170], [150, 170]], legs: true });
  const r = rng(11);
  for (let i = 0; i < 50; i++) {
    const cx = r() * W, cy = 560 + r() * 440, c = ['#e5484d', '#ffc857', '#7fd1ae', '#9ec5ff', '#c9a7ff'][i % 5];
    body += `<rect x="${n1(cx)}" y="${n1(cy)}" width="${10 + 8 * r()}" height="${18 + 8 * r()}" fill="${c}" opacity="0.9" transform="rotate(${n1(r() * 360)} ${n1(cx)} ${n1(cy)})"/>`;
  }
  present(body + grading(t));
}

// poster() once the pictures it names are decoded, like renderAsync for a frame of the film.
async function posterAsync() {
  await ready();
  WANTED_FRAMES.clear();
  poster();
  const missing = [...WANTED_FRAMES].filter((href) => !FRAME_IMAGES.has(href));
  if (missing.length) {
    await Promise.all(missing.map((href) => { const img = new Image(); img.src = href; return img.decode().then(() => FRAME_IMAGES.set(href, img)); }));
    poster();
  }
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}
