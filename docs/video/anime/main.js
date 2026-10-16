'use strict';
// Entry points of the page: render(t) draws the whole animation at time t (seconds); sheet() draws the cast for a visual check.

const stage = document.getElementById('stage');
const ASSET_SIZES = {};
if (typeof ASSETS !== 'undefined') for (const k of Object.keys(ASSETS)) if (ASSETS[k].file) ASSET_SIZES[ASSETS[k].file] = ASSETS[k].size;

// Resolves when every picture of the film is decoded, so that the first frame that uses one does not draw it empty.
function ready() {
  return Promise.all(Object.values(typeof ASSETS === 'undefined' ? {} : ASSETS).filter((a) => a.file).map((a) => { const img = new Image(); img.src = a.file; return img.decode(); }));
}


function present(body) {
  stage.innerHTML = `<defs>${DEFS.join('')}</defs>${body}`;
}

function begin() { DEFS = []; UID = 0; }

function sheet(t = 0) {
  begin();
  const names = Object.keys(CAST), poses = ['stand', 'present', 'point', 'wave', 'hold', 'cheer'], moods = ['smile', 'proud', 'surprised', 'calm', 'shout', 'happy'], mouths = ['smile', 'grin', 'o', 'flat', 'open', 'smirk'];
  let body = `<rect width="${W}" height="${H}" fill="${linear([[0, '#f4f6ff'], [1, '#d9e3f5']])}"/>`;
  names.forEach((name, i) => {
    body += person({ ...CAST[name], x: 170 + i * 315, y: 1030, scale: 0.98, pose: poses[i], mood: moods[i], mouth: mouths[i], talk: 0.6, blink: i === 2 ? 0.95 : 0, sway: 2 * Math.sin(t + i), breathe: 3 * Math.sin(t * 2 + i) });
  });
  present(body);
}

function closeup(name, t = 0, pose = 'stand', mood = 'smile', mouth = 'smile') {
  begin();
  let body = `<rect width="${W}" height="${H}" fill="${linear([[0, '#f4f6ff'], [1, '#d9e3f5']])}"/>`;
  body += person({ ...CAST[name], x: 960, y: 1650, scale: 2.0, pose, mood, mouth, talk: 0.5, sway: 2 * Math.sin(t), breathe: 3 * Math.sin(t * 2) });
  present(body);
}

// The six faces large, in different moods, for checking the drawing.
function faces(t = 0) {
  begin();
  const names = Object.keys(CAST), moods = ['smile', 'proud', 'surprised', 'calm', 'angry', 'happy'], mouths = ['smile', 'grin', 'o', 'flat', 'smirk', 'open'];
  let body = `<rect width="${W}" height="${H}" fill="${linear([[0, '#f4f6ff'], [1, '#d9e3f5']])}"/>`;
  names.forEach((name, i) => {
    body += person({ ...CAST[name], x: 170 + i * 315, y: 560 + 830 * 1.55, scale: 1.55, pose: 'stand', mood: moods[i], mouth: mouths[i], talk: 0.7, sway: 2 * Math.sin(t + i) });
  });
  present(body);
}

// A staff member drawn large, head and shoulders, for checking the detail of the art.
function portrait(name, t = 0, mood = 'smile', mouth = 'smile') {
  begin();
  let body = `<rect width="${W}" height="${H}" fill="${linear([[0, '#f4f6ff'], [1, '#cfdaf0']])}"/>`;
  body += person({ ...CAST[name], x: 960, y: 560 + 852 * 2.6, scale: 2.6, bust: true, pose: 'stand', mood, mouth, talk: 0.5, sway: 2 * Math.sin(t), breathe: 2 * Math.sin(t * 2) });
  present(body);
}

function mascotSheet(t = 0) {
  begin();
  let body = `<rect width="${W}" height="${H}" fill="${linear([[0, '#f4f6ff'], [1, '#d9e3f5']])}"/>`;
  body += cap({ x: 330, y: 600, scale: 1.0, angle: 0, mood: 'happy', mouth: 'grin', arms: [[40, 70], [150, 170]] });
  body += quanta({ x: 1020, y: 330, scale: 1.0, t, mood: 'smile' });
  body += enzyme({ x: 1480, y: 700, scale: 0.8, t, mood: 'worry', mouth: 'o' });
  body += jelly({ x: 720, y: 880, scale: 1.1, t });
  body += jelly({ x: 1000, y: 880, scale: 1.1, t, awake: false });
  present(body);
}

const TRANSITION = 1.0;

function sceneIndex(t) {
  const i = SCENES.findIndex((sc) => t >= sc.start && t < sc.end);
  return i < 0 ? SCENES.length - 1 : i;
}

// The incoming scene is revealed behind a diagonal band that crosses the screen; the outgoing scene is still drawn on the other side.
function transition(oldBody, newBody, p, color) {
  const x = lerp(-900, W + 900, inOutCubic(p)), id = uid('c');
  def(`<clipPath id="${id}"><path d="M-400 0 L${n1(x + 380)} 0 L${n1(x - 380)} ${H} L-400 ${H} Z"/></clipPath>`);
  return `<g>${oldBody}</g><g clip-path="url(#${id})">${newBody}</g>` +
    `<path d="M${n1(x + 380)} 0 L${n1(x + 520)} 0 L${n1(x - 240)} ${H} L${n1(x - 380)} ${H} Z" fill="${color}"/><path d="M${n1(x + 520)} 0 L${n1(x + 560)} 0 L${n1(x - 200)} ${H} L${n1(x - 240)} ${H} Z" fill="#fff"/>`;
}

function render(t) {
  begin();
  CURRENT_TIME = t;
  const i = sceneIndex(t), sc = SCENES[i], s = t - sc.start;
  let body = SCENE_FUNCTIONS[sc.id](s, t);
  if (sc.id !== 'open' && s > CARD * (sc.id === 'finale' ? 0 : 1)) {  // a slow push-in on the action of every scene
    const push = 1 + 0.035 * seg(s, 0, sc.end - sc.start);
    body = `<g transform="translate(960 540) scale(${n1(push * 1000) / 1000}) translate(-960 -540)">${body}</g>`;
  }
  if (i > 0 && s < TRANSITION) {
    const prev = SCENES[i - 1];
    const color = { kai: ACCENT.kai, scout: ACCENT.scout, foundry: ACCENT.foundry, bits: ACCENT.bits, stacker: ACCENT.stacker, gate: ACCENT.gatekeeper, dock: ACCENT.dock, auditor: ACCENT.auditor, finale: '#ffc857' }[sc.id] ?? '#e5484d';
    body = transition(SCENE_FUNCTIONS[prev.id](prev.end - prev.start + s, t), body, s / TRANSITION, color);
  }
  present(body + grading(t));
}

// render(t) and then wait until the browser has painted it, so that a screenshot taken afterwards shows the pictures too.
const FRAME_IMAGES = new Map();  // the decoded frames of the clips drawn recently

async function renderAsync(t) {
  await ready();
  WANTED_FRAMES.clear();
  render(t);
  const missing = [...WANTED_FRAMES].filter((href) => !FRAME_IMAGES.has(href));
  if (missing.length) {  // the first pass named the clip frames this time needs; load them, then draw again
    await Promise.all(missing.map((href) => { const img = new Image(); img.src = href; return img.decode().then(() => FRAME_IMAGES.set(href, img)); }));
    while (FRAME_IMAGES.size > 80) FRAME_IMAGES.delete(FRAME_IMAGES.keys().next().value);
    render(t);
  }
  await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

// A 53-bit string hash (cyrb53) and the digests the audit compares: the DOM of the frames of the film and of the poster is deterministic, the raster is not (see cypmode/audit/artifacts.py).
function cyrb53(str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) { const ch = str.charCodeAt(i); h1 = Math.imul(h1 ^ ch, 2654435761); h2 = Math.imul(h2 ^ ch, 1597334677); }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

// The digest of the frames [from, to) of the film at `fps`: each frame's DOM is hashed and the hashes are chained.
function domDigest(from, to, fps = 24) {
  let chain = 0;
  for (let i = from; i < to; i++) { render(i / fps); chain = cyrb53(String(chain) + ':' + cyrb53(stage.innerHTML), i); }
  return String(chain);
}

function posterDigest() { poster(); return String(cyrb53(stage.innerHTML)); }
