'use strict';
// The painted staff: each member is a set of expression pictures (calm, blink, talk, happy) of one portrait, cut out and drawn upper body only, like a presenter at the edge of the screen.
// ``staff`` and ``bustOf`` of film1.js call ``paintedStaff`` and use the vector people (people.js) when no pictures exist. The pictures are made by docs/video/characters (see its README of the
// storyboard) and listed in assets_characters.js; every picture a frame needs is named in WANTED_FRAMES so that renderAsync decodes it before the screenshot.

const SPRITE_SCALE = 0.86;  // the portrait is 768 x 1024 pixels; at this scale a member is about 880 pixels tall and his head about 220

function spriteHref(name, expression) {
  const c = typeof CHARACTERS === 'undefined' ? null : CHARACTERS[name];
  if (!c) return null;
  return c.expressions[expression] ?? c.expressions.calm ?? null;
}

// Which picture of the member shows at time t: a blink every few seconds, the open mouth while he speaks (the mouth opens and closes with the speech), a laugh now and then when he is happy.
function spriteExpression(t, o) {
  if (o.blink > 0.5) return 'blink';
  if ((o.talk ?? 0) > 0.55) return 'talk';
  if (o.mood === 'happy' && Math.sin(t * 1.9 + (o.phase ?? 0)) > 0.8) return 'happy';
  return 'calm';
}

// A member standing at the bottom edge of the screen with his centre at x. o: scale, blink (0..1, from blinkAt), talk (0..1), mood, phase, y (the bottom of the picture), flip.
function paintedStaff(name, x, t, o = {}) {
  const expression = spriteExpression(t, o), href = spriteHref(name, expression);
  if (!href) return null;
  const c = CHARACTERS[name], [w, h] = c.size, k = (o.scale ?? 1) * SPRITE_SCALE * (o.sizeFactor ?? 1), phase = o.phase ?? 0;
  WANTED_FRAMES.add(href);
  const breathe = 3.0 * Math.sin(t * 2.0 + phase), bob = (o.talk ?? 0) > 0.05 ? 2.5 * Math.sin(t * 13) : 0, sway = 0.55 * Math.sin(t * 1.3 + phase);
  const edge = o.bottom ?? (o.y ?? H) + 12, bottom = edge + breathe + bob;  // by default a few pixels below the frame, so that the cut at the waist never shows
  const pivot = `${n1(x)} ${n1(bottom)}`;
  let fade = '';
  if (o.fade) {  // a person who floats above the bottom of the frame fades out at the waist instead of ending in a straight cut
    const id = uid('m');
    def(`<linearGradient id="${id}g" gradientUnits="userSpaceOnUse" x1="0" y1="${n1(h * 0.80)}" x2="0" y2="${h}"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>` +
        `<mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="${w}" height="${h}"><rect width="${w}" height="${h}" fill="url(#${id}g)"/></mask>`);
    fade = ` mask="url(#${id})"`;
  }
  const shadow = `<ellipse cx="${n1(x)}" cy="${n1(bottom - 6)}" rx="${n1(w * k * 0.34)}" ry="18" fill="#1a1040" opacity="0.22" filter="${blur(9)}"/>`;
  return shadow + `<g transform="rotate(${n1(sway * 100) / 100} ${pivot}) scale(${(o.flip ?? 1) * 1} 1)" transform-origin="${pivot}">` +
    `<g transform="translate(${n1(x - w * k / 2)} ${n1(bottom - h * k)}) scale(${n1(k * 1000) / 1000})"><image href="${href}" x="0" y="0" width="${w}" height="${h}"${fade}/></g></g>`;
}
