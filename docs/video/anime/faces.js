'use strict';
// Shared by the mascots and the staff: the ink colour, the mascots' round eyes and the mouth shapes.

const INK = '#2b2140';

// Round, wide-open eyes with lashes: used by the mascots (Cap, the assistants), not by the staff.
function eyeShape(side, o) {
  const iris = o.eye, open = 1 - 0.93 * o.blink, look = o.look ?? [0, 0], big = o.eyeScale ?? 1;
  const sclera = 'M-24 -2 C-18 -19 18 -21 25 -4 C24 20 -22 26 -24 -2 Z', clip = uid('c');
  def(`<clipPath id="${clip}"><path d="${sclera}"/></clipPath>`);
  const irisFill = linear([[0, shadeOf(iris, 0.62, '#07041a')], [0.45, iris], [1, lightOf(iris, 0.5)]]);
  const lid = (open < 0.14)
    ? `<path d="M-26 1 C-14 11 14 11 26 1" fill="none" stroke="${INK}" stroke-width="3.6" stroke-linecap="round"/>`
    : '';
  let eye = '';
  if (open >= 0.14) {
    eye += `<g transform="translate(0 -10) scale(1 ${open}) translate(0 10)">`;
    eye += `<path d="${sclera}" fill="#fffaf6"/>`;
    eye += `<g clip-path="url(#${clip})">`;
    eye += `<ellipse cx="${n1(1 + look[0] * 5)}" cy="${n1(5 + look[1] * 4)}" rx="15.5" ry="19.5" fill="${irisFill}"/>`;
    eye += `<ellipse cx="${n1(1 + look[0] * 5)}" cy="${n1(7 + look[1] * 4)}" rx="7.4" ry="10.4" fill="#150b2a"/>`;
    eye += `<ellipse cx="${n1(1 + look[0] * 5)}" cy="${n1(17 + look[1] * 4)}" rx="9.5" ry="4" fill="${lightOf(iris, 0.6)}" opacity="0.55"/>`;
    eye += `<ellipse cx="${n1(7 + look[0] * 5)}" cy="${n1(-1 + look[1] * 4)}" rx="5.6" ry="6.6" fill="#fff"/>`;
    eye += `<circle cx="${n1(-7 + look[0] * 5)}" cy="${n1(14 + look[1] * 4)}" r="2.8" fill="#fff" opacity="0.95"/>`;
    eye += `</g>`;
    eye += `<path d="M-23 -9 C-10 -24 12 -25 24 -13" fill="none" stroke="${INK}" stroke-width="1.4" opacity="0.4"/>`;
    eye += `</g>`;
    eye += taper([[-27, -1], [-17, -19 + (1 - open) * 14], [16, -22 + (1 - open) * 14], [28, -5]], 5.4, 3, INK, { bulge: 0.2 });
  } else {
    eye += lid;
  }
  return `<g transform="translate(${side * 31 * big * (o.spacing ?? 1)} 6) scale(${side * big} ${big})">${eye}</g>`;
}

function mouthShape(kind, talk) {
  const dark = '#7a2e3a';
  switch (kind) {
    case 'open': { const h = 5 + 12 * talk; return `<path d="M-11 0 C-11 ${h + 4} 11 ${h + 4} 11 0 Z" fill="${dark}"/><ellipse cx="0" cy="${n1(h * 0.62 + 1)}" rx="6" ry="${n1(Math.max(1, h * 0.32))}" fill="#e8737f"/><path d="M-11 0 C-4 -2 4 -2 11 0" stroke="${INK}" stroke-width="2.2" fill="none" stroke-linecap="round"/>`; }
    case 'grin': return `<path d="M-18 -2 C-11 18 11 18 18 -2 Z" fill="${dark}"/><path d="M-15 1 L15 1 C11 5 -11 5 -15 1 Z" fill="#fff" opacity="0.95"/><ellipse cx="0" cy="11" rx="7" ry="3.2" fill="#e8737f"/><path d="M-18 -2 C-7 -4 7 -4 18 -2" stroke="${INK}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
    case 'o': return `<ellipse cx="0" cy="3" rx="6" ry="7.5" fill="${dark}"/><ellipse cx="0" cy="6" rx="3.6" ry="2.4" fill="#e8737f"/>`;
    case 'smirk': return `<path d="M-12 3 C-3 7 9 5 15 -5" fill="none" stroke="${INK}" stroke-width="2.8" stroke-linecap="round"/>`;
    case 'worry': return `<path d="M-13 5 C-7 -1 -1 7 4 1 C7 -3 10 3 13 2" fill="none" stroke="${INK}" stroke-width="2.6" stroke-linecap="round"/>`;
    case 'flat': return `<path d="M-10 3 L10 3" fill="none" stroke="${INK}" stroke-width="2.8" stroke-linecap="round"/>`;
    default: return `<path d="M-15 0 C-7 10 7 10 15 0" fill="none" stroke="${INK}" stroke-width="3" stroke-linecap="round"/>`;
  }
}

