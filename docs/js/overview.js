/* VEDJE pipeline animation (How it works): Figure 2 of the paper (VEDJE pipeline) as three scenes named after its
   panels, (A) Offline indexing, (B) Training only and (C) Online reranking. The script builds the whole card inside
   #vedjeAnim: the scene tabs, the play button, the SVG stage, the progress bar and the caption.

   One toy video runs through the three scenes, and every value is computed here from the scene before:
   - the video: T = 16 sampled frames (the paper's default) of 6 x 6 patches, a person walking to the right, drawn
     like the cover's frames;
   - X_t, the frozen patch features: a constant background (sky, ground, sun) plus the person's footprint;
   - Z_t, the frame's M = 4 tokens: four queries attend softly to the patches of that frame only, one around each
     quadrant, so the cache keeps a separate token group for each frame;
   - the target Delta_{t,3} = X_{t+3} - X_t (h = 3, the paper's default horizon) is the actual difference of those
     features: the background cancels and the cells light up where the person moves. The future-delta head is a
     kernel regression from Z_t onto the targets of the 13 valid pairs, and L_delta is the mean squared error over
     those pairs;
   - reranking: four candidates of the same scene (walking to the right, walking to the left, standing, nobody)
     with a stage-1 prior; the joint score reads the direction of motion from the ordered tokens of each cache, the
     prior is embedded by e_rho and added before the score head, and the list is sorted by the result.
   The storage counter uses the paper's numbers (Section 3.2): 2 bytes x M x d = 2 x 4 x 384 bytes = 3 KiB per frame
   and 48 KiB for 16 frames.
   URL flags: ?static=1 shows each scene's still frame; ?animScene=N (0 to 2) opens scene N, and with ?animT=ms it
   shows that moment, paused. */
(function () {
  'use strict';
  var root = document.getElementById('vedjeAnim'); if (!root) return;
  var NS = 'http://www.w3.org/2000/svg';
  var query = new URLSearchParams(location.search);
  var reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var STATIC = query.has('static') || reduced;

  /* ------------------------------------------------------------------ */
  /* Card markup                                                         */
  /* ------------------------------------------------------------------ */
  function H(tag, cls, parent, attrs) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (attrs) for (var a in attrs) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  var TABS = [['A', 'Offline indexing', 'Indexing'], ['B', 'Training only', 'Training'], ['C', 'Online reranking', 'Reranking']];
  var head = H('div', 'va-head', root);
  var tabs = H('div', 'seg va-tabs', head, { role: 'tablist', 'aria-label': 'Panel of the VEDJE pipeline' });
  TABS.forEach(function (t, i) {
    var b = H('button', null, tabs, { type: 'button', role: 'tab', id: 'vedjeAnimTab' + i, 'data-step': String(i),
      'aria-selected': i ? 'false' : 'true', 'aria-controls': 'vedjeAnimStage', tabindex: i ? '-1' : '0' });
    H('span', 'va-num', b, { 'aria-hidden': 'true' }).textContent = t[0];
    H('span', 'lbl-long', b).textContent = t[1];
    H('span', 'lbl-short', b).textContent = t[2];
  });
  var playBtn = H('button', 'va-play', head, { type: 'button', 'data-mode': 'play', 'aria-label': 'Play the animation' });
  playBtn.innerHTML = '<svg class="i-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z"/></svg>' +
    '<svg class="i-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>' +
    '<svg class="i-replay" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/></svg>';
  var stage = H('div', 'va-stage', root, { id: 'vedjeAnimStage', role: 'tabpanel', 'aria-labelledby': 'vedjeAnimTab0' });
  var svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'va-svg'); svg.setAttribute('role', 'img'); svg.setAttribute('viewBox', '0 0 960 380');
  stage.appendChild(svg);
  var prog = H('div', 'va-progress', root, { 'aria-hidden': 'true' });
  var bar = H('i', null, prog);
  var cap = H('p', 'va-caption', root, { 'aria-live': 'polite' });
  var capTitle = H('strong', null, cap);
  cap.appendChild(document.createTextNode(' '));
  var capText = H('span', null, cap);

  /* ------------------------------------------------------------------ */
  /* The toy video and everything computed from it                       */
  /* ------------------------------------------------------------------ */
  var T = 16, G = 6, HZ = 3, NPAIR = T - HZ;
  var SY = 3.35, SXW = 0.62, SYW = 0.85, AMP = 0.8, ATT = 1.25;
  function clamp(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function bgFeat(r, c) { return r === 0 && c === 5 ? 0.30 : r >= 4 ? 0.22 : 0.08; }
  /* the person's horizontal position (in patches) at frame t = 1 to T */
  var PATHS = {
    right: function (t) { return 0.8 + 4.4 * (t - 1) / (T - 1); },
    left: function (t) { return 5.2 - 4.4 * (t - 1) / (T - 1); },
    stand: function () { return 3.0; },
    nobody: null
  };
  function features(path, t) {
    var x = path ? path(t) : null, out = [];
    for (var r = 0; r < G; r++) for (var c = 0; c < G; c++) {
      var f = bgFeat(r, c);
      if (x != null) { var dx = c + 0.5 - x, dy = r + 0.5 - SY; f += AMP * Math.exp(-(dx * dx / (2 * SXW * SXW) + dy * dy / (2 * SYW * SYW))); }
      out.push(Math.min(1, f));
    }
    return out;
  }
  /* four learned queries, each attending softly around one quadrant of the frame's own patches */
  var QCEN = [[1.5, 1.5], [1.5, 4.5], [4.5, 1.5], [4.5, 4.5]];
  var ATTW = QCEN.map(function (cc) {
    var w = [], s = 0;
    for (var r = 0; r < G; r++) for (var c = 0; c < G; c++) {
      var d = (r + 0.5 - cc[0]) * (r + 0.5 - cc[0]) + (c + 0.5 - cc[1]) * (c + 0.5 - cc[1]), e = Math.exp(-d / (2 * ATT * ATT));
      w.push(e); s += e;
    }
    return w.map(function (v) { return v / s; });
  });
  function compress(X) { return ATTW.map(function (w) { var s = 0; for (var p = 0; p < G * G; p++) s += w[p] * X[p]; return s; }); }
  function video(kind) {
    var path = PATHS[kind], X = [], Z = [];
    for (var t = 1; t <= T; t++) { X.push(features(path, t)); Z.push(compress(X[t - 1])); }
    return { kind: kind, path: path, X: X, Z: Z };
  }
  var VID = { right: video('right'), left: video('left'), stand: video('stand'), nobody: video('nobody') };
  var MAIN = VID.right;
  var ZMIN = Infinity, ZMAX = -Infinity;
  Object.keys(VID).forEach(function (k) { VID[k].Z.forEach(function (z) { z.forEach(function (v) { ZMIN = Math.min(ZMIN, v); ZMAX = Math.max(ZMAX, v); }); }); });
  function zop(z) { return 0.14 + 0.86 * clamp((z - ZMIN) / (ZMAX - ZMIN)); }
  function d2(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]); return s; }

  /* Training only: targets, the head's predictions from the tokens, and the loss over the valid pairs */
  var DT = [], DH = [], ERR = [];
  for (var t0 = 0; t0 < NPAIR; t0++) DT.push(MAIN.X[t0 + HZ].map(function (v, i) { return v - MAIN.X[t0][i]; }));
  (function () {
    var nn = 0, i, j;
    for (i = 0; i < NPAIR; i++) { var m = Infinity; for (j = 0; j < NPAIR; j++) if (i !== j) m = Math.min(m, Math.sqrt(d2(MAIN.Z[i], MAIN.Z[j]))); nn += m; }
    var tau = 0.8 * nn / NPAIR;
    for (i = 0; i < NPAIR; i++) {
      var wsum = 0, acc = [];
      for (var p = 0; p < G * G; p++) acc.push(0);
      for (j = 0; j < NPAIR; j++) {
        var k = Math.exp(-d2(MAIN.Z[i], MAIN.Z[j]) / (2 * tau * tau)); wsum += k;
        for (p = 0; p < G * G; p++) acc[p] += k * DT[j][p];
      }
      DH.push(acc.map(function (v) { return v / wsum; }));
      ERR.push(d2(DH[i], DT[i]) / (G * G));
    }
  })();
  var DMAX = 0; DT.forEach(function (d) { d.forEach(function (v) { DMAX = Math.max(DMAX, Math.abs(v)); }); });
  var EMAX = Math.max.apply(null, ERR);
  function meanErr(n) { var s = 0; for (var i = 0; i < n; i++) s += ERR[i]; return n ? s / n : 0; }

  /* Online reranking: four candidates in the order of their stage-1 prior */
  function gelu(x) { return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
  var CLABEL = { right: 'Person walks right', left: 'Person walks left', stand: 'Person stands still', nobody: 'Empty field' };
  var CANDS = [['left', 0.74], ['right', 0.71], ['stand', 0.62], ['nobody', 0.40]].map(function (c) { return { vid: VID[c[0]], rho: c[1], label: CLABEL[c[0]], match: c[0] === 'right' }; });
  (function () {
    var base = VID.nobody.Z, refP = 0;
    function motion(v) {
      var us = [], ps = [];
      v.Z.forEach(function (z, t) {
        var d = z.map(function (x, q) { return Math.max(0, x - base[t][q]); }), s = d[0] + d[1] + d[2] + d[3];
        ps.push(s); us.push(s > 1e-6 ? (d[1] + d[3] - d[0] - d[2]) / s : 0);
      });
      var ub = 0, tb = (T - 1) / 2, num = 0, den = 0;
      us.forEach(function (u) { ub += u / T; });
      us.forEach(function (u, t) { num += (u - ub) * (t - tb); den += (t - tb) * (t - tb); });
      return { m: num / den * (T - 1) / 2, p: ps.reduce(function (a, b) { return a + b; }, 0) / T };
    }
    VID.right.Z.forEach(function (z, t) { refP = Math.max(refP, z.reduce(function (a, b, q) { return a + b - base[t][q]; }, 0)); });
    var mref = motion(VID.right).m;
    CANDS.forEach(function (cd) {
      var j = motion(cd.vid), m = Math.max(-1, Math.min(1, j.m / mref)), pr = Math.min(1, j.p / refP);
      /* joint representation c(q, v): presence of the person and the direction of motion read from the ordered tokens */
      cd.c = [pr * Math.max(m, 0), pr * Math.max(-m, 0), pr * (1 - Math.abs(m)), 1 - pr];
      cd.e = [0.5 * cd.rho, 0.5 * cd.rho, 0.5 * cd.rho, 0];
      var ct = cd.c.map(function (x, i) { return x + cd.e[i]; }), W = [3.2, -1.4, 0.6, -2.2], z = -0.9;
      ct.forEach(function (x, i) { z += W[i] * gelu(x); });
      cd.s = 1 / (1 + Math.exp(-z));
    });
  })();
  var ORDER = CANDS.map(function (c, i) { return i; }).sort(function (a, b) { return CANDS[b].s - CANDS[a].s; });
  var RANK = []; ORDER.forEach(function (ci, r) { RANK[ci] = r; });
  var QUERY = ['a', 'person', 'walks', 'to', 'the', 'right'];

  /* ------------------------------------------------------------------ */
  /* Drawing helpers                                                     */
  /* ------------------------------------------------------------------ */
  var L = { narrow: false, W: 960, F: 1 }, LH = 1.27;
  function E(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    if (attrs) for (var a in attrs) if (attrs[a] != null) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function attr(n, a, v) { var k = '_' + a; if (n[k] === v) return; n[k] = v; n.setAttribute(a, v); }
  function op(n, v) { attr(n, 'opacity', Math.max(0, Math.min(1, v)).toFixed(3)); }
  function on(n, name, flag) { if (!!n['_c' + name] === !!flag) return; n['_c' + name] = !!flag; n.classList.toggle(name, !!flag); }
  function ease(x) { return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; }
  function span(p, a, b) { return ease(clamp((p - a) / (b - a))); }
  function fs(kind) { return ({ t: 13, s: 11.5, c: 16, o: 17, m: 15 })[kind] * L.F; }
  function rnd(v) { return Math.round(v * 10) / 10; }

  /* text with inline math between $ signs: Latin and lowercase Greek letters are italic (KaTeX_Math), digits,
     brackets and capital Greek upright (KaTeX_Main); _x or _{...} is a subscript, _{~word} an upright one, and
     the script L of the loss is written as ℒ */
  var SUBF = 0.78, MATHF = 1.1, SHIFT = 0.24;
  function isItalic(ch) { return /[A-Za-zα-ωϕ]/.test(ch); }
  function mathRuns(runs, s, sub) {
    if (s.charAt(0) === '~') { runs.push({ s: s.slice(1), k: 'mn', sub: sub }); return; }
    var i = 0;
    while (i < s.length) {
      var ch = s.charAt(i);
      if (ch === 'ℒ') { runs.push({ s: 'L', k: 'mc', sub: sub }); i++; continue; }
      var it = isItalic(ch), j = i + 1;
      while (j < s.length && s.charAt(j) !== 'ℒ' && isItalic(s.charAt(j)) === it) j++;
      runs.push({ s: s.slice(i, j), k: it ? 'mi' : 'mn', sub: sub });
      i = j;
    }
  }
  function fill(n, src) {
    while (n.firstChild) n.removeChild(n.firstChild);
    var runs = [];
    String(src).split('$').forEach(function (part, i) {
      if (!part) return;
      if (i % 2 === 0) { runs.push({ s: part, k: 'tx', sub: false }); return; }
      var k = 0;
      while (k < part.length) {
        if (part.charAt(k) === '_') {
          var sub;
          if (part.charAt(k + 1) === '{') { var e = part.indexOf('}', k + 2); sub = part.slice(k + 2, e); k = e + 1; }
          else { sub = part.charAt(k + 1); k += 2; }
          mathRuns(runs, sub, true);
        } else {
          var j = k; while (j < part.length && part.charAt(j) !== '_') j++;
          mathRuns(runs, part.slice(k, j), false); k = j;
        }
      }
    });
    var cur = 0;
    runs.forEach(function (r) {
      var size = r.sub ? SUBF : r.k === 'tx' ? 1 : MATHF, target = r.sub ? SHIFT : 0;
      var ts = document.createElementNS(NS, 'tspan');
      ts.setAttribute('class', 'va-' + r.k + (r.sub ? ' va-sub' : ''));
      if (target !== cur) { ts.setAttribute('dy', ((target - cur) / size).toFixed(3) + 'em'); cur = target; }
      ts.textContent = r.s;
      n.appendChild(ts);
    });
    n._src = src;
  }
  function text(parent, x, y, src, cls, anchor) {
    var n = E('text', { x: rnd(x), y: rnd(y), 'class': cls || 'va-t', 'text-anchor': anchor || 'start' }, parent);
    fill(n, src);
    return n;
  }
  function retext(n, src) { if (n._src !== src) fill(n, src); }
  var meas = null;
  function measure(src, cls) {
    if (!meas || meas.parentNode !== svg) meas = E('text', { x: 0, y: -50, visibility: 'hidden', 'aria-hidden': 'true' }, svg);
    meas.setAttribute('class', cls || 'va-t'); fill(meas, src);
    var w = 0;
    try { w = meas.getComputedTextLength(); } catch (e) { w = 0; }
    if (!(w > 0)) {
      var k = /va-small/.test(cls) ? 's' : /va-cnt/.test(cls) ? 'c' : 't';
      w = String(src).replace(/[$_{}~]/g, '').length * 0.57 * fs(k);
    }
    return w;
  }
  function widest(lines, cls) { var w = 0; lines.forEach(function (s) { w = Math.max(w, measure(s, cls)); }); return w; }

  /* a component box: kind 'frozen' (gray, the frozen encoder and the first stage) or 'vedje' (blue, VEDJE's own parts) */
  function boxSize(lines, padX, padY) {
    return { w: widest(lines, 'va-t va-bt') + 2 * (padX == null ? 14 : padX), h: lines.length * fs('t') * LH + 2 * (padY == null ? 9 : padY), lines: lines };
  }
  /* one line when it fits the width, otherwise the two-line version */
  function fit(one, two, maxW) { var s = boxSize([one]); return s.w <= maxW ? s : boxSize(two); }
  function box(parent, x, y, w, h, lines, kind) {
    var g = E('g', { 'class': 'va-box va-' + kind }, parent);
    var r = E('rect', { x: rnd(x), y: rnd(y), width: rnd(w), height: rnd(h), rx: Math.min(12, h / 4) }, g);
    var f = fs('t'), lh = f * LH, y0 = y + h / 2 - (lines.length - 1) * lh / 2 + f * 0.35;
    var txt = lines.map(function (s, i) { return text(g, x + w / 2, y0 + i * lh, s, 'va-t va-bt', 'middle'); });
    return { g: g, rect: r, x: x, y: y, w: w, h: h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h, txt: txt };
  }
  function arrow(parent, pts, cls) {
    var d = 'M' + pts.map(function (p) { return rnd(p[0]) + ',' + rnd(p[1]); }).join(' L');
    var n = E('path', { d: d, 'class': 'va-arr' + (cls ? ' ' + cls : ''), 'marker-end': 'url(#vaArrow)' }, parent);
    return n;
  }
  function flow(n, flag) { on(n, 'is-on', flag); attr(n, 'marker-end', flag ? 'url(#vaArrowOn)' : 'url(#vaArrow)'); }
  /* a 6 x 6 patch grid; color 'gray' (frozen features) or 'blue' (computed by VEDJE) */
  function grid(parent, x, y, cell, color) {
    var g = E('g', { 'class': 'va-grid va-grid-' + color }, parent), gap = Math.max(1, cell * 0.12), w = G * cell - gap;
    E('rect', { x: rnd(x - 2), y: rnd(y - 2), width: rnd(w + 4), height: rnd(w + 4), rx: 3, 'class': 'va-gridbg' }, g);
    var cells = [];
    for (var r = 0; r < G; r++) for (var c = 0; c < G; c++) {
      cells.push(E('rect', { x: rnd(x + c * cell), y: rnd(y + r * cell), width: rnd(cell - gap), height: rnd(cell - gap), rx: Math.max(1, cell * 0.12), 'class': 'va-cell', 'fill-opacity': 0, 'stroke-opacity': 0 }, g));
    }
    return { g: g, cells: cells, x: x, y: y, w: w, cx: x + w / 2, cy: y + w / 2, cell: cell, gap: gap };
  }
  /* positive values fill the cell; negative ones (a decrease) outline it */
  function setGrid(gr, vals, scale, signed) {
    gr.cells.forEach(function (n, i) {
      var v = vals ? vals[i] : 0, a = clamp(Math.abs(v) / scale);
      if (!signed || v >= 0) { attr(n, 'fill-opacity', a.toFixed(3)); attr(n, 'stroke-opacity', '0'); }
      else { attr(n, 'fill-opacity', (a * 0.1).toFixed(3)); attr(n, 'stroke-opacity', Math.min(1, a * 1.15).toFixed(3)); }
    });
  }
  /* the M = 4 tokens of one frame as a 2 x 2 group, laid out like the quadrants their queries attend to */
  function tok4(parent, cx, cy, s, gap, empty) {
    var g = E('g', { 'class': 'va-tok4' }, parent), x0 = cx - s - gap / 2, y0 = cy - s - gap / 2, r = [];
    var slot = empty ? E('rect', { x: rnd(x0 - 2), y: rnd(y0 - 2), width: rnd(2 * s + gap + 4), height: rnd(2 * s + gap + 4), rx: 3, 'class': 'va-slot' }, g) : null;
    for (var q = 0; q < 4; q++) r.push(E('rect', { x: rnd(x0 + (q % 2) * (s + gap)), y: rnd(y0 + Math.floor(q / 2) * (s + gap)), width: rnd(s), height: rnd(s), rx: Math.max(1, s * 0.2), 'class': 'va-tok', 'fill-opacity': 0 }, g));
    return { g: g, r: r, slot: slot, cx: cx, cy: cy, s: s, gap: gap, w: 2 * s + gap };
  }
  function setTok(t4, z, f) { t4.r.forEach(function (n, q) { attr(n, 'fill-opacity', z ? (zop(z[q]) * (f == null ? 1 : f)).toFixed(3) : '0'); }); }
  /* a token group that flies from one place to another (scale follows the size) */
  function flier(parent, s, gap) { var f = tok4(parent, 0, 0, s, gap, false); op(f.g, 0); return f; }
  /* along a quadratic curve through the control point c, which keeps the flight clear of the boxes */
  function flyTo(f, a, b, k, s0, s1, c) {
    var e = ease(clamp(k)), u = 1 - e, x = u * u * a[0] + 2 * u * e * c[0] + e * e * b[0], y = u * u * a[1] + 2 * u * e * c[1] + e * e * b[1], sc = (s0 + (s1 - s0) * e) / f.s;
    attr(f.g, 'transform', 'translate(' + rnd(x) + ',' + rnd(y) + ') scale(' + sc.toFixed(3) + ')');
  }
  /* a frame drawn like the cover's: sky, clouds, a distant tree line, grass, the sun and the person at its position at
     frame t. The layout matches the toy features: sky above, ground on the lower third, the sun in the top-right
     patch, the person's footprint around row 3.35. */
  function frame(parent, x, y, w, h, vid, patches) {
    var g = E('g', { 'class': 'va-frame' }, parent), rr = Math.max(1.5, h * 0.07), gy = y + h * 4 / 6, u = h / 6;
    E('rect', { x: rnd(x), y: rnd(y), width: rnd(w), height: rnd(h), rx: rnd(rr), 'class': 'va-sky' }, g);
    E('ellipse', { cx: rnd(x + w * 0.27), cy: rnd(y + h * 0.17), rx: rnd(w * 0.16), ry: rnd(h * 0.045), 'class': 'va-cloud' }, g);
    E('ellipse', { cx: rnd(x + w * 0.38), cy: rnd(y + h * 0.125), rx: rnd(w * 0.09), ry: rnd(h * 0.04), 'class': 'va-cloud' }, g);
    [[0.05, 0.07, 0.05], [0.18, 0.08, 0.065], [0.31, 0.06, 0.045], [0.46, 0.085, 0.06], [0.6, 0.065, 0.05], [0.74, 0.07, 0.055], [0.9, 0.08, 0.06]]
      .forEach(function (b) { E('ellipse', { cx: rnd(x + w * b[0]), cy: rnd(gy), rx: rnd(w * b[1]), ry: rnd(h * b[2]), 'class': 'va-far' }, g); });
    E('path', { d: 'M' + rnd(x) + ',' + rnd(gy) + ' H' + rnd(x + w) + ' V' + rnd(y + h - rr) + ' Q' + rnd(x + w) + ',' + rnd(y + h) + ' ' + rnd(x + w - rr) + ',' + rnd(y + h) +
      ' H' + rnd(x + rr) + ' Q' + rnd(x) + ',' + rnd(y + h) + ' ' + rnd(x) + ',' + rnd(y + h - rr) + ' Z', 'class': 'va-ground' }, g);
    E('circle', { cx: rnd(x + w * 5.45 / 6), cy: rnd(y + h * 0.6 / 6), r: rnd(h * 0.12), 'class': 'va-halo' }, g);
    E('circle', { cx: rnd(x + w * 5.45 / 6), cy: rnd(y + h * 0.6 / 6), r: rnd(h * 0.075), 'class': 'va-sun' }, g);
    var fig = null, P = {};
    if (vid.path) {
      fig = E('g', { 'class': 'va-person' }, g);
      E('ellipse', { cx: 0, cy: rnd(0.03 * u), rx: rnd(0.4 * u), ry: rnd(0.07 * u), 'class': 'va-shadow' }, fig);
      var lw = Math.max(0.7, 0.2 * u).toFixed(2), aw = Math.max(0.6, 0.14 * u).toFixed(2);
      P.legF = E('path', { 'class': 'va-limb va-leg is-far', 'stroke-width': lw }, fig);
      P.armF = E('path', { 'class': 'va-limb va-arm is-far', 'stroke-width': aw }, fig);
      P.shoeF = E('ellipse', { rx: rnd(0.1 * u), ry: rnd(0.05 * u), 'class': 'va-shoe' }, fig);
      P.handF = E('circle', { r: rnd(0.06 * u), 'class': 'va-skin' }, fig);
      P.torso = E('path', { 'class': 'va-coat' }, fig);
      P.legN = E('path', { 'class': 'va-limb va-leg', 'stroke-width': lw }, fig);
      P.shoeN = E('ellipse', { rx: rnd(0.1 * u), ry: rnd(0.05 * u), 'class': 'va-shoe' }, fig);
      P.armN = E('path', { 'class': 'va-limb va-arm', 'stroke-width': aw }, fig);
      P.handN = E('circle', { r: rnd(0.06 * u), 'class': 'va-skin' }, fig);
      E('ellipse', { cx: rnd(0.03 * u), cy: rnd(-1.66 * u), rx: rnd(0.165 * u), ry: rnd(0.19 * u), 'class': 'va-skin' }, fig);
      E('ellipse', { cx: rnd(-0.01 * u), cy: rnd(-1.77 * u), rx: rnd(0.165 * u), ry: rnd(0.1 * u), 'class': 'va-hair' }, fig);
    }
    if (patches) for (var k = 1; k < G; k++) {
      E('line', { x1: rnd(x + k * w / G), x2: rnd(x + k * w / G), y1: rnd(y), y2: rnd(y + h), 'class': 'va-patchline' }, g);
      E('line', { y1: rnd(y + k * h / G), y2: rnd(y + k * h / G), x1: rnd(x), x2: rnd(x + w), 'class': 'va-patchline' }, g);
    }
    E('rect', { x: rnd(x), y: rnd(y), width: rnd(w), height: rnd(h), rx: rnd(rr), 'class': 'va-frameline' }, g);
    var o = { g: g, x: x, y: y, w: w, h: h, cx: x + w / 2, cy: y + h / 2, b: y + h, r: x + w, vid: vid };
    function pt(p) { return rnd(p[0]) + ',' + rnd(p[1]); }
    function limb(n, a, b, c) { attr(n, 'd', 'M' + pt(a) + ' L' + pt(b) + ' L' + pt(c)); }
    function at(n, p, kx, ky) { attr(n, kx || 'cx', rnd(p[0])); attr(n, ky || 'cy', rnd(p[1])); }
    o.set = function (t, vd) {
      if (vd) o.vid = vd;
      if (!fig) return;
      var v = o.vid, walk = v.kind === 'right' || v.kind === 'left', dir = v.kind === 'left' ? -1 : 1;
      attr(fig, 'transform', 'translate(' + rnd(x + v.path(t) / G * w) + ',' + rnd(y + h * 4.12 / 6) + ') scale(' + dir + ',1)');
      var hip = [0, -0.82 * u], sh = [0, -1.36 * u];
      if (walk) {   /* side view: thighs swing with the stride, the trailing knee bends, the arms swing against the legs */
        var sw = Math.sin(t * 1.9);
        [[-0.36 * sw, P.legF, P.shoeF], [0.36 * sw, P.legN, P.shoeN]].forEach(function (l) {
          var a = l[0], bb = a < 0 ? a - 0.22 : a - 0.08, kn = [0.42 * u * Math.sin(a), hip[1] + 0.42 * u * Math.cos(a)];
          var ft = [kn[0] + 0.43 * u * Math.sin(bb), Math.min(0, kn[1] + 0.43 * u * Math.cos(bb))];
          limb(l[1], hip, kn, ft); at(l[2], [ft[0] + 0.06 * u, ft[1]]);
        });
        [[0.42 * sw, P.armF, P.handF], [-0.42 * sw, P.armN, P.handN]].forEach(function (l) {
          var a = l[0], el = [0.27 * u * Math.sin(a), sh[1] + 0.27 * u * Math.cos(a)];
          var hd = [el[0] + 0.25 * u * Math.sin(a + 0.35), el[1] + 0.25 * u * Math.cos(a + 0.35)];
          limb(l[1], sh, el, hd); at(l[2], hd);
        });
        attr(P.torso, 'd', 'M' + pt([-0.17 * u, -1.43 * u]) + ' L' + pt([0.17 * u, -1.43 * u]) + ' L' + pt([0.155 * u, -0.79 * u]) + ' L' + pt([-0.155 * u, -0.79 * u]) + ' Z');
      } else {      /* front view, standing still with the arms at the sides */
        [[-1, P.legF, P.shoeF], [1, P.legN, P.shoeN]].forEach(function (l) {
          var s = l[0], top = [s * 0.08 * u, hip[1]], ft = [s * 0.09 * u, -0.02 * u];
          limb(l[1], top, [s * 0.085 * u, -0.41 * u], ft); at(l[2], [s * 0.1 * u, 0]);
        });
        [[-1, P.armF, P.handF], [1, P.armN, P.handN]].forEach(function (l) {
          var s = l[0], top = [s * 0.21 * u, -1.36 * u], el = [s * 0.24 * u, -1.1 * u], hd = [s * 0.26 * u, -0.86 * u];
          limb(l[1], top, el, hd); at(l[2], hd);
        });
        attr(P.torso, 'd', 'M' + pt([-0.22 * u, -1.43 * u]) + ' L' + pt([0.22 * u, -1.43 * u]) + ' L' + pt([0.19 * u, -0.79 * u]) + ' L' + pt([-0.19 * u, -0.79 * u]) + ' Z');
      }
    };
    o.set(1);
    return o;
  }
  /* a candidate's label; the video that matches the query carries a check mark */
  function candLabel(parent, x, y, cd) {
    var dx = 0;
    if (cd.match) { E('path', { d: 'M' + rnd(x) + ',' + rnd(y - 4) + ' l3,3 l5.5,-6.5', 'class': 'va-check' }, parent); dx = 12; }
    return text(parent, x + dx, y, cd.label, 'va-t va-small va-clab');
  }
  function ring(parent, pad) { var n = E('rect', { 'class': 'va-ring', rx: 4, opacity: 0 }, parent); n._pad = pad || 3; return n; }
  function ringAt(n, x, y, w, h) { var p = n._pad; attr(n, 'x', rnd(x - p)); attr(n, 'y', rnd(y - p)); attr(n, 'width', rnd(w + 2 * p)); attr(n, 'height', rnd(h + 2 * p)); }
  function brace(parent, x1, x2, y) {
    var k = 5, xm = (x1 + x2) / 2;
    return E('path', { d: 'M' + rnd(x1) + ',' + rnd(y) + ' q0,' + k + ' ' + k + ',' + k + ' H' + rnd(xm - k) + ' q' + k + ',0 ' + k + ',' + k +
      ' q0,-' + k + ' ' + k + ',-' + k + ' H' + rnd(x2 - k) + ' q' + k + ',0 ' + k + ',-' + k, 'class': 'va-brace' }, parent);
  }
  /* a short vector of cells (a pooled representation) */
  function vec(parent, x, y, s, n) {
    var g = E('g', { 'class': 'va-vec' }, parent), cells = [];
    E('rect', { x: rnd(x - 2), y: rnd(y - 2), width: rnd(n * (s + 2) + 2), height: rnd(s + 4), rx: 3, 'class': 'va-gridbg' }, g);
    for (var i = 0; i < n; i++) cells.push(E('rect', { x: rnd(x + i * (s + 2)), y: rnd(y), width: rnd(s), height: rnd(s), rx: 2, 'class': 'va-tok', 'fill-opacity': 0 }, g));
    return { g: g, cells: cells, x: x, y: y, w: n * (s + 2) - 2, h: s, cx: x + (n * (s + 2) - 2) / 2, cy: y + s / 2 };
  }
  function setVec(v, vals, f) { v.cells.forEach(function (n, i) { attr(n, 'fill-opacity', vals ? (clamp(0.12 + 0.88 * vals[i]) * (f == null ? 1 : f)).toFixed(3) : '0'); }); }
  function plus(parent, cx, cy, r) {
    var g = E('g', { 'class': 'va-plus' }, parent);
    E('circle', { cx: rnd(cx), cy: rnd(cy), r: rnd(r) }, g);
    E('path', { d: 'M' + rnd(cx - r * 0.5) + ',' + rnd(cy) + ' H' + rnd(cx + r * 0.5) + ' M' + rnd(cx) + ',' + rnd(cy - r * 0.5) + ' V' + rnd(cy + r * 0.5) }, g);
    return { g: g, cx: cx, cy: cy, r: r };
  }
  /* the hat of a predicted delta, placed over the first glyph of its label once the text is laid out */
  var HATS = [];
  function hatText(parent, x, y, sub, cls, anchor) {
    var n = text(parent, x, y, '$Δ_{' + sub + '}$', cls, anchor);
    var hat = E('path', { 'class': 'va-hat' }, parent);
    HATS.push({ n: n, hat: hat, x: x, y: y, anchor: anchor || 'start' });
    return n;
  }
  function placeHats() {
    HATS.forEach(function (h) {
      var f = fs('t') * MATHF, x0, w;
      try { var ex = h.n.getExtentOfChar(0); x0 = ex.x; w = ex.width; } catch (e) { x0 = null; }
      if (x0 == null || !(w > 0)) {
        var tw = measure(h.n._src, h.n.getAttribute('class')); w = 0.833 * f;
        x0 = h.anchor === 'middle' ? h.x - tw / 2 : h.anchor === 'end' ? h.x - tw : h.x;
      }
      attr(h.hat, 'd', 'M' + rnd(x0 + 0.1 * w) + ',' + rnd(h.y - 0.74 * f) + ' L' + rnd(x0 + 0.5 * w) + ',' + rnd(h.y - 0.9 * f) + ' L' + rnd(x0 + 0.9 * w) + ',' + rnd(h.y - 0.74 * f));
    });
  }

  /* ------------------------------------------------------------------ */
  /* Scene A: offline indexing (Figure 2A, Section 3.2, Eqs. 1 and 2)    */
  /* ------------------------------------------------------------------ */
  var A0 = 400, A1 = 2150, AS = 380;
  function aStart(t) { return t === 1 ? A0 : A1 + (t - 2) * AS; }
  function aFlyEnd(t) { return t === 1 ? A0 + 1700 : aStart(t) + 370; }
  function sceneIndex(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), fC = fs('c'), lh = fT * LH;
    var o = { title: 'Offline indexing.', text: 'The frozen visual encoder runs once per video. Each sampled frame is compressed into a few tokens, kept in temporal order.', dur: 8700 };
    var mx = 20, yLab = 4 + fS, yStrip = yLab + 8, thumbs = [], xs = [], ysT = [], tw, th, pitch;
    var gb, cb, gr, cell, tk, ts, slots = [], fr = null, labs = {}, arrows = [], H0;
    var sb = { lines: ['stored', 'per video'] }, stored, cnt, per;
    text(g, mx, yLab, 'frames $f_1, …, f_T$', 'va-t va-small va-lab');
    if (!N) {
      var SW = Math.max(widest(sb.lines, 'va-t va-bt'), measure('48 KiB', 'va-t va-cnt'), measure('3 KiB per frame', 'va-t va-small')) + 32;
      SW = Math.max(SW, 150);
      var sxx = W - mx - SW, XL = sxx - 38, leftW = XL - mx;
      pitch = (leftW + 4) / T; tw = pitch - 4; th = Math.round(tw * 0.74);
      for (var i = 0; i < T; i++) { xs.push(mx + i * pitch); ysT.push(yStrip); }
      labs.t = text(g, XL, yLab, '$t = 1$', 'va-t va-small va-idx', 'end');
      var fw = 74, fh = 55; cell = 12; ts = 14;
      gb = boxSize(['frozen visual', 'encoder $g_ϕ$']); cb = boxSize(['frame-indexed', 'compressor $C_η$']);
      var bh = Math.max(gb.h, cb.h), gw = G * cell - Math.max(1, cell * 0.12), tokW = 2 * ts + 3;
      var fixed = fw + gb.w + gw + cb.w + tokW, gap = Math.max(26, Math.min(64, (leftW - fixed) / 4));
      var x0 = mx + (leftW - (fixed + 4 * gap)) / 2, tall = Math.max(fh, bh, gw), yC = yStrip + th + 30 + tall / 2;
      fr = frame(g, x0, yC - fh / 2, fw, fh, MAIN, true);
      var xG = x0 + fw + gap, xX = xG + gb.w + gap, xC = xX + gw + gap, xZ = xC + cb.w + gap;
      gb = box(g, xG, yC - bh / 2, gb.w, bh, gb.lines, 'frozen');
      gr = grid(g, xX, yC - gw / 2, cell, 'gray');
      cb = box(g, xC, yC - bh / 2, cb.w, bh, cb.lines, 'vedje');
      tk = tok4(g, xZ + tokW / 2, yC, ts, 3, false);
      arrows.push(arrow(g, [[fr.r + 4, yC], [gb.x - 4, yC]]), arrow(g, [[gb.r + 4, yC], [gr.x - 6, yC]]),
        arrow(g, [[gr.x + gw + 6, yC], [cb.x - 4, yC]]), arrow(g, [[cb.r + 4, yC], [tk.cx - tk.w / 2 - 6, yC]]));
      var yLb = yC + tall / 2 + 8 + fT;
      text(g, fr.cx, yLb, '$f_t$', 'va-t va-sym', 'middle');
      text(g, gr.cx, yLb, '$X_t$', 'va-t va-sym', 'middle');
      labs.z = text(g, tk.cx, yLb, '$Z_t$', 'va-t va-sym va-blue', 'middle');
      var yZ = yLb + 28, s2 = 12;
      for (i = 0; i < T; i++) slots.push(tok4(g, xs[i] + tw / 2, yZ + s2 + 1, s2, 2, true));
      var yL2 = yZ + 2 * s2 + 2 + 8 + fT;
      [[0, '$Z_1$'], [1, '$Z_2$'], [2, '$Z_3$'], [T - 1, '$Z_T$']].forEach(function (a) { text(g, slots[a[0]].cx, yL2, a[1], 'va-t va-sym va-blue', 'middle'); });
      text(g, (slots[3].cx + slots[T - 2].cx) / 2, yL2, '$⋯$', 'va-t va-sym va-blue', 'middle');
      var x1 = slots[0].cx - slots[0].w / 2, x2 = slots[T - 1].cx + slots[T - 1].w / 2;
      brace(g, x1, x2, yL2 + 7);
      labs.cache = text(g, (x1 + x2) / 2, yL2 + 7 + 14 + fT, 'ordered cache $Z(v) = [Z_1; …; Z_T]$', 'va-t va-cachelab', 'middle');
      var sh = 14 + 2 * lh + 8 + fC + 8 + fS + 12, scy = yZ + s2 + 1;
      stored = box(g, sxx, Math.max(yStrip + th + 26, scy - sh / 2), SW, sh, [], 'vedje');
      text(stored.g, stored.cx, stored.y + 14 + fT * 0.8, 'stored', 'va-t va-bt', 'middle');
      text(stored.g, stored.cx, stored.y + 14 + fT * 0.8 + lh, 'per video', 'va-t va-bt', 'middle');
      cnt = text(stored.g, stored.cx, stored.y + 14 + 2 * lh + 8 + fC * 0.8, '0 KiB', 'va-t va-cnt', 'middle');
      per = text(stored.g, stored.cx, stored.b - 12, '3 KiB per frame', 'va-t va-small', 'middle');
      arrows.push(arrow(g, [[x2 + 8, scy], [stored.x - 4, scy]]));
      H0 = Math.max(yL2 + 7 + 14 + fT + 8, stored.b + 6);
    } else {
      pitch = (W - 2 * mx + 6) / 8; tw = pitch - 6; th = Math.round(tw * 0.74);
      for (i = 0; i < T; i++) { xs.push(mx + (i % 8) * pitch); ysT.push(yStrip + Math.floor(i / 8) * (th + 6)); }
      labs.t = text(g, W - mx, yLab, '$t = 1$', 'va-t va-small va-idx', 'end');
      var cxN = W / 2, y = yStrip + 2 * th + 6 + 30, maxW = W - 2 * mx;
      gb = fit('frozen visual encoder $g_ϕ$', ['frozen visual', 'encoder $g_ϕ$'], maxW);
      gb = box(g, cxN - gb.w / 2, y, gb.w, gb.h, gb.lines, 'frozen');
      cell = 12.5; var gwN = G * cell - Math.max(1, cell * 0.12);
      gr = grid(g, cxN - gwN / 2, gb.b + 24, cell, 'gray');
      text(g, gr.x - 12, gr.cy + fT * 0.38, '$X_t$', 'va-t va-sym', 'end');
      cb = fit('frame-indexed compressor $C_η$', ['frame-indexed', 'compressor $C_η$'], maxW);
      cb = box(g, cxN - cb.w / 2, gr.y + gwN + 24, cb.w, cb.h, cb.lines, 'vedje');
      ts = 14; tk = tok4(g, cxN, cb.b + 22 + ts + 1.5, ts, 3, false);
      labs.z = text(g, tk.cx - tk.w / 2 - 12, tk.cy + fT * 0.38, '$Z_t$', 'va-t va-sym va-blue', 'end');
      arrows.push(null, arrow(g, [[cxN, gb.b + 4], [cxN, gr.y - 6]]), arrow(g, [[cxN, gr.y + gwN + 6], [cxN, cb.y - 4]]), arrow(g, [[cxN, cb.b + 4], [cxN, tk.cy - tk.w / 2 - 5]]));
      var yZn = tk.cy + tk.w / 2 + 26, sN = 8.5, pN = (W - 2 * mx + 4) / T;
      for (i = 0; i < T; i++) slots.push(tok4(g, mx + i * pN + (pN - 4) / 2, yZn + sN + 1, sN, 1.6, true));
      var x1n = slots[0].cx - slots[0].w / 2, x2n = slots[T - 1].cx + slots[T - 1].w / 2, yb = yZn + 2 * sN + 1.6 + 6;
      brace(g, x1n, x2n, yb);
      var cl = measure('ordered cache $Z(v) = [Z_1; …; Z_T]$', 'va-t va-cachelab') <= maxW;
      var yl = yb + 14 + fT;
      if (cl) labs.cache = text(g, W / 2, yl, 'ordered cache $Z(v) = [Z_1; …; Z_T]$', 'va-t va-cachelab', 'middle');
      else { labs.cache = text(g, W / 2, yl, 'ordered cache', 'va-t va-cachelab', 'middle'); yl += lh; text(g, W / 2, yl, '$Z(v) = [Z_1; …; Z_T]$', 'va-t va-cachelab', 'middle'); }
      var shN = 12 + lh + 4 + fS * 1.25 + 10, sy = yl + 16;
      stored = box(g, mx, sy, W - 2 * mx, shN, [], 'vedje');
      text(stored.g, mx + 14, sy + 12 + fT * 0.85, 'stored per video', 'va-t va-bt');
      cnt = text(stored.g, W - mx - 14, sy + 12 + fT * 0.85, '0 KiB', 'va-t va-cnt', 'end');
      per = text(stored.g, mx + 14, sy + 12 + lh + 4 + fS * 0.85, '3 KiB per frame', 'va-t va-small');
      arrows.push(arrow(g, [[W / 2, yl + 4], [W / 2, sy - 4]]));
      H0 = stored.b + 6;
    }
    for (i = 0; i < T; i++) { thumbs.push(frame(g, xs[i], ysT[i], tw, th, MAIN, false)); thumbs[i].set(i + 1); }
    var rg = ring(g, 2.5);
    var drop = E('path', { 'class': 'va-drop', opacity: 0 }, g);
    var quads = [0, 1, 2, 3].map(function (q) {
      var c = gr.cell, x = gr.x + (q % 2) * 3 * c - 1.5, y = gr.y + Math.floor(q / 2) * 3 * c - 1.5;
      return E('rect', { x: rnd(x), y: rnd(y), width: rnd(3 * c - gr.gap + 3), height: rnd(3 * c - gr.gap + 3), rx: 3, 'class': 'va-quad', opacity: 0 }, g);
    });
    var fly = flier(g, ts, 3);
    o.height = H0;
    o.update = function (p) {
      op(g, span(p, 0, 300));
      var cur = 1, a = p - A0, slow = true;
      if (p >= A1) { cur = Math.min(T, 2 + Math.floor((p - A1) / AS)); a = p - aStart(cur); slow = false; }
      var started = p >= A0;
      /* stage windows in local time: frame shown, encoder, features, compressor, tokens, flight */
      var w = slow ? { enc: [180, 600], feat: 520, comp: [880, 1300], fly: [1350, 1700] } : { enc: [0, 120], feat: 60, comp: [110, 225], fly: [225, 370] };
      var th0 = thumbs[cur - 1];
      ringAt(rg, th0.x, th0.y, th0.w, th0.h); op(rg, started ? 1 : 0);
      if (fr) fr.set(cur);
      retext(labs.t, '$t = ' + cur + '$');
      if (fr) {
        attr(drop, 'd', 'M' + rnd(th0.cx) + ',' + rnd(th0.b + 3) + ' C' + rnd(th0.cx) + ',' + rnd(th0.b + 18) + ' ' + rnd(fr.cx) + ',' + rnd(fr.y - 18) + ' ' + rnd(fr.cx) + ',' + rnd(fr.y - 3));
      } else {
        attr(drop, 'd', 'M' + rnd(th0.cx) + ',' + rnd(th0.b + 3) + ' C' + rnd(th0.cx) + ',' + rnd(th0.b + 16) + ' ' + rnd(gb.cx) + ',' + rnd(gb.y - 16) + ' ' + rnd(gb.cx) + ',' + rnd(gb.y - 3));
      }
      op(drop, fr && started && slow ? span(a, 0, 200) * (1 - span(a, 1500, 1750)) : 0);
      var encOn = started && a >= w.enc[0] && a < w.enc[1];
      on(gb.g, 'is-on', encOn);
      if (arrows[0]) flow(arrows[0], encOn && a < (slow ? 360 : 60));
      flow(arrows[1], started && a >= (slow ? 420 : 50) && a < (slow ? 640 : 110));
      var featOn = started && a >= w.feat;
      setGrid(gr, featOn || !slow ? MAIN.X[(featOn ? cur : cur - 1) - 1] : null, 1, false);
      op(gr.g, 1);
      if (slow) gr.cells.forEach(function (n) { op(n, started ? span(a, w.feat, w.feat + 260) : 0); });
      else gr.cells.forEach(function (n) { op(n, 1); });
      var compOn = started && a >= w.comp[0] && a < w.comp[1];
      on(cb.g, 'is-on', compOn);
      flow(arrows[2], compOn && a < w.comp[0] + (slow ? 120 : 40));
      flow(arrows[3], started && a >= w.comp[0] + (slow ? 60 : 40) && a < w.fly[0]);
      /* each query attends to its quadrant of this frame's patches: one at a time for the first frame */
      quads.forEach(function (n, q) {
        var v = 0;
        if (slow && compOn) v = a >= w.comp[0] + q * 100 && a < w.comp[0] + (q + 1) * 100 + 40 ? 1 : 0;
        else if (!slow && compOn) v = 0.85;
        op(n, v);
      });
      var tokShown = [0, 1, 2, 3].map(function (q) { return started && a >= w.comp[0] + (slow ? q * 100 + 60 : 40); });
      tk.r.forEach(function (n, q) { attr(n, 'fill-opacity', tokShown[q] ? zop(MAIN.Z[cur - 1][q]).toFixed(3) : '0'); });
      var k = (a - w.fly[0]) / (w.fly[1] - w.fly[0]);
      if (started && k > 0 && k < 1) {
        setTok(fly, MAIN.Z[cur - 1]); op(fly.g, 1);
        flyTo(fly, [tk.cx, tk.cy], [slots[cur - 1].cx, slots[cur - 1].cy], k, tk.s, slots[cur - 1].s, [tk.cx, slots[cur - 1].cy]);
      } else op(fly.g, 0);
      var filled = 0;
      slots.forEach(function (s, i) { var f = p >= aFlyEnd(i + 1); setTok(s, f ? MAIN.Z[i] : null); if (f) filled++; });
      retext(cnt, 3 * filled + ' KiB');
      var fin = span(p, 7900, 8300);
      on(stored.g, 'is-on', p >= 7900);
      on(labs.cache, 'is-strong', p >= 7900);
      op(per, 0.55 + 0.45 * fin);
    };
    return o;
  }

  /* ------------------------------------------------------------------ */
  /* Scene B: training only (Figure 2B, Section 3.3, Eqs. 3 and 4)       */
  /* ------------------------------------------------------------------ */
  var B0 = 400, B1 = 2350, BS = 390, BX = 7300;
  function bStart(t) { return t === 1 ? B0 : B1 + (t - 2) * BS; }
  function sceneTrain(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), lh = fT * LH;
    var o = { title: 'Training only.', text: "A future-delta head predicts, from one frame's tokens, how the frozen patch features change a few sampled frames later. It is discarded after training and adds no work at query time.", dur: 8600 };
    var mx = 20, yLab = 4 + fS, slots = [], i, bx0 = mx, bx1 = W - mx;
    var sN = N ? 8.5 : 9, sg = 1.6, pitch = N ? (W - 2 * mx + 4) / T : Math.min(27, (440 + 7) / T);
    var yS = yLab + 8, cacheLab = text(g, mx, yLab, 'ordered cache $Z(v)$', 'va-t va-small va-lab');
    var idx = text(g, W - mx, yLab, '$t = 1$', 'va-t va-small va-idx', 'end');
    var rg = ring(g, 2.5);
    var ts = 14, cell = N ? 12.5 : 13, gw = G * cell - Math.max(1, cell * 0.12), tokW = 2 * ts + 3, fb = boxSize(['future-delta', 'head $F_ω$']);
    var opW = 30, rightW = 3 * gw + 2 * opW, gap = 64, leftW = tokW + fb.w + gw + 2 * gap, mid = 0;
    if (!N) {
      /* left the prediction from the tokens, right the target from the frozen features, centred together */
      mid = Math.min(150, W - 2 * mx - leftW - rightW);
      if (mid < 56) { gap = Math.max(28, (W - 2 * mx - rightW - 56 - (tokW + fb.w + gw)) / 2); leftW = tokW + fb.w + gw + 2 * gap; mid = W - 2 * mx - leftW - rightW; }
      bx0 = mx + (W - 2 * mx - (leftW + mid + rightW)) / 2; bx1 = bx0 + leftW + mid + rightW;
    }
    attr(cacheLab, 'x', rnd(bx0));
    for (i = 0; i < T; i++) { slots.push(tok4(g, bx0 + i * pitch + (pitch - 4) / 2, yS + sN + 1, sN, sg, true)); setTok(slots[i], MAIN.Z[i]); }
    var tk, gHat, gx1, gx0, gd, th1, th0, yC, sign = [], labsY;
    var fwT = N ? 40 : 46, fhT = Math.round(fwT * 0.74);
    if (!N) {
      yC = yS + 2 * sN + sg + 30 + Math.max(fb.h, gw) / 2 + 6;
      tk = tok4(g, bx0 + tokW / 2, yC, ts, 3, false);
      fb = box(g, bx0 + tokW + gap, yC - fb.h / 2, fb.w, fb.h, fb.lines, 'vedje');
      gHat = grid(g, fb.r + gap, yC - gw / 2, cell, 'blue');
      var rx0 = bx0 + leftW + mid;
      gx1 = grid(g, rx0, yC - gw / 2, cell, 'gray');
      gx0 = grid(g, rx0 + gw + opW, yC - gw / 2, cell, 'gray');
      gd = grid(g, rx0 + 2 * (gw + opW), yC - gw / 2, cell, 'gray');
      th1 = frame(g, gx1.cx - fwT / 2, yS - 2, fwT, fhT, MAIN, false);
      th0 = frame(g, gx0.cx - fwT / 2, yS - 2, fwT, fhT, MAIN, false);
      text(g, th1.x - 8, th1.cy + fT * 0.38, '$f_{t+h}$', 'va-t va-sym', 'end');
      text(g, th0.x - 8, th0.cy + fT * 0.38, '$f_t$', 'va-t va-sym', 'end');
      labsY = yC + gw / 2 + 8 + fT;
    } else {
      var gapN = Math.max(16, (W - 2 * mx - tokW - fb.w - gw) / 2);
      yC = yS + 2 * sN + sg + 30 + Math.max(fb.h, gw) / 2;
      tk = tok4(g, mx + tokW / 2, yC, ts, 3, false);
      fb = box(g, mx + tokW + gapN, yC - fb.h / 2, fb.w, fb.h, fb.lines, 'vedje');
      gHat = grid(g, W - mx - gw, yC - gw / 2, cell, 'blue');
      labsY = yC + gw / 2 + 8 + fT;
      var cell2 = 11.5, gw2 = G * cell2 - Math.max(1, cell2 * 0.12), rowR = gHat.cx - 22, opWn = (rowR - mx - 3 * gw2) / 2, yT = labsY + 22, yG2 = yT + fhT + 26;
      gx1 = grid(g, mx, yG2, cell2, 'gray');
      gx0 = grid(g, mx + gw2 + opWn, yG2, cell2, 'gray');
      gd = grid(g, rowR - gw2, yG2, cell2, 'gray');
      th1 = frame(g, gx1.cx - fwT / 2, yT, fwT, fhT, MAIN, false);
      th0 = frame(g, gx0.cx - fwT / 2, yT, fwT, fhT, MAIN, false);
      text(g, th1.r + 6, th1.cy + fT * 0.38, '$f_{t+h}$', 'va-t va-sym');
      text(g, th0.r + 6, th0.cy + fT * 0.38, '$f_t$', 'va-t va-sym');
    }
    var aIn = arrow(g, [[tk.cx + tk.w / 2 + 6, yC], [fb.x - 4, yC]]), aOut = arrow(g, [[fb.r + 4, yC], [gHat.x - 6, yC]]);
    var aT1 = arrow(g, [[th1.cx, th1.b + 4], [gx1.cx, gx1.y - 6]]), aT0 = arrow(g, [[th0.cx, th0.b + 4], [gx0.cx, gx0.y - 6]]);
    text(g, tk.cx, labsY, '$Z_t$', 'va-t va-sym va-blue', 'middle');
    hatText(g, gHat.cx, labsY, 't,h', 'va-t va-sym va-blue', 'middle');
    var gyL = gx1.y + gx1.w + 8 + fT;
    text(g, gx1.cx, gyL, '$X_{t+h}$', 'va-t va-sym', 'middle');
    text(g, gx0.cx, gyL, '$X_t$', 'va-t va-sym', 'middle');
    text(g, N ? gd.x + gd.w : gd.cx, gyL, 'target $Δ_{t,h}$', 'va-t va-sym', N ? 'end' : 'middle');
    sign.push(text(g, (gx1.x + gx1.w + gx0.x) / 2, gx1.cy + fs('o') * 0.35, '$−$', 'va-t va-op', 'middle'));
    sign.push(text(g, (gx0.x + gx0.w + gd.x) / 2, gx1.cy + fs('o') * 0.35, '$=$', 'va-t va-op', 'middle'));
    var gone = text(g, fb.cx, fb.b + 8 + fS, 'Discarded after training', 'va-t va-small va-gone', 'middle'); op(gone, 0);
    /* the loss over the valid pairs (t, h) with t + h <= T */
    var py = Math.max(gyL, labsY) + 20, pw = bx1 - bx0;
    if (!N) attr(idx, 'x', rnd(bx1));
    var sub1 = 'mean squared error over the pairs with $t + h ≤ T$', twoLines = measure(sub1, 'va-t va-small') > pw - 40 - measure('$ℒ_{~delta}$', 'va-t va-loss');
    var ph = 12 + fs('m') * 1.1 + (twoLines ? 2 * fS * 1.3 : 0) + 14 + 22 + 8 + 30 + 14;
    var panel = box(g, bx0, py, pw, ph, [], 'vedje');
    on(panel.g, 'va-panel', true);
    text(g, bx0 + 14, py + 12 + fs('m') * 0.85, '$ℒ_{~delta}$', 'va-t va-loss');
    var lw = measure('$ℒ_{~delta}$', 'va-t va-loss');
    if (!twoLines) text(g, bx0 + 14 + lw + 12, py + 12 + fs('m') * 0.85, sub1, 'va-t va-small');
    else {
      text(g, bx0 + 14, py + 12 + fs('m') * 1.1 + fS * 1.1, 'mean squared error over the pairs', 'va-t va-small');
      text(g, bx0 + 14, py + 12 + fs('m') * 1.1 + fS * 2.4, 'with $t + h ≤ T$', 'va-t va-small');
    }
    var trY = py + ph - 14 - 30 - 8, tx0 = bx0 + 28, tx1 = bx1 - 28, tp = (tx1 - tx0) / (T - 1);
    function px(t) { return tx0 + (t - 1) * tp; }
    E('line', { x1: rnd(tx0), x2: rnd(tx1), y1: rnd(trY), y2: rnd(trY), 'class': 'va-track' }, g);
    var dots = [], arcs = [], bars = [];
    for (i = 1; i <= T; i++) dots.push(E('circle', { cx: rnd(px(i)), cy: rnd(trY), r: 2.6, 'class': 'va-dot' + (i > NPAIR ? ' is-out' : '') }, g));
    for (i = 1; i <= NPAIR; i++) {
      var xa = px(i), xb = px(i + HZ), hA = Math.min(20, (xb - xa) * 0.42);
      arcs.push(E('path', { d: 'M' + rnd(xa) + ',' + rnd(trY - 4) + ' C' + rnd(xa) + ',' + rnd(trY - 4 - hA) + ' ' + rnd(xb) + ',' + rnd(trY - 4 - hA) + ' ' + rnd(xb) + ',' + rnd(trY - 4), 'class': 'va-arc', opacity: 0 }, g));
      bars.push(E('rect', { x: rnd(xa - Math.min(5, tp * 0.28)), y: rnd(trY + 8), width: rnd(Math.min(10, tp * 0.56)), height: 0, rx: 1.5, 'class': 'va-errbar' }, g));
    }
    var BH = 30, meanLine = E('line', { x1: rnd(tx0 - 8), x2: rnd(px(NPAIR) + 10), y1: rnd(trY + 8), y2: rnd(trY + 8), 'class': 'va-mean', opacity: 0 }, g);
    var meanLab = text(g, px(NPAIR) + 16, trY + 8, 'mean', 'va-t va-small va-blue'); op(meanLab, 0);
    var aL1 = arrow(g, [[gHat.cx, labsY + 6], [gHat.cx, py - 4]]), aL2 = arrow(g, [[gd.cx, gyL + 6], [gd.cx, py - 4]]);
    var fly = flier(g, ts, 3);
    o.height = py + ph + 6;
    o.update = function (p) {
      op(g, span(p, 0, 300));
      var cur = 1, a = p - B0, slow = true;
      if (p >= B1) { cur = Math.min(NPAIR, 2 + Math.floor((p - B1) / BS)); a = p - bStart(cur); slow = false; }
      if (p >= BX) { cur = NPAIR; a = 9999; }
      var started = p >= B0;
      var w = slow ? { read: [0, 300], head: [300, 700], hat: 420, tgt: [700, 1100], dlt: 950, loss: [1150, 1650] } : { read: [0, 90], head: [70, 170], hat: 110, tgt: [130, 250], dlt: 200, loss: [250, 380] };
      retext(idx, '$t = ' + cur + '$');
      var sl = slots[cur - 1];
      ringAt(rg, sl.cx - sl.w / 2 - 1, sl.cy - sl.w / 2 - 1, sl.w + 2, sl.w + 2); op(rg, started && p < BX ? 1 : 0);
      var k = (a - w.read[0]) / (w.read[1] - w.read[0]);
      if (started && k > 0 && k < 1) { setTok(fly, MAIN.Z[cur - 1]); op(fly.g, 1); flyTo(fly, [sl.cx, sl.cy], [tk.cx, tk.cy], k, sl.s, tk.s, [tk.cx, sl.cy]); }
      else op(fly.g, 0);
      setTok(tk, started && a >= w.read[1] ? MAIN.Z[cur - 1] : null);
      var headOn = started && a >= w.head[0] && a < w.head[1];
      on(fb.g, 'is-on', headOn);
      flow(aIn, headOn && a < w.head[0] + (slow ? 160 : 50));
      flow(aOut, started && a >= w.hat - (slow ? 80 : 20) && a < w.head[1]);
      var hatOn = started && a >= w.hat;
      setGrid(gHat, hatOn ? DH[cur - 1] : (slow ? null : DH[cur - 2]), DMAX, true);
      if (slow) op(gHat.g, started ? 0.25 + 0.75 * span(a, w.hat, w.hat + 220) : 0.25); else op(gHat.g, 1);
      var tgtOn = started && a >= w.tgt[0];
      var tt = tgtOn ? cur : Math.max(1, cur - 1);
      th0.set(tt); th1.set(tt + HZ);
      op(th0.g, started ? 1 : 0.4); op(th1.g, started ? 1 : 0.4);
      flow(aT0, tgtOn && a < w.tgt[0] + (slow ? 200 : 60)); flow(aT1, tgtOn && a < w.tgt[0] + (slow ? 200 : 60));
      setGrid(gx0, started ? MAIN.X[tt - 1] : null, 1, false); setGrid(gx1, started ? MAIN.X[tt - 1 + HZ] : null, 1, false);
      if (slow) { op(gx0.g, 0.3 + 0.7 * span(a, w.tgt[0], w.tgt[0] + 220)); op(gx1.g, 0.3 + 0.7 * span(a, w.tgt[0], w.tgt[0] + 220)); }
      else { op(gx0.g, 1); op(gx1.g, 1); }
      var dOn = started && a >= w.dlt;
      setGrid(gd, dOn ? DT[cur - 1] : (slow ? null : DT[cur - 2]), DMAX, true);
      if (slow) op(gd.g, 0.25 + 0.75 * (started ? span(a, w.dlt, w.dlt + 220) : 0)); else op(gd.g, 1);
      sign.forEach(function (n) { op(n, slow ? 0.3 + 0.7 * (started ? span(a, w.tgt[0], w.tgt[0] + 200) : 0) : 1); });
      var lossOn = started && a >= w.loss[0] && p < BX;
      flow(aL1, lossOn && a < w.loss[1]); flow(aL2, lossOn && a < w.loss[1]);
      on(panel.g, 'is-on', lossOn && a < w.loss[1]);
      /* pairs done so far, and the bar of the current one growing */
      var done = started ? (a >= w.loss[0] ? cur : cur - 1) : 0;
      arcs.forEach(function (n, i) {
        var t = i + 1, v = t < cur || (t === cur && a >= w.loss[0]) ? 1 : 0;
        var isCur = t === cur && p < BX && started;
        op(n, v * (isCur ? 1 : 0.55)); on(n, 'is-cur', isCur);
        var hgt = 0;
        if (t < cur) hgt = 1; else if (t === cur && started) hgt = span(a, w.loss[0] + (slow ? 150 : 40), w.loss[1]);
        var h = ERR[i] / EMAX * BH * hgt;
        attr(bars[i], 'height', rnd(Math.max(0, h)));
      });
      dots.forEach(function (n, i) { on(n, 'is-cur', started && p < BX && (i + 1 === cur || i + 1 === cur + HZ)); });
      var mE = meanErr(done), my = trY + 8 + mE / EMAX * BH;
      attr(meanLine, 'y1', rnd(my)); attr(meanLine, 'y2', rnd(my)); attr(meanLab, 'y', rnd(my + fS * 0.35));
      op(meanLine, done ? 1 : 0); op(meanLab, done ? 1 : 0);
      /* after training the head is discarded */
      var gn = span(p, BX, BX + 600);
      op(fb.g, 1 - 0.62 * gn); op(aIn, 1 - 0.62 * gn); op(aOut, 1 - 0.62 * gn);
      on(fb.g, 'is-gone', p >= BX);
      op(gone, span(p, BX + 200, BX + 700));
    };
    return o;
  }

  /* ------------------------------------------------------------------ */
  /* Scene C: online reranking (Figure 2C, Section 3.4, Eq. 5)           */
  /* ------------------------------------------------------------------ */
  var C0 = 500, C1 = 2550, CS = 900, CR = 5450, CR1 = 6450;
  function cStart(k) { return k === 0 ? C0 : C1 + (k - 1) * CS; }
  function sceneRerank(g) {
    var N = L.narrow, W = L.W, fT = fs('t'), fS = fs('s'), lh = fT * LH, mx = 20, i;
    var o = { title: 'Online reranking.', text: "The joint encoder reads the query with each candidate's cache, and the embedded first-stage score is added to its output before the score head. The candidates are sorted by the final score, and indexed videos need no visual encoding at query time.", dur: 8600 };
    var NOG = 'no visual encoding at query time', yLab = 4 + fS;
    /* the joint input: the query tokens and the candidate's cached tokens */
    var pad = N ? 5 : 7, pgap = 5, ph = fS * 1.7, ip = 10;
    var pw = QUERY.map(function (w) { return measure(w, 'va-t va-small va-qt') + 2 * pad; });
    var pillsW = pw.reduce(function (a, b) { return a + b + pgap; }, -pgap);
    var inW = N ? W - 2 * mx - 2 * ip : Math.max(340, pillsW);
    var xIn = mx + ip, ey0 = 4, yQ = ey0 + ip + fS * 0.85;
    text(g, xIn, yQ, 'query $q$', 'va-t va-small va-lab');
    var px0 = xIn, py0 = yQ + 7, pos = [];
    pw.forEach(function (w) { if (px0 + w > xIn + inW + 0.5 && px0 > xIn) { px0 = xIn; py0 += ph + 6; } pos.push([px0, py0]); px0 += w + pgap; });
    var pills = pos.map(function (p, k) {
      var pg0 = E('g', { 'class': 'va-pill' }, g);
      E('rect', { x: rnd(p[0]), y: rnd(p[1]), width: rnd(pw[k]), height: rnd(ph), rx: rnd(ph / 2) }, pg0);
      text(pg0, p[0] + pw[k] / 2, p[1] + ph / 2 + fS * 0.36, QUERY[k], 'va-t va-small va-qt', 'middle');
      return pg0;
    });
    var yCl = py0 + ph + 12 + fS * 0.85;
    text(g, xIn, yCl, 'cache $Z(v)$', 'va-t va-small va-lab');
    var cs = 8, cg = 1.5, cpitch = (inW + 5) / T, cache = [];
    for (i = 0; i < T; i++) cache.push(tok4(g, xIn + i * cpitch + (cpitch - 5) / 2, yCl + 7 + cs + 1, cs, cg, true));
    var ey1 = yCl + 7 + 2 * cs + cg + 2 + ip, inR = mx + inW + 2 * ip;
    E('rect', { x: mx, y: ey0, width: rnd(inW + 2 * ip), height: rnd(ey1 - ey0), rx: 12, 'class': 'va-input' }, g);
    /* no frozen visual encoder on the query path: its symbol in a crossed box, and the fact */
    function nog(x, y, maxW) {
      var gg = E('g', { 'class': 'va-nog' }, g), sw = measure('$g_ϕ$', 'va-t va-sym') + 12, bh0 = fT * 1.45;
      E('rect', { x: rnd(x), y: rnd(y - fS * 0.36 - bh0 / 2), width: rnd(sw), height: rnd(bh0), rx: 4, 'class': 'va-nogbox' }, gg);
      text(gg, x + sw / 2, y - fS * 0.36 + fT * 0.3, '$g_ϕ$', 'va-t va-sym', 'middle');
      E('path', { d: 'M' + rnd(x + 2) + ',' + rnd(y - fS * 0.36 + bh0 / 2 - 2) + ' L' + rnd(x + sw - 2) + ',' + rnd(y - fS * 0.36 - bh0 / 2 + 2), 'class': 'va-strike' }, gg);
      var tx = x + sw + 8, w1 = measure(NOG, 'va-t va-small');
      if (sw + 8 + w1 <= maxW) { text(gg, tx, y, NOG, 'va-t va-small va-nogt'); return { g: gg, b: Math.max(y + fS * 0.3, y - fS * 0.36 + bh0 / 2), w: sw + 8 + w1 }; }
      text(gg, tx, y, 'no visual encoding', 'va-t va-small va-nogt');
      text(gg, tx, y + fS * 1.25, 'at query time', 'va-t va-small va-nogt');
      return { g: gg, b: y + fS * 1.25 + fS * 0.3, w: sw + 8 + measure('no visual encoding', 'va-t va-small') };
    }
    var enc, cvec, pl, hd, sVal, st1, st1v, er, evec, aIn, aC, aE1, aE2, aH, H1, vs = N ? 11 : 12, vW = 4 * (vs + 2) - 2, ngp;
    var vw0 = measure('0.00', 'va-t va-small va-mono');
    if (!N) {
      var eb = boxSize(['joint', 'encoder']), hb = boxSize(['score head', '$h_ψ$']), pr = 12;
      /* the note sits at the top right, on one line when it clears the c(q, v) label, else on two */
      var hx = W - mx - hb.w, plx = hx - 40 - pr, ccx = (inR + 36 + eb.w + plx - pr) / 2;
      var cRight = ccx + measure('$c(q, v)$', 'va-t va-sym') / 2, gw0 = measure('$g_ϕ$', 'va-t va-sym') + 20;
      var nw = gw0 + measure(NOG, 'va-t va-small'), nw2 = gw0 + measure('no visual encoding', 'va-t va-small');
      ngp = W - mx - nw >= cRight + 16 ? nog(W - mx - nw, yLab, nw + 1) : nog(W - mx - nw2, yLab, nw2 - 1);
      var yE = Math.min(ey1 - 14, Math.max((ey0 + ey1) / 2, ngp.b + 14 + Math.max(eb.h, hb.h) / 2));
      enc = box(g, inR + 36, yE - eb.h / 2, eb.w, eb.h, eb.lines, 'vedje');
      aIn = arrow(g, [[inR + 4, yE], [enc.x - 4, yE]]);
      hd = box(g, hx, yE - hb.h / 2, hb.w, hb.h, hb.lines, 'vedje');
      pl = plus(g, plx, yE, pr);
      cvec = vec(g, ccx - vW / 2, yE - vs - 10, vs, 4);
      text(g, cvec.cx, cvec.y - 8, '$c(q, v)$', 'va-t va-sym va-blue', 'middle');
      aC = arrow(g, [[enc.r + 4, yE], [pl.cx - pr - 4, yE]]);
      aH = arrow(g, [[pl.cx + pr + 4, yE], [hd.x - 4, yE]]);
      text(g, hd.cx, hd.b + 10 + fT, '$s_θ(q, v)$', 'va-t va-sym va-blue', 'middle');
      sVal = text(g, hd.cx, hd.b + 12 + fT + lh, '', 'va-t va-cnt', 'middle');
      var sb = boxSize(['first stage', '$ρ(q, v)$']), yS1 = Math.max(ey1 + 18, enc.b + 20);
      st1 = box(g, mx, yS1, sb.w, sb.h, sb.lines, 'frozen');
      st1v = text(g, st1.r + 15 + vw0 / 2, st1.cy - 7, '', 'va-t va-small va-mono', 'middle');
      var rb = boxSize(['$e_ρ(·)$']);
      er = box(g, st1.r + vw0 + 30, st1.cy - rb.h / 2, rb.w, rb.h, rb.lines, 'vedje');
      evec = vec(g, er.r + 24, st1.cy - vs - 9, vs, 4);
      aE1 = arrow(g, [[st1.r + 4, st1.cy], [er.x - 4, st1.cy]]);
      aE2 = arrow(g, [[er.r + 4, st1.cy], [pl.cx, st1.cy], [pl.cx, pl.cy + pr + 4]]);
      H1 = Math.max(st1.b, hd.b + 12 + fT + lh + 6);
    } else {
      var cxE = W - mx - 100, ebN = fit('joint encoder', ['joint', 'encoder'], 2 * (W - mx - cxE)), prN = 12;
      enc = box(g, cxE - ebN.w / 2, ey1 + 24, ebN.w, ebN.h, ebN.lines, 'vedje');
      aIn = arrow(g, [[cxE, ey1 + 4], [cxE, enc.y - 4]]);
      pl = plus(g, cxE, enc.b + 16 + vs + 22 + prN, prN);
      cvec = vec(g, cxE - 12 - vW, (enc.b + pl.cy - prN) / 2 - vs / 2, vs, 4);
      text(g, cxE + 12, cvec.cy + fT * 0.38, '$c(q, v)$', 'va-t va-sym va-blue');
      aC = arrow(g, [[cxE, enc.b + 4], [cxE, pl.cy - prN - 4]]);
      var hbN = fit('score head $h_ψ$', ['score head', '$h_ψ$'], 2 * (W - mx - cxE));
      hd = box(g, cxE - hbN.w / 2, pl.cy + prN + 22, hbN.w, hbN.h, hbN.lines, 'vedje');
      aH = arrow(g, [[cxE, pl.cy + prN + 4], [cxE, hd.y - 4]]);
      text(g, cxE - 8, hd.b + 10 + fT, '$s_θ(q, v)$', 'va-t va-sym va-blue', 'end');
      sVal = text(g, cxE + 8, hd.b + 10 + fT, '', 'va-t va-cnt');
      /* left column: the stage-1 score above e_rho, which feeds the + node from the left */
      var sbN = boxSize(['first stage', '$ρ(q, v)$']), rbN = boxSize(['$e_ρ(·)$']), colC = mx + Math.max(sbN.w, rbN.w) / 2;
      st1 = box(g, colC - sbN.w / 2, enc.y, sbN.w, sbN.h, sbN.lines, 'frozen');
      er = box(g, colC - rbN.w / 2, pl.cy - rbN.h / 2, rbN.w, rbN.h, rbN.lines, 'vedje');
      st1v = text(g, colC + 10, (st1.b + er.y) / 2 + fS * 0.36, '', 'va-t va-small va-mono');
      aE1 = arrow(g, [[colC, st1.b + 4], [colC, er.y - 4]]);
      aE2 = arrow(g, [[er.r + 4, pl.cy], [pl.cx - prN - 4, pl.cy]]);
      evec = vec(g, (er.r + pl.cx - prN) / 2 - vW / 2, pl.cy - vs - 9, vs, 4);
      /* the note goes into the free column left of the score head when it fits there, else below */
      var colW = hd.x - 12 - mx, nw2 = measure('$g_ϕ$', 'va-t va-sym') + 20 + measure('no visual encoding', 'va-t va-small');
      if (nw2 <= colW) { ngp = nog(mx, Math.max(er.b + 14 + fS, hd.y + fS), colW); H1 = Math.max(ngp.b, hd.b + 10 + fT) + 6; }
      else { ngp = nog(mx, hd.b + 10 + fT + 18 + fS, W - 2 * mx); H1 = ngp.b + 6; }
    }
    /* the candidate list: the same scene with different events, first in stage-1 order */
    var rows = [], slotPos = [], tg = 3, hdr = null, hdr2 = null, hRho = null, hS = null, mark = null, barRef;
    if (!N) {
      var yH = H1 + 20 + fS;
      hdr = text(g, mx, yH, 'candidates $C(q)$, sorted by $ρ(q, v)$', 'va-t va-small va-lab');
      hdr2 = text(g, mx, yH, 'candidates $C(q)$, sorted by $s_θ(q, v)$', 'va-t va-small va-lab'); op(hdr2, 0);
      var fwC = 38, fhC = 28, cardW = (W - 2 * mx - 3 * 16) / 4, lineB = fS * 1.45, cardH = 10 + fhC + 8 + 3 * lineB + 6, y0 = yH + 10;
      for (i = 0; i < CANDS.length; i++) slotPos.push([mx + i * (cardW + 16), y0]);
      var symW = Math.max(measure('$ρ$', 'va-t va-small va-sym'), measure('$s_θ$', 'va-t va-small va-sym'));
      var barX = 12 + symW + 8, barW = cardW - barX - 10 - vw0 - 10;
      CANDS.forEach(function (cd, ci) {
        var rg0 = E('g', { 'class': 'va-cand' }, g);
        var bx = E('rect', { x: 0, y: 0, width: rnd(cardW), height: rnd(cardH), rx: 10, 'class': 'va-card' }, rg0);
        var rk = text(rg0, 16, 10 + fhC / 2 + fS * 0.36, String(ci + 1), 'va-t va-small va-rank', 'middle');
        [1, 6, 11, 16].forEach(function (t, k) { frame(rg0, 30 + k * (fwC + tg), 10, fwC, fhC, cd.vid, false).set(t); });
        function line(row, sym, cls) {
          var yy = 10 + fhC + 8 + row * lineB + lineB / 2;
          text(rg0, 12, yy + fS * 0.36, sym, 'va-t va-small va-sym');
          E('rect', { x: rnd(barX), y: rnd(yy - 3.5), width: rnd(barW), height: 7, rx: 3.5, 'class': 'va-bartrack' }, rg0);
          var b = E('rect', { x: rnd(barX), y: rnd(yy - 3.5), width: 0, height: 7, rx: 3.5, 'class': cls }, rg0);
          var v = text(rg0, cardW - 10, yy + fS * 0.36, '', 'va-t va-small va-mono', 'end');
          return { b: b, v: v, x: barX, y: yy, w: barW };
        }
        candLabel(rg0, 12, 10 + fhC + 8 + lineB / 2 + fS * 0.36, cd);
        var r1 = line(1, '$ρ$', 'va-bar-rho'), r2 = line(2, '$s_θ$', 'va-bar-s');
        attr(r1.b, 'width', rnd(barW * cd.rho)); retext(r1.v, cd.rho.toFixed(2));
        rows.push({ g: rg0, box: bx, rk: rk, s: r2, ci: ci });
      });
      o.height = y0 + cardH + 6;
    } else {
      var NTN = [1, 8, 16], yHn = H1 + 20 + fS, fwN = 36, fhN = 27, stripW = NTN.length * fwN + (NTN.length - 1) * tg, rowH = Math.max(fhN + 10, fS * 1.6 + 12);
      var labH = fS * 1.35, cardHn = rowH + labH;
      var sx = 24, rhoC = sx + stripW + 12 + vw0 / 2, barXn = rhoC + vw0 / 2 + 12, barWn = (W - 2 * mx) - 10 - vw0 - 8 - barXn;
      text(g, mx, yHn, 'candidates $C(q)$', 'va-t va-small va-lab');
      hRho = text(g, mx + rhoC, yHn, '$ρ$', 'va-t va-small va-sym', 'middle');
      hS = text(g, mx + barXn, yHn, '$s_θ$', 'va-t va-small va-sym');
      mark = E('path', { d: 'M-3.5,-2 L3.5,-2 L0,2.5 Z', 'class': 'va-sortmark' }, g);
      var markA = [mx + rhoC + measure('$ρ$', 'va-t va-small va-sym') / 2 + 7, yHn - fS * 0.32];
      var markB = [mx + barXn + measure('$s_θ$', 'va-t va-small va-sym') + 7, yHn - fS * 0.32];
      mark._a = markA; mark._b = markB;
      var y0n = yHn + 10;
      for (i = 0; i < CANDS.length; i++) slotPos.push([mx, y0n + i * (cardHn + 6)]);
      CANDS.forEach(function (cd, ci) {
        var rg0 = E('g', { 'class': 'va-cand' }, g);
        var bx = E('rect', { x: 0, y: 0, width: rnd(W - 2 * mx), height: rnd(cardHn), rx: 9, 'class': 'va-card' }, rg0);
        candLabel(rg0, sx, rowH - 2 + labH / 2 + fS * 0.36, cd);
        var rk = text(rg0, 12, rowH / 2 + fS * 0.36, String(ci + 1), 'va-t va-small va-rank', 'middle');
        NTN.forEach(function (t, k) { frame(rg0, sx + k * (fwN + tg), (rowH - fhN) / 2, fwN, fhN, cd.vid, false).set(t); });
        var rv = text(rg0, rhoC, rowH / 2 + fS * 0.36, cd.rho.toFixed(2), 'va-t va-small va-mono va-rhov', 'middle');
        E('rect', { x: rnd(barXn), y: rnd(rowH / 2 - 3.5), width: rnd(barWn), height: 7, rx: 3.5, 'class': 'va-bartrack' }, rg0);
        var b = E('rect', { x: rnd(barXn), y: rnd(rowH / 2 - 3.5), width: 0, height: 7, rx: 3.5, 'class': 'va-bar-s' }, rg0);
        var v = text(rg0, W - 2 * mx - 10, rowH / 2 + fS * 0.36, '', 'va-t va-small va-mono', 'end');
        rows.push({ g: rg0, box: bx, rk: rk, s: { b: b, v: v, x: barXn, y: rowH / 2, w: barWn }, ci: ci, rv: rv });
      });
      o.height = y0n + CANDS.length * (cardHn + 6);
    }
    rows.slice().sort(function (r1, r2) { return RANK[r2.ci] - RANK[r1.ci]; }).forEach(function (r) { g.appendChild(r.g); });
    var flyDot = E('circle', { r: 4.5, 'class': 'va-flydot', opacity: 0 }, g);
    function rowXY(ci, p) {
      var k = span(p, CR, CR1), a = slotPos[ci], b = slotPos[RANK[ci]];
      return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
    }
    function done(ci, p) { return p >= cStart(ci) + (ci === 0 ? 1950 : 860); }
    o.update = function (p) {
      op(g, span(p, 0, 300));
      /* which candidate is being scored, and the stage within it */
      var k = -1, a = 0, slow = false;
      if (p >= C0 && p < C1) { k = 0; a = p - C0; slow = true; }
      else if (p >= C1 && p < C1 + 3 * CS) { k = 1 + Math.floor((p - C1) / CS); a = p - cStart(k); }
      var hold = p >= CR1, act = k >= 0, any = p >= C0;
      var w = slow ? { fetch: [0, 350], enc: [350, 800], c: 600, pri: [800, 1200], e: 1000, head: [1200, 1600], s: 1400, wr: [1600, 1950] }
                   : { fetch: [0, 200], enc: [160, 380], c: 300, pri: [360, 560], e: 470, head: [540, 720], s: 640, wr: [700, 860] };
      var shown = act ? k : hold ? ORDER[0] : (p >= C1 + 3 * CS ? CANDS.length - 1 : 0), cd = CANDS[shown];
      /* fetch the candidate's cached tokens, frame group by frame group */
      cache.forEach(function (c4, t) {
        var vis = act ? a >= w.fetch[0] + (w.fetch[1] - w.fetch[0]) * t / T : any;
        setTok(c4, vis ? cd.vid.Z[t] : null);
      });
      var encOn = act && a >= w.enc[0] && a < w.enc[1];
      on(enc.g, 'is-on', encOn);
      flow(aIn, encOn && a < w.enc[0] + 160);
      pills.forEach(function (n) { on(n, 'is-on', encOn); });
      var cOn = act ? a >= w.c : any;
      setVec(cvec, cOn ? cd.c : null, act ? span(a, w.c, w.c + 150) : 1);
      flow(aC, act && a >= w.c && a < w.pri[0] + 100);
      var priOn = act && a >= w.pri[0] && a < w.pri[1];
      on(st1.g, 'is-on', priOn); on(er.g, 'is-on', priOn && a >= w.pri[0] + 80);
      retext(st1v, (act ? a >= w.pri[0] : any) ? cd.rho.toFixed(2) : '');
      flow(aE1, priOn && a < w.e); flow(aE2, act && a >= w.e && a < w.head[0] + 80);
      var eOn = act ? a >= w.e : any;
      setVec(evec, eOn ? cd.e.map(function (v) { return v / 0.5; }) : null, act ? span(a, w.e, w.e + 150) : 1);
      var headOn = act && a >= w.head[0] && a < w.head[1];
      on(pl.g, 'is-on', headOn); on(hd.g, 'is-on', headOn && a >= w.head[0] + 80);
      flow(aH, headOn);
      var sOn = act ? a >= w.s : any;
      retext(sVal, sOn ? cd.s.toFixed(2) : '');
      op(sVal, act ? span(a, w.s, w.s + 120) : 1);
      /* each score is written into its candidate's row; then the rows move to their new order */
      rows.forEach(function (r) {
        var xy = rowXY(r.ci, p), c0 = CANDS[r.ci];
        attr(r.g, 'transform', 'translate(' + rnd(xy[0]) + ',' + rnd(xy[1]) + ')');
        /* a row that moves down passes behind the others */
        op(r.g, RANK[r.ci] > r.ci ? 1 - 0.45 * Math.sin(Math.PI * span(p, CR, CR1)) : 1);
        var fl = done(r.ci, p) ? 1 : (r.ci === k && a >= w.wr[0]) ? span(a, w.wr[0] + 120, w.wr[1]) : 0;
        attr(r.s.b, 'width', rnd(r.s.w * c0.s * fl));
        retext(r.s.v, fl > 0.5 ? c0.s.toFixed(2) : '');
        on(r.box, 'is-on', r.ci === k || (hold && RANK[r.ci] === 0));
        retext(r.rk, String((p >= CR1 ? RANK[r.ci] : r.ci) + 1));
      });
      var kk = (a - w.wr[0]) / 170;
      if (act && kk > 0 && kk < 1) {
        var tgt = rowXY(k, p), r0 = rows[k], bxy = [tgt[0] + r0.s.x + 4, tgt[1] + r0.s.y];
        var from = [+sVal.getAttribute('x') + (N ? 16 : 0), +sVal.getAttribute('y') - fS * 0.45];
        attr(flyDot, 'cx', rnd(from[0] + (bxy[0] - from[0]) * ease(kk))); attr(flyDot, 'cy', rnd(from[1] + (bxy[1] - from[1]) * ease(kk)));
        op(flyDot, 1);
      } else op(flyDot, 0);
      var sw = span(p, CR, CR + 300);
      if (hdr) { op(hdr, 1 - sw); op(hdr2, sw); }
      if (mark) {
        var m = p >= CR ? mark._b : mark._a;
        attr(mark, 'transform', 'translate(' + rnd(m[0]) + ',' + rnd(m[1]) + ')');
        on(hRho, 'is-sort', p < CR); on(hS, 'is-sort', p >= CR);
      }
    };
    return o;
  }

  /* ------------------------------------------------------------------ */
  /* Layout, player and controls                                         */
  /* ------------------------------------------------------------------ */
  var SCENES = [], state = { step: 0, t: 0, playing: false, done: false, userPaused: false }, raf = 0, last = 0;
  function contentWidth() {
    var cs = getComputedStyle(stage);
    return stage.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
  }
  function measureLayout() {
    var cw = contentWidth(), narrow = cw > 0 && cw < 560, W = narrow ? 400 : 960;
    var rendered = cw > 0 ? (narrow ? Math.min(cw, 400) : cw) : W;
    /* text near 12.5px on screen; the stacked layout never goes below 1.12, so that its subscripts stay at 10px */
    var F = Math.max(narrow ? 1.12 : 1, Math.min(1.75, 0.96 / (rendered / W)));
    return { narrow: narrow, W: W, F: Math.round(F * 50) / 50 };
  }
  function defs() {
    var d = E('defs', null, svg);
    /* the frames' sky and grass, as on the cover */
    [['vaSkyG', [[0, '#A9C2D9'], [0.66, '#E9EFF2']]], ['vaGrassG', [[0, '#A9C78A'], [1, '#5B8D43']]]].forEach(function (gd) {
      var lg = E('linearGradient', { id: gd[0], x1: 0, y1: 0, x2: 0, y2: 1 }, d);
      gd[1].forEach(function (s) { E('stop', { offset: s[0], 'stop-color': s[1] }, lg); });
    });
    [['vaArrow', 'va-head-off'], ['vaArrowOn', 'va-head-on']].forEach(function (m) {
      var mk = E('marker', { id: m[0], viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6.5, markerHeight: 6.5, orient: 'auto-start-reverse' }, d);
      E('path', { d: 'M0,1 L9,5 L0,9 z', 'class': m[1] }, mk);
    });
  }
  function build() {
    L = measureLayout();
    root.classList.toggle('is-narrow', L.narrow);
    svg.style.setProperty('--va-f', L.F.toFixed(2));
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    meas = null; HATS = [];
    defs();
    SCENES = [sceneIndex, sceneTrain, sceneRerank].map(function (make) {
      var g = E('g', { 'class': 'va-scene' }, svg), inner = E('g', null, g), s = make(inner); s.g = g; s.inner = inner; return s;
    });
    var Hm = 0; SCENES.forEach(function (s) { Hm = Math.max(Hm, s.height); });
    Hm = Math.ceil(Hm + 4);
    SCENES.forEach(function (s) { attr(s.g, 'transform', 'translate(0,' + rnd((Hm - s.height) / 2) + ')'); });
    svg.setAttribute('viewBox', '0 0 ' + L.W + ' ' + Hm);
    placeHats();
    if (meas && meas.parentNode) meas.parentNode.removeChild(meas);
    showScene(state.step);
    fitCaption();
  }
  function showScene(n) {
    SCENES.forEach(function (s, i) { s.g.style.display = i === n ? '' : 'none'; });
    var btns = tabs.querySelectorAll('button[data-step]');
    Array.prototype.forEach.call(btns, function (b) {
      var sel = +b.getAttribute('data-step') === n;
      b.setAttribute('aria-selected', String(sel)); b.setAttribute('tabindex', sel ? '0' : '-1');
    });
    stage.setAttribute('aria-labelledby', 'vedjeAnimTab' + n);
    capTitle.textContent = SCENES[n].title;
    capText.textContent = SCENES[n].text;
    svg.setAttribute('aria-label', 'Figure 2 of the paper, panel ' + TABS[n][0] + '. ' + SCENES[n].title + ' ' + SCENES[n].text);
  }
  function fitCaption() {
    var probe = cap.cloneNode(true), h = 0;
    probe.removeAttribute('aria-live'); probe.setAttribute('aria-hidden', 'true');
    probe.style.cssText = 'position:absolute;left:0;top:0;visibility:hidden;min-height:0;width:' + cap.clientWidth + 'px';
    root.appendChild(probe);
    SCENES.forEach(function (s) { probe.firstChild.textContent = s.title; probe.lastChild.textContent = s.text; h = Math.max(h, probe.offsetHeight); });
    root.removeChild(probe);
    if (h) cap.style.minHeight = h + 'px';
  }
  function render() {
    var s = SCENES[state.step], tt = STATIC && state.t === 0 ? s.dur : Math.min(state.t, s.dur);
    s.update(tt);
    bar.style.width = (STATIC ? 100 : Math.min(100, state.t / s.dur * 100)).toFixed(2) + '%';
  }
  function go(n) { state.step = n; state.t = 0; showScene(n); render(); }
  function frameTick(now) {
    raf = 0; if (!state.playing) return;
    var dt = last ? Math.min(64, now - last) : 16; last = now;
    state.t += dt;
    if (state.t >= SCENES[state.step].dur) {
      if (state.step < SCENES.length - 1) go(state.step + 1);
      else { state.t = SCENES[state.step].dur; state.playing = false; state.done = true; render(); button(); return; }
    }
    render(); raf = requestAnimationFrame(frameTick);
  }
  function play() {
    if (STATIC) return;
    if (state.done) { state.done = false; go(0); }
    state.playing = true; last = 0;
    if (!raf) raf = requestAnimationFrame(frameTick);
    button();
  }
  function pause() { state.playing = false; button(); }
  function button() {
    var mode = state.playing ? 'pause' : state.done ? 'replay' : 'play';
    playBtn.setAttribute('data-mode', mode);
    playBtn.setAttribute('aria-label', mode === 'pause' ? 'Pause the animation' : mode === 'replay' ? 'Replay the animation' : 'Play the animation');
  }
  tabs.addEventListener('click', function (e) {
    var b = e.target.closest('button[data-step]'); if (!b) return;
    state.done = false; go(+b.getAttribute('data-step'));
    if (!STATIC && !state.userPaused) play();
  });
  tabs.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'Home' && e.key !== 'End') return;
    var n = e.key === 'Home' ? 0 : e.key === 'End' ? SCENES.length - 1 : (state.step + (e.key === 'ArrowRight' ? 1 : SCENES.length - 1)) % SCENES.length;
    e.preventDefault(); state.done = false; go(n);
    var b = tabs.querySelector('button[data-step="' + n + '"]'); if (b) b.focus();
    if (!STATIC && !state.userPaused) play();
  });
  playBtn.addEventListener('click', function () {
    if (state.playing) { state.userPaused = true; pause(); } else { state.userPaused = false; play(); }
  });

  build();
  var qs = parseInt(query.get('animScene'), 10), qt = parseInt(query.get('animT'), 10);
  if (qs >= 0 && qs < SCENES.length) go(qs);
  if (STATIC) { root.classList.add('is-static'); render(); button(); }
  else if (!isNaN(qt)) { state.t = Math.max(0, qt); state.userPaused = true; render(); button(); }
  else {
    render(); button();
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting && en.intersectionRatio >= 0.35) { if (!state.userPaused && !state.done) play(); }
          else if (state.playing) pause();
        });
      }, { threshold: [0, 0.35, 0.6] }).observe(root);
    } else play();
  }
  /* rebuild when the stage changes between the two compositions or the text scale changes */
  var lastKey = L.narrow + ':' + L.F, rt = 0;
  function onResize() {
    clearTimeout(rt);
    rt = setTimeout(function () {
      var m = measureLayout(), key = m.narrow + ':' + m.F;
      if (key !== lastKey) { lastKey = key; build(); render(); } else fitCaption();
    }, 120);
  }
  if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(stage);
  else window.addEventListener('resize', onResize);
  /* the web fonts change the text widths: lay out again once they are in */
  var ft = 0;
  function relayout() { clearTimeout(ft); ft = setTimeout(function () { build(); render(); }, 60); }
  if (document.fonts) {
    if (document.fonts.ready) document.fonts.ready.then(relayout);
    if (document.fonts.addEventListener) document.fonts.addEventListener('loadingdone', relayout);
  }
})();
