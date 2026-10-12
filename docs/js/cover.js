/* VEDJE project page: the interactive cover, "Index once, rerank every query".
   Builds everything inside #vedjeCover. Vanilla JS, no dependencies.

   Taken from the paper: T = 16 sampled frames, P = 256 patch positions per frame (16 x 16), M = 4 tokens per frame,
   d = 384 and two bytes per element (Section 3.2, Appendix A.2), so the cache payload is 2 x T x M x d bytes: 3,072
   bytes per frame and 48 KiB per video. The frame-and-patch features of the same video take 2 x 16 x 257 x 768 bytes
   = 6.02 MiB (Table 9). The order of the steps follows Figure 2 and Algorithm 1: the frozen encoder runs once per
   video, the compressor turns each frame-indexed slice into M tokens stored at its temporal position, the first stage
   returns the candidates with their scores, the joint encoder reads the query tokens next to each cached sequence,
   the first-stage score joins its output as a prior, and the candidates are sorted by the final score, with no visual
   encoder on the query path.
   The second mode draws a typical joint reranker on the same video, query, candidates and steps, kept generic: offline
   the encoder writes one first-stage embedding and no cache; online a visual encoder runs on the frames of each
   candidate and a large joint model scores them. It shows no storage, latency or cost figure of its own. Its caption
   cites Appendix B (Table 22): the LamRA reproduction, with a 7.6B decoder, reaches 59.7 MSR-VTT text-to-video R@1
   and VEDJE 59.8 with about 157M online parameters, so both modes end in the same order.
   Illustrative: the drawn frames, the query and all scores. Three candidates share the field of the indexed video and
   differ only in what happens, which is why they look alike to the first stage.

   URL flags: ?static=1 freezes step 20 (the third candidate is read, the cache is complete) without autoplay or
   transitions; ?coverMode=typical opens the typical reranker; ?coverStep=N opens step N (1 to 23), paused. */
(function () {
  'use strict';

  var root = document.getElementById('vedjeCover');
  if (!root) return;

  var SVGNS = 'http://www.w3.org/2000/svg';
  var search = location.search;
  var STATIC = /[?&]static\b/.test(search);
  var reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var stepFlag = search.match(/[?&]coverStep=(\d+)/);

  /* ---------------- facts of the schematic ---------------- */
  var T = 16, P = 256, M = 4, D = 384, BYTES = 2, NC = 5;
  var S_QUERY = T + 1;          // 17: the query arrives and the first stage returns its candidates
  var S_READ = T + 2;           // 18 to 22: the reranker reads one candidate per step
  var S_SORT = T + 2 + NC;      // 23: the candidates are sorted by the final score
  var N = S_SORT;
  var STATIC_STEP = S_READ + 2; // 20: the indexed video (stage-1 rank 3) is read
  var EMB = 8;                  // cells drawn for the first-stage embedding of the typical reranker
  var FULL_BYTES = BYTES * T * 257 * 768;   // frame-and-patch features of one video (Table 9)

  function cacheBytes(frames) { return BYTES * frames * M * D; }
  function kib(bytes) { return (Math.round(bytes / 1024 * 100) / 100) + '\u00a0KiB'; }
  function mib(bytes) { return (bytes / 1048576).toFixed(2) + '\u00a0MiB'; }

  var QUERY = 'a person walks across a field';
  var QUERY_TOKENS = 6;
  /* the five candidates in stage-1 order; prior is the first-stage score, score the final VEDJE score and typ the score
     of the typical reranker (all illustrative; typ keeps the order of score, as the two reach matching R@1) */
  var CANDS = [
    { kind: 'stand', label: 'Person stands in the field', prior: 0.74, score: 0.38, typ: 0.41 },
    { kind: 'dog', label: 'Dog runs across the field', prior: 0.71, score: 0.21, typ: 0.18 },
    { kind: 'walk', label: 'Person walks across the field', prior: 0.69, score: 0.91, typ: 0.89, match: true },
    { kind: 'beach', label: 'Person walks on a beach', prior: 0.63, score: 0.64, typ: 0.60 },
    { kind: 'bike', label: 'Person rides a bike on a road', prior: 0.58, score: 0.12, typ: 0.15 }
  ];
  var order = CANDS.map(function (c, i) { return i; }).sort(function (a, b) { return CANDS[b].score - CANDS[a].score; });
  CANDS.forEach(function (c, i) { c.s1 = i + 1; c.rank = order.indexOf(i) + 1; });
  var MATCH = 2;

  var ILLUSTRATIVE = 'The frames, the query, the scores and the timing are illustrative.';
  var FINE = '<small class="vc-fine">' + ILLUSTRATIVE + ' Cache sizes follow Table 9 of the paper, and parameter counts Appendix B.</small>';
  /* per query, for both pipelines: the visual encoder runs for the candidates shown, and the online parameters of
     VEDJE and of the 7.6B decoder of the LamRA reproduction (Appendix B, Table 22) */
  var COSTS = [['Visual encoder runs', [0, '0'], [NC, String(NC)]], ['Online parameters', [157e6, '157M'], [7.6e9, '7.6B']]];
  /* the typical caption is kept no longer than the VEDJE one, so the shared caption cell keeps the card's height */
  var CAPTIONS = {
    vedje: '<strong>Index once, rerank every query.</strong> Offline, each frame is compressed into four tokens, kept in ' +
      'temporal order. Online, the first stage ranks look-alike scenes first. VEDJE reads each candidate\'s cache with ' +
      'the query and moves the matching video to the top. ' + FINE,
    typical: '<strong>Encode every candidate at query time.</strong> A typical joint reranker encodes each candidate\'s ' +
      'frames for every query. The LamRA reproduction runs a 7.6B decoder and reaches 59.7 MSR-VTT text-to-video R@1. ' +
      'VEDJE reaches 59.8 with about 157M online parameters (Appendix&nbsp;B). <small class="vc-fine">' + ILLUSTRATIVE + '</small>'
  };

  /* ---------------- small helpers ---------------- */
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function sv(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function hash(a, b, c) {
    var h = Math.imul(a, 374761393) ^ Math.imul(b, 668265263) ^ Math.imul(c, 2246822519);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }
  /* a fixed shade per cached token, so the cache of the indexed video looks the same in both panels */
  function tokShade(video, t, m) { return (0.6 + 0.4 * hash(video + 3, t + 11, m + 5)).toFixed(2); }
  /* the first-stage embedding of the typical reranker shifts a little with each frame the encoder reads */
  function embShade(j, t) { return (0.4 + 0.42 * hash(17, 0, j) + 0.18 * hash(17, t, j)).toFixed(2); }

  /* ---------------- the drawn scenes (32 x 24 frame units) ----------------
     Each scene is described once and painted twice: as SVG for the frames and thumbnails, and on a small canvas whose
     16 x 16 cell averages make the patch grid of the indexed video. The encoder reads each frame stretched to a square,
     as scripts/extract_features.py does, so the canvas is square. Three candidates share one field and one tree, so
     they look alike to the first stage and differ only in what happens. */
  var PAL = {
    skyTop: '#A9C2D9', skyLow: '#E9EFF2', cloud: '#FFFFFF', far: '#8BA596', far2: '#A2B7A8',
    grassFar: '#A9C78A', grassNear: '#5B8D43', canopy: '#4A7435', canopyLit: '#6B974A', canopyDark: '#385B28', trunk: '#584232',
    skin: '#DFB593', hair: '#3A2A1F', jacket: '#3D6084', red: '#B8463A', yellow: '#D9A23A', helmet: '#2D5D8C',
    trousers: '#2F333B', trousersFar: '#22252B', shoe: '#1D1E21',
    sea: '#5C8FB1', seaLow: '#8AB3C8', foam: '#F2F5F3', wet: '#C3AA80', sand: '#E3D2AA', sandNear: '#D2BB8B',
    verge: '#8CB370', road: '#7B7D80', roadNear: '#626467', dash: '#ECEBE3', frame: '#BF3F2C', wheel: '#25272A',
    dog: '#8C5B33', dogDark: '#5E3C1E'
  };
  function shade(hex, f) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgb(' + [n >> 16 & 255, n >> 8 & 255, n & 255].map(function (v) { return Math.round(v * f); }).join(',') + ')';
  }
  /* painters: rect, oval, poly, line and ring, with a colour or a vertical gradient [y0, y1, top colour, bottom colour] */
  var gid = 0;
  function svgPainter(s) {
    var defs = sv('defs', {}); s.appendChild(defs);
    function paint(f) {
      if (typeof f === 'string') return f;
      var id = 'vcg' + (++gid), g = sv('linearGradient', { id: id, gradientUnits: 'userSpaceOnUse', x1: 0, y1: f[0], x2: 0, y2: f[1] });
      g.appendChild(sv('stop', { offset: 0, 'stop-color': f[2] })); g.appendChild(sv('stop', { offset: 1, 'stop-color': f[3] }));
      defs.appendChild(g);
      return 'url(#' + id + ')';
    }
    function add(tag, a, f, o) { a.fill = paint(f); if (o != null) a['fill-opacity'] = o; s.appendChild(sv(tag, a)); }
    function n(v) { return +v.toFixed(2); }
    return {
      rect: function (x, y, w, h, f, o) { add('rect', { x: n(x), y: n(y), width: n(w), height: n(h) }, f, o); },
      oval: function (cx, cy, rx, ry, f, o) { add('ellipse', { cx: n(cx), cy: n(cy), rx: n(rx), ry: n(ry) }, f, o); },
      poly: function (pts, f, o) { add('polygon', { points: pts.map(function (q) { return n(q[0]) + ',' + n(q[1]); }).join(' ') }, f, o); },
      line: function (x1, y1, x2, y2, c, w) { s.appendChild(sv('line', { x1: n(x1), y1: n(y1), x2: n(x2), y2: n(y2), stroke: c, 'stroke-width': w, 'stroke-linecap': 'round' })); },
      ring: function (cx, cy, r, c, w) { s.appendChild(sv('circle', { cx: n(cx), cy: n(cy), r: r, fill: 'none', stroke: c, 'stroke-width': w })); }
    };
  }
  function canvasPainter(ctx, sx, sy) {
    function paint(f) {
      if (typeof f === 'string') return f;
      var g = ctx.createLinearGradient(0, f[0] * sy, 0, f[1] * sy); g.addColorStop(0, f[2]); g.addColorStop(1, f[3]); return g;
    }
    function fill(f, o) { ctx.globalAlpha = o == null ? 1 : o; ctx.fillStyle = paint(f); ctx.fill(); }
    function stroke(c, w) { ctx.globalAlpha = 1; ctx.strokeStyle = c; ctx.lineWidth = w * (sx + sy) / 2; ctx.lineCap = 'round'; ctx.stroke(); }
    return {
      rect: function (x, y, w, h, f, o) { ctx.beginPath(); ctx.rect(x * sx, y * sy, w * sx, h * sy); fill(f, o); },
      oval: function (cx, cy, rx, ry, f, o) { ctx.beginPath(); ctx.ellipse(cx * sx, cy * sy, rx * sx, ry * sy, 0, 0, 2 * Math.PI); fill(f, o); },
      poly: function (pts, f, o) { ctx.beginPath(); pts.forEach(function (q, i) { ctx[i ? 'lineTo' : 'moveTo'](q[0] * sx, q[1] * sy); }); ctx.closePath(); fill(f, o); },
      line: function (x1, y1, x2, y2, c, w) { ctx.beginPath(); ctx.moveTo(x1 * sx, y1 * sy); ctx.lineTo(x2 * sx, y2 * sy); stroke(c, w); },
      ring: function (cx, cy, r, c, w) { ctx.beginPath(); ctx.ellipse(cx * sx, cy * sy, r * sx, r * sy, 0, 0, 2 * Math.PI); stroke(c, w); }
    };
  }

  /* backgrounds */
  function sky(p, horizon) {
    p.rect(0, 0, 32, horizon + 0.6, [0, horizon, PAL.skyTop, PAL.skyLow]);
    p.oval(8.5, 4.4, 5.4, 1.2, PAL.cloud, 0.6); p.oval(11.9, 3.7, 3.2, 1.0, PAL.cloud, 0.55); p.oval(20.6, 6.1, 3.6, 0.8, PAL.cloud, 0.45);
  }
  function treeLine(p, y) {   // distant trees along the horizon, pale with distance
    p.rect(0, y - 0.9, 32, 1.4, PAL.far2);
    [[1.5, 1.7, 1.2], [5.4, 2.3, 1.5], [9.6, 1.9, 1.1], [14.2, 2.5, 1.4], [18.6, 1.9, 1.2], [22.4, 1.6, 0.9], [30.8, 2.1, 1.3]]
      .forEach(function (b) { p.oval(b[0], y - 0.7, b[1], b[2], PAL.far); });
  }
  function bigTree(p, x, base) {
    p.oval(x + 0.7, base + 0.1, 3.6, 0.42, '#000000', 0.12);
    p.poly([[x - 0.55, base], [x - 0.32, base - 5.3], [x + 0.32, base - 5.3], [x + 0.62, base]], PAL.trunk);
    p.oval(x, base - 7.6, 4.3, 3.6, PAL.canopyDark);
    p.oval(x - 1.6, base - 8.2, 2.7, 2.4, PAL.canopy); p.oval(x + 1.5, base - 8.6, 2.6, 2.3, PAL.canopy);
    p.oval(x - 0.4, base - 9.9, 2.5, 2.0, PAL.canopyLit); p.oval(x - 2.1, base - 9.0, 1.3, 1.0, PAL.canopyLit);
  }
  function fieldScene(p) {   // shared by the standing person, the dog and the walker
    sky(p, 14.6); treeLine(p, 14.6);
    p.rect(0, 14.3, 32, 9.7, [14.3, 24, PAL.grassFar, PAL.grassNear]);
    bigTree(p, 26.3, 15.6);
  }
  function beachScene(p) {
    sky(p, 12.4);
    p.rect(0, 12.2, 32, 3.1, [12.2, 15.3, PAL.sea, PAL.seaLow]);
    p.rect(0, 15.1, 32, 0.45, PAL.foam, 0.9);
    p.rect(0, 15.5, 32, 2.0, PAL.wet);
    p.rect(0, 17.4, 32, 6.6, [17.4, 24, PAL.sand, PAL.sandNear]);
  }
  function roadScene(p) {
    sky(p, 14.0); treeLine(p, 14.0);
    p.rect(0, 13.7, 32, 4.9, [13.7, 18.6, PAL.grassFar, PAL.verge]);
    p.rect(0, 18.4, 32, 4.7, [18.4, 23.1, PAL.road, PAL.roadNear]);
    for (var d = 0.6; d < 32; d += 5) p.rect(d, 19.6, 2.6, 0.36, PAL.dash);
    p.rect(0, 23.0, 32, 1.0, PAL.verge);
  }

  /* figures, about half the frame height so that they read at thumbnail size */
  function walker(p, x, ph, coat) {   // seen from the side, walking to the right; ph is the phase of the stride
    var sw = Math.sin(ph), bob = 0.18 * Math.abs(Math.cos(ph)), hip = 15.8 - bob, sh = 12.3 - bob, ground = 21.2;
    p.oval(x + 0.2, ground + 0.15, 2.4, 0.4, '#000000', 0.16);
    function leg(a, col) {   // thigh angle a from the vertical, forward positive; the trailing leg lifts its heel
      var b = a < 0 ? a - 0.22 : a - 0.08;
      var kx = x + 2.7 * Math.sin(a), ky = hip + 2.7 * Math.cos(a);
      var fx = kx + 2.75 * Math.sin(b), fy = Math.min(ground, ky + 2.75 * Math.cos(b));
      p.line(x, hip, kx, ky, col, 1.2); p.line(kx, ky, fx, fy, col, 1.05); p.oval(fx + 0.35, fy + 0.05, 0.62, 0.3, PAL.shoe);
    }
    function arm(a, col) {
      var ex = x + 1.7 * Math.sin(a), ey = sh + 1.7 * Math.cos(a), hx = ex + 1.6 * Math.sin(a + 0.35), hy = ey + 1.6 * Math.cos(a + 0.35);
      p.line(x, sh, ex, ey, col, 0.9); p.line(ex, ey, hx, hy, col, 0.82); p.oval(hx, hy, 0.36, 0.36, PAL.skin);
    }
    leg(-0.36 * sw, PAL.trousersFar); arm(0.42 * sw, shade(coat, 0.78));
    p.poly([[x - 1.05, 11.9 - bob], [x + 1.05, 11.9 - bob], [x + 0.95, 16.1 - bob], [x - 0.95, 16.1 - bob]], coat);
    leg(0.36 * sw, PAL.trousers); arm(-0.42 * sw, coat);
    p.oval(x + 0.15, 10.35 - bob, 1.0, 1.15, PAL.skin); p.oval(x - 0.1, 9.7 - bob, 1.0, 0.62, PAL.hair);
  }
  function stander(p, x) {   // seen from the front, arms at the sides
    var ground = 21.2;
    p.oval(x, ground + 0.15, 2.1, 0.4, '#000000', 0.16);
    p.line(x - 0.5, 15.9, x - 0.55, 20.8, PAL.trousers, 1.15); p.line(x + 0.5, 15.9, x + 0.55, 20.8, PAL.trousers, 1.15);
    p.oval(x - 0.62, 20.95, 0.55, 0.3, PAL.shoe); p.oval(x + 0.62, 20.95, 0.55, 0.3, PAL.shoe);
    p.poly([[x - 1.35, 11.9], [x + 1.35, 11.9], [x + 1.15, 16.2], [x - 1.15, 16.2]], PAL.jacket);
    p.line(x - 1.35, 12.3, x - 1.6, 15.6, PAL.jacket, 0.85); p.line(x + 1.35, 12.3, x + 1.6, 15.6, PAL.jacket, 0.85);
    p.oval(x - 1.62, 15.95, 0.36, 0.36, PAL.skin); p.oval(x + 1.62, 15.95, 0.36, 0.36, PAL.skin);
    p.oval(x, 10.35, 1.05, 1.18, PAL.skin); p.oval(x, 9.68, 1.05, 0.6, PAL.hair);
  }
  function dog(p, x) {   // running to the right, legs stretched
    var g = 21.2, y = 18.4;
    p.oval(x + 0.3, g + 0.15, 3.4, 0.38, '#000000', 0.14);
    p.line(x - 2.2, y + 0.6, x - 3.9, g - 0.6, PAL.dogDark, 0.62); p.line(x - 1.6, y + 0.7, x - 2.6, g - 0.1, PAL.dog, 0.62);
    p.line(x + 1.9, y + 0.6, x + 3.6, g - 0.5, PAL.dogDark, 0.62); p.line(x + 1.5, y + 0.7, x + 2.5, g - 0.05, PAL.dog, 0.62);
    p.line(x - 2.6, y - 0.4, x - 4.0, y - 1.6, PAL.dog, 0.55);
    p.oval(x, y, 2.9, 1.15, PAL.dog); p.oval(x + 2.2, y - 0.2, 1.1, 1.0, PAL.dog);
    p.poly([[x + 2.5, y - 0.9], [x + 3.4, y - 2.0], [x + 3.9, y - 1.2], [x + 3.1, y - 0.2]], PAL.dog);
    p.oval(x + 3.9, y - 1.9, 0.95, 0.8, PAL.dog); p.oval(x + 4.75, y - 1.65, 0.6, 0.38, PAL.dogDark);
    p.poly([[x + 3.5, y - 2.5], [x + 3.9, y - 3.2], [x + 4.1, y - 2.4]], PAL.dogDark);
  }
  function cyclist(p, x) {   // riding to the right along the road
    var hub = 20.6, r = 2.15, rear = x - 2.6, front = x + 2.7, bb = x - 0.1, seat = [x - 0.9, 17.6], bar = [x + 1.9, 17.7];
    p.oval(x, 22.8, 3.6, 0.35, '#000000', 0.15);
    p.ring(rear, hub, r, PAL.wheel, 0.42); p.ring(front, hub, r, PAL.wheel, 0.42);
    p.line(rear, hub, bb, hub + 0.1, PAL.frame, 0.45); p.line(rear, hub, seat[0], seat[1], PAL.frame, 0.45);
    p.line(bb, hub + 0.1, seat[0], seat[1], PAL.frame, 0.5); p.line(seat[0], seat[1], bar[0], bar[1], PAL.frame, 0.5);
    p.line(bb, hub + 0.1, bar[0], bar[1], PAL.frame, 0.5); p.line(bar[0], bar[1], front, hub, PAL.frame, 0.45);
    p.line(bar[0], bar[1], x + 2.2, 16.6, PAL.wheel, 0.4);
    p.line(x - 0.9, 17.0, x + 0.5, 18.9, PAL.trousers, 1.1); p.line(x + 0.5, 18.9, bb + 0.3, hub + 0.7, PAL.trousers, 1.0);
    p.poly([[x - 1.5, 16.7], [x - 0.4, 17.1], [x + 0.9, 14.3], [x - 0.2, 13.8]], PAL.yellow);
    p.line(x + 0.4, 14.3, x + 2.15, 16.5, PAL.yellow, 0.8);
    p.oval(x + 0.9, 12.95, 0.95, 1.05, PAL.skin); p.oval(x + 0.75, 12.4, 1.1, 0.62, PAL.helmet);
  }

  function walkX(t) { return 4.6 + (t - 1) * 1.13; }   // the walker crosses the field from left to right
  function drawScene(p, kind, t) {
    if (kind === 'walk') { fieldScene(p); walker(p, walkX(t), t * 1.9, PAL.jacket); }
    else if (kind === 'stand') { fieldScene(p); stander(p, 15.5); }
    else if (kind === 'dog') { fieldScene(p); dog(p, 13.5); }
    else if (kind === 'beach') { beachScene(p); walker(p, 13.5, 2.2, PAL.red); }
    else if (kind === 'bike') { roadScene(p); cyclist(p, 15.5); }
  }
  function scene(kind, t) {
    var s = sv('svg', { viewBox: '0 0 32 24', preserveAspectRatio: 'xMidYMid slice', 'aria-hidden': 'true', focusable: 'false' });
    drawScene(svgPainter(s), kind, t);
    return s;
  }

  /* the 16 x 16 patch grid of a frame of the indexed video: the frame stretched to a square, averaged per cell */
  var patchCanvas = document.createElement('canvas'); patchCanvas.width = patchCanvas.height = 64;
  var pctx = patchCanvas.getContext ? patchCanvas.getContext('2d', { willReadFrequently: true }) : null;
  function patchColors(t) {
    var out = [], data = null, i, j, x, y, k, r, g, b;
    if (pctx) {
      pctx.clearRect(0, 0, 64, 64);
      drawScene(canvasPainter(pctx, 2, 64 / 24), 'walk', t);
      try { data = pctx.getImageData(0, 0, 64, 64).data; } catch (e) { data = null; }
    }
    for (j = 0; j < 16; j++) for (i = 0; i < 16; i++) {
      if (!data) { out.push(j < 9 ? PAL.skyLow : PAL.grassFar); continue; }
      r = g = b = 0;
      for (y = 0; y < 4; y++) for (x = 0; x < 4; x++) { k = ((j * 4 + y) * 64 + i * 4 + x) * 4; r += data[k]; g += data[k + 1]; b += data[k + 2]; }
      out.push('rgb(' + Math.round(r / 16) + ',' + Math.round(g / 16) + ',' + Math.round(b / 16) + ')');
    }
    return out;
  }

  /* ---------------- build ---------------- */
  root.setAttribute('role', 'group');
  root.innerHTML = '';
  if (STATIC) root.classList.add('vc-static');

  /* top: pipeline toggle and step counter */
  var top = el('div', 'sim-top');
  var seg = el('div', 'seg'); seg.setAttribute('role', 'tablist'); seg.setAttribute('aria-label', 'Pipeline');
  [['vedje', 'VEDJE', 'VEDJE'], ['typical', 'Typical reranker', 'Typical']].forEach(function (d) {
    var b = el('button'); b.type = 'button'; b.setAttribute('role', 'tab'); b.setAttribute('data-mode', d[0]);
    b.innerHTML = '<span class="lbl-long">' + d[1] + '</span><span class="lbl-short">' + d[2] + '</span>';
    seg.appendChild(b);
  });
  var stepBox = el('div', 'sim-step'); stepBox.innerHTML = 'Step <strong>1</strong> / ' + N;
  var stepNum = stepBox.querySelector('strong');
  top.appendChild(seg); top.appendChild(stepBox); root.appendChild(top);

  /* panel 1: index, offline */
  var pIndex = el('div', 'sim-canvas vc-panel vc-index');
  var hIndex = el('div', 'vc-head'); hIndex.appendChild(el('span', 'eyebrow', 'Index · offline'));
  var stIndex = el('span', 'vc-status'); hIndex.appendChild(stIndex); pIndex.appendChild(hIndex);
  var vLabel = el('p', 'vc-vlabel'); vLabel.innerHTML = 'Indexed video: <b>' + CANDS[MATCH].label.toLowerCase() + '</b>';
  pIndex.appendChild(vLabel);

  var framesRow = el('div', 'vc-frames'); framesRow.setAttribute('role', 'img');
  framesRow.setAttribute('aria-label', 'The indexed video, drawn: a person walks across the field, 16 sampled frames');
  var frames = [];
  for (var t = 1; t <= T; t++) { var fr = el('span', 'vc-frame'); fr.appendChild(scene('walk', t)); framesRow.appendChild(fr); frames.push(fr); }
  pIndex.appendChild(framesRow);

  var pipe = el('div', 'vc-pipe');
  var enc = el('div', 'vc-box vc-enc'); enc.innerHTML = '<span>Frozen visual encoder</span><small>Runs once per video</small>';
  var mosaic = el('div', 'vc-mosaic'); var mosaicSvg = sv('svg', { viewBox: '0 0 16 16', 'aria-hidden': 'true', focusable: 'false' });
  var cells = [];
  for (var c = 0; c < P; c++) {
    var r = sv('rect', { x: (c % 16) + 0.06, y: Math.floor(c / 16) + 0.06, width: 0.88, height: 0.88 });
    mosaicSvg.appendChild(r); cells.push(r);
  }
  mosaic.appendChild(mosaicSvg);
  var COLORS = []; for (t = 1; t <= T; t++) COLORS.push(patchColors(t));
  var comp = el('div', 'vc-box vc-comp'); comp.innerHTML = '<span>Compressor</span>';
  var out = el('div', 'vc-out');
  var outTok = el('span', 'vc-tok-group vc-out-tok'); var outToks = [];
  for (var m = 0; m < M; m++) { var oi = el('i'); outTok.appendChild(oi); outToks.push(oi); }
  var outLbl = el('span', 'vc-out-lbl');
  out.appendChild(outTok); out.appendChild(outLbl);
  function arrow(label) {
    var a = el('span', 'vc-arrow' + (label ? ' has-lbl' : '')); a.setAttribute('aria-hidden', 'true');
    if (label) a.appendChild(el('em'));
    return a;
  }
  var arrPatch = arrow(true), arrTok = arrow(true);
  arrTok.classList.add('vc-arrow-tok');
  /* the typical reranker: the encoder writes one first-stage embedding per video in place of the compressor's tokens */
  var emb = el('div', 'vc-emb'), embVec = el('span', 'vc-emb-vec'), embCells = [];
  for (var j = 0; j < EMB; j++) { var ec = el('i'); embVec.appendChild(ec); embCells.push(ec); }
  embVec.setAttribute('aria-hidden', 'true');
  emb.appendChild(embVec); emb.appendChild(el('span', 'vc-emb-lbl', 'First-stage embedding'));
  pipe.appendChild(enc); pipe.appendChild(arrow(false)); pipe.appendChild(mosaic); pipe.appendChild(arrPatch);
  pipe.appendChild(comp); pipe.appendChild(arrTok); pipe.appendChild(out); pipe.appendChild(emb);
  arrPatch.firstChild.textContent = P + ' patches';
  arrTok.firstChild.textContent = M + ' tokens';
  pIndex.appendChild(pipe);

  var cacheRow = el('div', 'vc-cache'); cacheRow.setAttribute('aria-hidden', 'true');
  var zRow = el('div', 'vc-z'); zRow.setAttribute('aria-hidden', 'true');
  var slots = [], zl = [];
  for (t = 1; t <= T; t++) {
    var sl = el('span', 'vc-tok-group vc-slot');
    for (m = 0; m < M; m++) { var ti = el('i'); ti.style.opacity = tokShade(MATCH, t, m); sl.appendChild(ti); }
    cacheRow.appendChild(sl); slots.push(sl);
    var z = el('span', null, 'Z' + t); zRow.appendChild(z); zl.push(z);
  }
  /* the labels Z1 to Z16 and the typical reranker's empty-cache line share one grid cell, so the panel keeps its height */
  var zBox = el('div', 'vc-zbox');
  var noCache = el('p', 'vc-nocache');
  noCache.innerHTML = '<span class="lbl-long">No cache: the reranker encodes the frames again at query time</span>' +
    '<span class="lbl-short">No cache: frames are encoded again online</span>';
  zBox.appendChild(zRow); zBox.appendChild(noCache);
  pIndex.appendChild(cacheRow); pIndex.appendChild(zBox);

  var meter = el('div', 'vc-meter');
  meter.appendChild(el('span', 'eyebrow', 'Stored per video'));
  var track = el('span', 'vc-meter-track'); var fill = el('span', 'vc-meter-fill'); track.appendChild(fill);
  var meterVal = el('span', 'vc-meter-val');
  meter.appendChild(track); meter.appendChild(meterVal); pIndex.appendChild(meter);
  var ref = el('p', 'vc-ref'); ref.innerHTML = 'Frame-and-patch features of the same video: <b>' + mib(FULL_BYTES) + '</b>';
  pIndex.appendChild(ref);
  root.appendChild(pIndex);

  /* panel 2: rerank, online */
  var pOnline = el('div', 'sim-canvas vc-panel vc-online');
  var hOnline = el('div', 'vc-head'); hOnline.appendChild(el('span', 'eyebrow', 'Rerank · online'));
  var stOnline = el('span', 'vc-status'); hOnline.appendChild(stOnline); pOnline.appendChild(hOnline);

  var body = el('div', 'vc-onbody');
  var side = el('div', 'vc-side');
  var qBox = el('div', 'vc-query');
  qBox.appendChild(el('span', 'eyebrow vc-qtag', 'Query'));
  /* an invisible copy of the full query sizes the box, so the card keeps its height while the query is typed */
  var qText = el('span', 'vc-qtext'), qSizer = el('span', 'vc-qsizer', QUERY), qTyped = el('span', 'vc-qtyped');
  qSizer.setAttribute('aria-hidden', 'true'); qText.appendChild(qSizer); qText.appendChild(qTyped); qBox.appendChild(qText);
  var s1Box = el('div', 'vc-box vc-stage vc-s1'); s1Box.innerHTML = '<span>First stage</span>';
  var jeBox = el('div', 'vc-box vc-stage vc-je'); jeBox.innerHTML = '<span>Joint encoder</span>';
  /* the typical reranker runs two steps for every candidate where VEDJE runs its joint encoder */
  var pair = el('div', 'vc-pair');
  var veBox = el('div', 'vc-box vc-stage vc-ve'); veBox.innerHTML = '<span>Visual encoder</span>';
  var jmBox = el('div', 'vc-box vc-stage vc-jm'); jmBox.innerHTML = '<span>Large joint model</span>';
  pair.appendChild(veBox); pair.appendChild(jmBox);
  var calls = el('p', 'vc-calls'); calls.innerHTML = 'Visual encoder calls at query time: <b>0</b>';
  var callsNum = calls.querySelector('b');
  side.appendChild(qBox); side.appendChild(s1Box); side.appendChild(jeBox); side.appendChild(pair); side.appendChild(calls);

  var links = sv('svg', { class: 'vc-links', 'aria-hidden': 'true', focusable: 'false' });
  var fan = [], k;
  for (k = 0; k < NC; k++) { var fp = sv('path', { class: 'vc-link vc-link-s1' }); links.appendChild(fp); fan.push(fp); }
  var veLink = sv('path', { class: 'vc-link vc-link-ve' });
  var jeLink = sv('path', { class: 'vc-link vc-link-je' }), jeDot = sv('circle', { class: 'vc-link-dot', r: 2.6 });
  links.appendChild(veLink); links.appendChild(jeLink); links.appendChild(jeDot);

  var list = el('ol', 'vc-list');
  var scores = [], strips = [];
  var rows = CANDS.map(function (cd, i) {
    var li = el('li', 'vc-cand' + (cd.match ? ' is-match' : ''));
    li.style.setProperty('--i', i);
    var rk = el('span', 'vc-rk');
    rk.appendChild(el('b', 'vc-rank', 'Rank ' + cd.rank));
    rk.appendChild(el('span', 'vc-s1rank', 'First-stage rank ' + cd.s1));
    var th = el('span', 'vc-thumb'); th.setAttribute('aria-hidden', 'true');
    th.appendChild(scene(cd.kind, 8));
    var main = el('span', 'vc-main'), desc = el('span', 'vc-desc');
    if (cd.match) desc.innerHTML = '<svg class="vc-check" viewBox="0 0 12 12" role="img" aria-label="Matches the query" focusable="false">' +
      '<path d="M2.3 6.4l2.5 2.5 4.9-5.5"/></svg>';
    desc.appendChild(document.createTextNode(cd.label));
    var inp = el('span', 'vc-in'); inp.setAttribute('aria-hidden', 'true');
    var qt = el('span', 'vc-qt'); for (var q = 0; q < QUERY_TOKENS; q++) qt.appendChild(el('i'));
    var cs = el('span', 'vc-cs');
    for (var tt = 1; tt <= T; tt++) {
      var g = el('span', 'vc-tok-group');
      for (var mm = 0; mm < M; mm++) { var ii = el('i'); ii.style.opacity = tokShade(i, tt, mm); g.appendChild(ii); }
      cs.appendChild(g);
    }
    /* the typical reranker has no cache to read: the candidate's frames, lit while the visual encoder reads them */
    var fs = el('span', 'vc-fs'); strips.push(fs);
    inp.appendChild(qt); inp.appendChild(cs); inp.appendChild(fs);
    main.appendChild(desc); main.appendChild(inp);
    var pr = el('span', 'vc-prior', cd.prior.toFixed(2)); pr.setAttribute('title', 'First-stage score');
    var sc = el('span', 'vc-score', cd.score.toFixed(2)); scores.push(sc);
    li.appendChild(rk); li.appendChild(th); li.appendChild(main); li.appendChild(pr); li.appendChild(sc);
    list.appendChild(li);
    return li;
  });
  var listWrap = el('div', 'vc-listwrap'), listHead = el('div', 'vc-listhead');
  listHead.innerHTML = '<span class="vc-lh-c">Candidates</span><span class="vc-lh-s">First stage</span><span class="vc-lh-v">VEDJE</span>';
  var lhScore = listHead.querySelector('.vc-lh-v');
  listWrap.appendChild(listHead); listWrap.appendChild(list);
  body.appendChild(side); body.appendChild(links); body.appendChild(listWrap);
  pOnline.appendChild(body);
  /* both pipelines side by side, whichever is shown: VEDJE runs no visual encoder per query and has far fewer
     online parameters; the bars of each row share one scale */
  var costs = el('div', 'vc-costs');
  costs.appendChild(el('span', 'eyebrow vc-costs-t', 'Per query'));
  costs.appendChild(el('span', 'vc-costs-h is-vedje', 'VEDJE'));
  costs.appendChild(el('span', 'vc-costs-h is-typ', 'Typical reranker'));
  COSTS.forEach(function (c) {
    costs.appendChild(el('span', 'vc-cost-l', c[0]));
    [[c[1], 'is-vedje'], [c[2], 'is-typ']].forEach(function (v) {
      var cell = el('span', 'vc-cost ' + v[1]), track = el('span', 'vc-cost-track'), fill = el('i');
      fill.style.width = (100 * v[0][0] / Math.max(c[1][0], c[2][0])).toFixed(2) + '%';
      track.appendChild(fill); cell.appendChild(track); cell.appendChild(el('b', 'vc-cost-v', v[0][1]));
      costs.appendChild(cell);
    });
  });
  pOnline.appendChild(costs);
  root.appendChild(pOnline);

  /* controls and caption */
  var controls = el('div', 'sim-controls');
  var play = el('button', 'sim-play'); play.type = 'button'; play.setAttribute('aria-label', 'Play animation');
  play.innerHTML = '<svg class="ico-play" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l12-7.5z"/></svg>' +
    '<svg class="ico-pause" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 4.5h4v15H6zM14 4.5h4v15h-4z"/></svg>';
  var wrap = el('div', 'slider-track-wrapper');
  var slider = el('input', 'styled-slider'); slider.type = 'range'; slider.min = '1'; slider.max = String(N); slider.step = '1';
  slider.setAttribute('aria-label', 'Animation step');
  var sFill = el('div', 'slider-fill');
  wrap.appendChild(slider); wrap.appendChild(sFill);
  controls.appendChild(play); controls.appendChild(wrap);
  root.appendChild(controls);
  /* both captions share one grid cell, so the card keeps the height of the longer one in either mode */
  var capBox = el('div', 'vc-captions');
  var caps = {};
  Object.keys(CAPTIONS).forEach(function (md) { var p = el('p', 'sim-caption'); p.innerHTML = CAPTIONS[md]; capBox.appendChild(p); caps[md] = p; });
  root.appendChild(capBox);

  /* the frame strips (16 small frames per candidate) are drawn the first time the typical reranker is shown,
     so the default view builds no more than it shows */
  function drawStrips() {
    strips.forEach(function (fs, i) {
      if (fs.firstChild) return;
      for (var u = 1; u <= T; u++) { var mf = el('span', 'vc-mf'); mf.style.setProperty('--t', u - 1); mf.appendChild(scene(CANDS[i].kind, u)); fs.appendChild(mf); }
    });
  }

  /* ---------------- state and render ---------------- */
  var mode = /[?&]coverMode=typical\b/.test(search) ? 'typical' : 'vedje';
  var step = 1, playing = false, timer = null, typing = null, jmTimer = null;

  function setMode(md) {
    mode = md;
    var typ = md === 'typical';
    if (typ) drawStrips();
    root.classList.toggle('is-typical', typ);
    [].forEach.call(seg.querySelectorAll('button'), function (b) { b.setAttribute('aria-selected', b.getAttribute('data-mode') === md ? 'true' : 'false'); });
    Object.keys(caps).forEach(function (key) {
      var on = key === md;
      caps[key].classList.toggle('is-on', on);
      if (on) caps[key].removeAttribute('aria-hidden'); else caps[key].setAttribute('aria-hidden', 'true');
    });
    lhScore.textContent = typ ? 'Typical' : 'VEDJE';
    scores.forEach(function (s, i) { s.textContent = (typ ? CANDS[i].typ : CANDS[i].score).toFixed(2); });
    render(false);
  }

  function summary() {
    var typ = mode === 'typical', head = 'Step ' + step + ' of ' + N + '. ';
    if (step <= T) return head + 'Frame ' + step + ' of ' + T + (typ ?
      ' goes through the frozen visual encoder, which writes one first-stage embedding per video and no cache.' :
      ' is compressed into ' + M + ' tokens and stored as Z' + step + '. Stored per video: ' + kib(cacheBytes(step)) + '.');
    if (step === S_QUERY) return head + (typ ? 'The query arrives, and the first stage returns five candidates with their scores.' :
      'The query arrives, the first stage returns five candidates with their scores, and their caches are fetched.');
    if (step < S_SORT) { var i = step - S_READ, cd = CANDS[i];
      if (typ) return head + 'The visual encoder encodes the frames of candidate ' + (i + 1) + ', ' + cd.label.toLowerCase() +
        ', first-stage rank ' + cd.s1 + ', and the large joint model scores them with the query: ' + cd.typ.toFixed(2) +
        '. Visual encoder calls at query time: ' + (i + 1) + '.';
      return head + 'The joint encoder reads the query next to the cache of candidate ' + (i + 1) + ', ' +
        cd.label.toLowerCase() + ', first-stage rank ' + cd.s1 + '. With its first-stage score of ' + cd.prior.toFixed(2) +
        ', it scores ' + cd.score.toFixed(2) + '.'; }
    return head + 'The candidates are sorted by score, and the video where the person walks across the field ' +
      'moves from first-stage rank 3 to rank 1.' + (typ ? ' The visual encoder ran once for each of the five candidates.' : '');
  }

  function render(animate) {
    var typ = mode === 'typical', idx = Math.min(step, T), online = step > T;
    var reading = step >= S_READ && step < S_SORT ? step - S_READ : -1, sorted = step >= S_SORT;

    /* index panel */
    stIndex.textContent = online ? (typ ? 'Embedding stored' : 'Cache complete') : 'Frame ' + step + ' of ' + T;
    frames.forEach(function (f, i) {
      f.classList.toggle('is-done', i < idx || online);
      f.classList.toggle('is-now', !online && i === idx - 1);
    });
    enc.classList.toggle('is-on', !online);
    comp.classList.toggle('is-on', !online);
    pipe.classList.toggle('is-idle', online);
    var col = COLORS[idx - 1];
    for (var c2 = 0; c2 < P; c2++) cells[c2].setAttribute('fill', col[c2]);
    outToks.forEach(function (o, j) { o.style.opacity = tokShade(MATCH, idx, j); });
    outLbl.textContent = 'Z' + idx;
    embCells.forEach(function (c3, j) { c3.style.opacity = embShade(j, idx); });
    /* the typical reranker writes no cache, so its slots stay empty */
    slots.forEach(function (s, i) {
      var now = !typ && !online && i === idx - 1;
      s.classList.toggle('is-full', !typ && i < idx);
      s.classList.toggle('is-now', now);
      s.classList.toggle('is-new', !!animate && now);
    });
    zl.forEach(function (z, i) { z.classList.toggle('is-now', !online && i === idx - 1); });
    /* the meter: the cache on its 48 KiB scale, or the first-stage embedding alone (its width comes from cover.css) */
    fill.style.width = typ ? '' : (cacheBytes(idx) / cacheBytes(T) * 100) + '%';
    meterVal.textContent = typ ? 'Embedding only' : kib(cacheBytes(idx));

    /* online panel */
    pOnline.classList.toggle('is-wait', !online);
    stOnline.textContent = !online ? 'No query yet' : step === S_QUERY ? 'Five candidates' : sorted ? 'Sorted by score' : 'Candidate ' + (reading + 1) + ' of ' + NC;
    if (typing) { clearInterval(typing); typing = null; }
    if (!online) { qTyped.textContent = ''; qText.classList.add('is-empty'); }
    else {
      qText.classList.remove('is-empty');
      if (animate && step === S_QUERY && !reduced && !STATIC) {
        var n = 0; qTyped.textContent = '';
        typing = setInterval(function () { n += 1; qTyped.textContent = QUERY.slice(0, n); if (n >= QUERY.length) { clearInterval(typing); typing = null; } }, 24);
      } else qTyped.textContent = QUERY;
    }
    s1Box.classList.toggle('is-on', step === S_QUERY);
    jeBox.classList.toggle('is-on', !typ && reading >= 0);
    veBox.classList.toggle('is-on', typ && reading >= 0);
    /* the large joint model scores each candidate once the visual encoder has gone over its frames */
    clearTimeout(jmTimer);
    jmBox.classList.toggle('is-on', typ && reading >= 0 && !animate);
    if (typ && reading >= 0 && animate) jmTimer = setTimeout(function () { jmBox.classList.add('is-on'); }, 1800);
    /* one visual encoder call per candidate read by the typical reranker; none for VEDJE */
    callsNum.textContent = typ ? (sorted ? NC : reading + 1) : 0;
    rows.forEach(function (row, i) {
      var cd = CANDS[i];
      row.classList.toggle('is-reading', i === reading);
      row.classList.toggle('is-scored', (reading >= 0 && i < reading) || sorted);
      row.classList.toggle('is-sorted', sorted);
      row.classList.toggle('is-top', sorted && cd.rank === 1);
      row.style.setProperty('--pos', sorted ? cd.rank - 1 : i);
    });
    list.classList.toggle('is-sorted', sorted);
    body.setAttribute('data-phase', !online ? 'wait' : step === S_QUERY ? 'first' : sorted ? 'sorted' : 'read');
    drawLinks();

    /* controls */
    stepNum.textContent = step;
    slider.value = String(step);
    slider.setAttribute('aria-valuetext', summary());
    var ratio = (step - 1) / (N - 1);
    sFill.style.width = 'calc(' + ratio.toFixed(4) + ' * (100% - var(--vc-thumb, 24px)) + var(--vc-thumb, 24px) / 2)';
  }

  /* the pairing lines beside the candidate list (only where the side column sits left of the list) */
  function drawLinks() {
    var br = body.getBoundingClientRect(), lr = list.getBoundingClientRect(), sr = side.getBoundingClientRect();
    var beside = sr.right <= lr.left + 1 && br.width > 0;
    links.classList.toggle('is-off', !beside);
    if (!beside) return;
    /* on wide screens the cover is zoomed: client rects are in screen pixels, the lines are drawn in the cover's own */
    var k = br.width / (body.offsetWidth || br.width), bw = Math.round(br.width / k), bh = Math.round(br.height / k);
    links.setAttribute('width', bw); links.setAttribute('height', bh);
    links.setAttribute('viewBox', '0 0 ' + bw + ' ' + bh);
    var cs = getComputedStyle(list), rowH = parseFloat(cs.getPropertyValue('--vc-row')) || 28, gap = parseFloat(cs.getPropertyValue('--vc-gap')) || 5;
    var x1 = (lr.left - br.left) / k - 3;
    function rowY(i) { var pos = parseFloat(rows[i].style.getPropertyValue('--pos')) || 0; return (lr.top - br.top) / k + pos * (rowH + gap) + rowH / 2; }
    function from(box) { var r = box.getBoundingClientRect(); return [(r.right - br.left) / k + 1, (r.top - br.top + r.height / 2) / k]; }
    function curve(a, y) { var mx = (a[0] + x1) / 2; return 'M' + a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' C' + mx.toFixed(1) + ' ' + a[1].toFixed(1) + ' ' + mx.toFixed(1) + ' ' + y.toFixed(1) + ' ' + x1.toFixed(1) + ' ' + y.toFixed(1); }
    /* the typical reranker joins both of its steps to the candidate it reads: the visual encoder and the joint model */
    var typ = mode === 'typical', a1 = from(s1Box), a2 = from(typ ? jmBox : jeBox), phase = body.getAttribute('data-phase');
    fan.forEach(function (p, i) { p.setAttribute('d', curve(a1, rowY(i))); });
    links.classList.toggle('show-s1', phase === 'first');
    var reading = step >= S_READ && step < S_SORT ? step - S_READ : -1;
    links.classList.toggle('show-je', reading >= 0);
    links.classList.toggle('show-ve', typ && reading >= 0);
    if (reading >= 0) {
      var y = rowY(reading);
      jeLink.setAttribute('d', curve(a2, y)); jeDot.setAttribute('cx', x1.toFixed(1)); jeDot.setAttribute('cy', y.toFixed(1));
      if (typ) veLink.setAttribute('d', curve(from(veBox), y));
    }
  }

  function setStep(s, animate) {
    step = Math.max(1, Math.min(N, s));
    render(animate);
  }

  /* ---------------- playback ---------------- */
  function dur(s) {
    if (s <= T) return 520;            // one frame per beat
    if (s === S_QUERY) return 1500;    // the query arrives and the first stage returns its candidates
    if (s < S_SORT) return mode === 'typical' ? 2400 : 1100;   // one candidate per beat; re-encoding takes longer
    return 3800;                       // hold the sorted list, then loop
  }
  function schedule() {
    clearTimeout(timer); timer = null;
    if (!playing) return;
    timer = setTimeout(function () { setStep(step >= N ? 1 : step + 1, true); schedule(); }, dur(step));
  }
  /* the cover plays while it is in view, the page is visible and the reader has not paused it */
  var userPaused = STATIC || !!stepFlag || reduced, inView = true;
  function setPlaying(p) {
    playing = p;
    play.classList.toggle('is-playing', p);
    play.setAttribute('aria-label', p ? 'Pause animation' : 'Play animation');
    schedule();
  }
  function sync() {
    var want = inView && !userPaused && !document.hidden;
    if (want !== playing) setPlaying(want);
  }
  play.addEventListener('click', function () {
    userPaused = playing;
    if (!userPaused && step >= N) setStep(1, true);
    setPlaying(!userPaused);
  });
  slider.addEventListener('input', function () {
    userPaused = true; setPlaying(false);
    setStep(parseInt(slider.value, 10), false);
  });
  seg.addEventListener('click', function (e) {
    var b = e.target.closest('button[data-mode]');
    if (b && b.getAttribute('data-mode') !== mode) setMode(b.getAttribute('data-mode'));
  });
  document.addEventListener('visibilitychange', sync);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      inView = entries[entries.length - 1].isIntersecting; sync();
    }, { threshold: 0.15 }).observe(document.getElementById('heroDemo') || root);
  }
  if ('ResizeObserver' in window) new ResizeObserver(function () { drawLinks(); }).observe(root);
  else window.addEventListener('resize', drawLinks);

  /* wide screens (style.css, "The cover on wide screens"): zoom the 560px cover to fill its column, as far as
     the window height allows; the caption (the last 80px of the 756px reserved height) may fall below the fold */
  var host = document.getElementById('heroDemo'), header = host && host.parentElement, fitQueued = false;
  function fit() {
    fitQueued = false;
    if (!host || !header) return;
    var beside = window.innerWidth >= 1440 && getComputedStyle(header).flexDirection === 'row';
    var z = beside ? Math.min(host.clientWidth / 560, (window.innerHeight - host.getBoundingClientRect().top - window.scrollY - 16) / 676, 1.6) : 0;
    root.style.zoom = beside ? Math.max(1, z).toFixed(3) : '';
    drawLinks();
  }
  window.addEventListener('resize', function () { if (!fitQueued) { fitQueued = true; requestAnimationFrame(fit); } });
  fit();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawLinks);

  /* ---------------- start ---------------- */
  setMode(mode);
  setStep(stepFlag ? parseInt(stepFlag[1], 10) : (STATIC || reduced ? STATIC_STEP : 1), false);
  sync();
})();
