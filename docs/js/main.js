/* VEDJE: project page interactions.
   Vanilla JS, no dependencies. The cover lives in js/cover.js and the pipeline animation in js/overview.js.
   Sections:
   1. Mobile nav drawer (ported from Decart Research nav.js)
   2. Charts built from the paper's tables (Figure 1 and Table 22, Tables 1b, 12, 3, 9, 4, 5 and 6)
   3. Small screens: jump bar and the short version of each act
   4. Quick-links rail, share, BibTeX copy, reveal on scroll
   5. Colab links */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVGNS = 'http://www.w3.org/2000/svg';
  /* ?static=1 freezes every transition so headless captures show the final state */
  if (/[?&]static\b/.test(location.search)) document.documentElement.classList.add('is-static');

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function mk(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function segSelect(seg, attr, value) {
    $$('button', seg).forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute(attr) === value ? 'true' : 'false'); });
  }
  function one(v) { return (Math.round(v * 10) / 10).toFixed(1); }
  function signed(v) { var r = Math.round(v * 10) / 10; return (r > 0 ? '+' : (r < 0 ? '−' : '')) + Math.abs(r).toFixed(1); }

  /* ------------------------------------------------------------------ */
  /* 1. Mobile navigation drawer                                          */
  /* ------------------------------------------------------------------ */
  (function nav() {
    var toggle = $('.nav-toggle'), links = $('#navLinks');
    if (!toggle || !links) return;
    function setOpen(open) {
      document.body.classList.toggle('nav-open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    }
    toggle.addEventListener('click', function () { setOpen(!document.body.classList.contains('nav-open')); });
    links.addEventListener('click', function (e) { if (e.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setOpen(false); });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) setOpen(false); });
  })();

  /* ------------------------------------------------------------------ */
  /* 2. Charts                                                            */
  /* ------------------------------------------------------------------ */
  var tip = $('#chartTip');
  function bindTips(root) {
    $$('[data-tip]', root).forEach(function (n) {
      if (n.__tip) return; n.__tip = true;
      n.addEventListener('pointerenter', function (e) { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); moveTip(e); });
      n.addEventListener('pointermove', moveTip);
      n.addEventListener('pointerleave', function () { tip.classList.remove('is-on'); });
      n.addEventListener('focus', function () { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); var r = n.getBoundingClientRect(); tip.style.left = (r.left + r.width / 2) + 'px'; tip.style.top = r.top + 'px'; });
      n.addEventListener('blur', function () { tip.classList.remove('is-on'); });
    });
  }
  function moveTip(e) { tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; }
  function onResize(node, fn) {
    var last = 0;
    function run() { var w = node.clientWidth; if (w && Math.abs(w - last) > 2) { last = w; fn(w); } }
    if ('ResizeObserver' in window) new ResizeObserver(run).observe(node); else window.addEventListener('resize', run);
    run();
  }

  /* Gain bars that grow from a zero line at the left.
     rows: [{label, sub, gain, from, to, ours, tip}]; max: axis maximum in points; ticks: values */
  function renderGains(host, rows, max, ticks, unit) {
    host.innerHTML = '';
    rows.forEach(function (r) {
      var row = el('div', 'gain-row' + (r.ours ? ' is-ours' : ''));
      var lab = el('div', 'gain-label'); lab.appendChild(document.createTextNode(r.label));
      if (r.sub) lab.appendChild(el('small', null, r.sub));
      if (r.from != null) lab.appendChild(el('small', 'gain-from', one(r.from) + ' to ' + one(r.to)));
      row.appendChild(lab);
      var track = el('div', 'gain-track'); track.setAttribute('tabindex', '0');
      if (r.tip) track.setAttribute('data-tip', r.tip);
      track.appendChild(el('span', 'gain-zero'));
      var w = Math.max(0, Math.min(r.gain, max)) / max * 100, inside = w > 74;
      var bar = el('div', 'gain-bar ' + (r.base ? 'is-loss' : 'is-gain'));
      bar.style.width = '0%'; bar.setAttribute('data-w', Math.max(0.6, w));
      track.appendChild(bar);
      var val = el('span', 'gain-val ' + (r.base ? 'is-loss' : 'is-gain') + (inside ? ' is-inside' : ''), signed(r.gain) + (unit || ''));
      if (inside) { val.style.right = (100 - w) + '%'; } else { val.style.left = w + '%'; }
      track.appendChild(val);
      row.appendChild(track);
      host.appendChild(row);
    });
    var axis = el('div', 'gain-row gain-axis'); axis.appendChild(el('div'));
    var tk = el('div', 'gain-ticks');
    ticks.forEach(function (t) { var s = el('span', t === 0 ? 'is-first' : null, t === 0 ? '0' : '+' + t); s.style.left = (t / max * 100) + '%'; tk.appendChild(s); });
    axis.appendChild(tk); host.appendChild(axis);
    bindTips(host);
    requestAnimationFrame(function () { $$('.gain-bar', host).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
  }
  function rangeText(list) {
    var lo = Math.min.apply(null, list), hi = Math.max.apply(null, list);
    return lo === hi ? one(lo) : one(lo) + ' to ' + one(hi);
  }

  /* --- Figure 1 and Table 22: MSR-VTT text-to-video R@1 against online parameters ---
     Values as printed in Table 22 (parameters in millions). Counts cover online text encoding and learned
     scoring; the MLLM rows count the base decoder and Video-ColBERT its full SigLIP text module. */
  var PUBLISHED = [
    { name: 'CLIP4Clip', params: 63.4, r1: 46.4 },
    { name: 'X-CLIP', params: 64.0, r1: 49.3 },
    { name: 'EERCF', params: 63.7, r1: 49.9 },
    { name: 'Video-ColBERT', params: 110.3, r1: 51.5 },
    { name: 'LamRA', params: 7600, r1: 59.7, decoder: true },
    { name: 'CaRe-DPO', params: 7600, r1: 64.1, decoder: true }
  ];
  var VEDJE_FT = { name: 'VEDJE', params: 157, r1: 59.8 };
  function fmtParams(m) { return m >= 1000 ? (m / 1000).toFixed(1).replace(/\.0$/, '') + 'B' : (m >= 100 || m === Math.round(m) ? Math.round(m) : m.toFixed(1)) + 'M'; }

  (function frontier() {
    var host = $('#chartFrontier'); if (!host) return;
    var note = $('#frontierNote');
    note.textContent = 'VEDJE uses the fine-tuned VideoCLIP-XL first stage. Counts cover online text encoding and scoring (Appendix B). CaRe-DPO reaches 64.1 with a decoder of the same size as LamRA.';
    function draw(W) {
      var narrow = W < 560;
      var H = narrow ? Math.round(W * 0.92) : Math.round(Math.min(420, Math.max(300, W * 0.46)));
      var m = narrow ? { l: 44, r: 14, t: 18, b: 48 } : { l: 54, r: 24, t: 18, b: 50 };
      var xr = [50, 12000], yr = [44, 66];
      var lx0 = Math.log10(xr[0]), lx1 = Math.log10(xr[1]);
      function X(p) { return m.l + (Math.log10(p) - lx0) / (lx1 - lx0) * (W - m.l - m.r); }
      function Y(r) { return m.t + (1 - (r - yr[0]) / (yr[1] - yr[0])) * (H - m.t - m.b); }
      var svg = mk('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'class': narrow ? 'is-narrow' : '',
        'aria-label': 'Text-to-video R@1 on MSR-VTT against online parameters for VEDJE and six published systems' });
      [45, 50, 55, 60, 65].forEach(function (v) {
        svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), 'class': 'grid' }));
        svg.appendChild(mk('text', { x: m.l - 8, y: Y(v) + 4, 'class': 'tick', 'text-anchor': 'end' }, String(v)));
      });
      (narrow ? [100, 1000, 10000] : [63, 157, 300, 1000, 3000, 7600]).forEach(function (v) {
        svg.appendChild(mk('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, 'class': 'grid' }));
        svg.appendChild(mk('text', { x: X(v), y: H - m.b + 17, 'class': 'tick', 'text-anchor': 'middle' }, fmtParams(v)));
      });
      svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, 'class': 'axis' }));
      svg.appendChild(mk('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'class': 'axis-title', 'text-anchor': 'middle' }, 'Online parameters, log scale'));
      svg.appendChild(mk('text', { x: 12, y: (m.t + H - m.b) / 2, 'class': 'axis-title', 'text-anchor': 'middle', transform: 'rotate(-90 12 ' + ((m.t + H - m.b) / 2) + ')' }, 'R@1 (%)'));
      function point(p, cls, r, tipText) {
        var g = mk('g', { 'class': 'pt-g', tabindex: '0', 'data-tip': tipText });
        if (cls === 'is-ours') g.appendChild(mk('circle', { cx: X(p[0]), cy: Y(p[1]), r: r + 10, 'class': 'ring' }));
        g.appendChild(mk('circle', { cx: X(p[0]), cy: Y(p[1]), r: r, 'class': 'pt ' + cls }));
        svg.appendChild(g);
      }
      function label(x, y, text, cls, anchor) { svg.appendChild(mk('text', { x: x, y: y, 'class': 'lbl ' + (cls || ''), 'text-anchor': anchor || 'start' }, text)); }
      /* label placement for the clustered points near 63M (CLIP4Clip, X-CLIP, EERCF) */
      var LBL = {
        'CLIP4Clip': { dx: 12, dy: 4 },
        'X-CLIP': { dx: 12, dy: 2, leader: true, lift: -16 },
        'EERCF': { dx: 12, dy: -6, leader: true, lift: -34 },
        'Video-ColBERT': { dx: 10, dy: -9 },
        'LamRA': { dx: -10, dy: 4, anchor: 'end' },
        'CaRe-DPO': { dx: -10, dy: 4, anchor: 'end' }
      };
      /* phones: the points near 63M and 110M are too close for side labels, so their labels form a column
         right of VEDJE, each joined to its point by a leader */
      var COL = { 'X-CLIP': 48.6, 'EERCF': 51.2, 'Video-ColBERT': 53.8 };
      PUBLISHED.forEach(function (p) {
        point([p.params, p.r1], 'is-base', narrow ? 6 : 5.5, p.name + ' · ' + (p.decoder ? '7.6B-parameter decoder' : fmtParams(p.params) + ' online parameters') + ' · R@1 ' + one(p.r1));
        var L = LBL[p.name], x = X(p.params), y = Y(p.r1);
        if (narrow && COL[p.name]) {
          var cx = X(240), cy = Y(COL[p.name]);
          svg.appendChild(mk('line', { x1: x + 5, y1: y, x2: cx - 4, y2: cy - 4, 'class': 'leader' }));
          label(cx, cy, p.name, '', 'start');
        } else if (L.leader) {
          var ly = y + L.lift;
          svg.appendChild(mk('line', { x1: x + 4, y1: y - 3, x2: x + L.dx + 4, y2: ly + 2, 'class': 'leader' }));
          label(x + L.dx + 6, ly, p.name, '', 'start');
        } else label(x + L.dx, y + L.dy, p.name, '', L.anchor);
      });
      point([VEDJE_FT.params, VEDJE_FT.r1], 'is-ours', narrow ? 8 : 7.5, 'VEDJE · about 157M online parameters · R@1 59.8');
      label(X(VEDJE_FT.params) + 24, Y(VEDJE_FT.r1) - 10, 'VEDJE', 'is-ours');
      host.innerHTML = ''; host.appendChild(svg); bindTips(host);
    }
    onResize(host, draw);
  })();

  /* --- Table 1b: R@1 of each first stage on MSR-VTT, before and after VEDJE, as printed: [stage 1, VEDJE] --- */
  var MATCHED = [
    { label: 'VideoCLIP-XL', sub: 'fine-tuned', t2v: [56.2, 59.8], v2t: [55.1, 58.8] },
    { label: 'VideoCLIP-XL', sub: 'zero-shot', t2v: [50.1, 56.5], v2t: [49.9, 57.1] },
    { label: 'VideoPrism', sub: '', t2v: [50.1, 54.6], v2t: [49.8, 54.6] },
    { label: 'PE-Core-B', sub: '', t2v: [47.6, 53.9], v2t: [47.3, 53.9] }
  ];
  var DIR_NAMES = { t2v: 'text to video', v2t: 'video to text' };
  (function matched() {
    var host = $('#chartMatched'); if (!host) return;
    var seg = $('#matchedDir'), note = $('#matchedNote');
    function draw(dir) {
      var rows = MATCHED.map(function (m) {
        var a = m[dir][0], b = m[dir][1];
        return { label: m.label, sub: m.sub, from: a, to: b, gain: b - a,
          tip: m.label + (m.sub ? ', ' + m.sub : '') + ' · ' + DIR_NAMES[dir] + ' · R@1 ' + one(a) + ' to ' + one(b) + ' with VEDJE (Table 1b)' };
      });
      renderGains(host, rows, 8, [0, 2, 4, 6, 8]);
      note.innerHTML = '<strong>' + (dir === 't2v' ? 'Text to video.' : 'Video to text.') + '</strong> R@1 rises by ' + rangeText(rows.map(function (r) { return Math.round(r.gain * 10) / 10; })) +
        ' points over the matched first stages.';
    }
    seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-dir]'); if (!b) return; segSelect(seg, 'data-dir', b.getAttribute('data-dir')); draw(b.getAttribute('data-dir')); });
    draw('t2v');
  })();

  /* --- Table 12: text-to-video R@1 with the zero-shot VideoCLIP-XL first stage, as printed: [stage 1, VEDJE] --- */
  var DATASETS = [
    { label: 'MSR-VTT', t2v: [50.1, 56.5] },
    { label: 'MSVD', t2v: [51.9, 56.8] },
    { label: 'DiDeMo', t2v: [47.7, 56.6] },
    { label: 'ActivityNet', t2v: [46.4, 51.6] }
  ];
  (function datasets() {
    var host = $('#chartDatasets'); if (!host) return;
    var note = $('#datasetsNote');
    var rows = DATASETS.map(function (d) {
      var a = d.t2v[0], b = d.t2v[1];
      return { label: d.label, from: a, to: b, gain: b - a, tip: d.label + ' · text to video · R@1 ' + one(a) + ' to ' + one(b) + ' with VEDJE (Table 12)' };
    });
    renderGains(host, rows, 10, [0, 2, 4, 6, 8, 10]);
    note.innerHTML = 'Text-to-video R@1 rises by ' + rangeText(rows.map(function (r) { return Math.round(r.gain * 10) / 10; })) + ' points with the zero-shot VideoCLIP-XL first stage.';
  })();

  /* --- Tables 9 and 3: visual storage per video at two bytes per element, and T2V R@1 of the two caches --- */
  var STORAGE = [
    { label: 'Frame-and-patch features', sub: 'uncompressed', bytes: 16 * 257 * 768 * 2, text: '6.02\u00a0MiB' },
    { label: 'VEDJE cache', sub: '64 tokens', bytes: 64 * 384 * 2, text: '48\u00a0KiB', r1: '54.6', ours: true },
    { label: 'VEDJE cache', sub: '16 tokens', bytes: 16 * 384 * 2, text: '12\u00a0KiB', r1: '54.4', ours: true }
  ];
  (function storage() {
    var host = $('#chartStorage'); if (!host) return;
    var note = $('#storageNote');
    var LO = 12, HI = 23; /* log2 of 4 KiB and 8 MiB */
    function x(bytes) { return (Math.log(bytes) / Math.LN2 - LO) / (HI - LO) * 100; }
    host.innerHTML = '';
    STORAGE.forEach(function (s) {
      var row = el('div', 'store-row');
      var lab = el('div', 'store-label'); lab.appendChild(document.createTextNode(s.label)); lab.appendChild(el('small', null, s.sub)); row.appendChild(lab);
      var track = el('div', 'store-track'); track.setAttribute('tabindex', '0');
      track.setAttribute('data-tip', s.label + ', ' + s.sub + ' · ' + s.bytes.toLocaleString('en-US') + ' bytes per video' + (s.r1 ? ' · text-to-video R@1 ' + s.r1 + ' (Table 3)' : ' (Table 9)'));
      var w = x(s.bytes), inside = w > 70;
      var bar = el('div', 'store-bar' + (s.ours ? ' is-ours' : '')); bar.style.width = '0%'; bar.setAttribute('data-w', w); track.appendChild(bar);
      var val = el('span', 'store-val' + (s.ours ? ' is-ours' : '') + (inside ? ' is-inside' : ''));
      val.appendChild(el('b', null, s.text));
      if (s.r1) val.appendChild(el('em', null, 'T2V R@1 ' + s.r1));
      if (inside) val.style.right = (100 - w) + '%'; else val.style.left = w + '%';
      track.appendChild(val);
      row.appendChild(track); host.appendChild(row);
    });
    var axis = el('div', 'store-row'); axis.appendChild(el('div'));
    var tk = el('div', 'store-ticks');
    [[12, '4\u00a0KiB'], [14, '16\u00a0KiB'], [16, '64\u00a0KiB'], [18, '256\u00a0KiB'], [20, '1\u00a0MiB'], [22, '4\u00a0MiB']].forEach(function (t, i, a) {
      var sp = el('span', i === 0 ? 'is-first' : (i % 2 ? 'is-minor' : null), t[1]); sp.style.left = ((t[0] - LO) / (HI - LO) * 100) + '%'; tk.appendChild(sp);
    });
    axis.appendChild(tk); host.appendChild(axis);
    bindTips(host);
    requestAnimationFrame(function () { $$('.store-bar', host).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
    note.innerHTML = 'Both caches are trained with feature-change supervision.';
  })();

  /* --- Table 4: cache structure at a fixed storage budget, MSR-VTT R@1, mean of three runs, as printed --- */
  var STRUCTURE = {
    attention: { t2v: 52.0, v2t: 51.6 },  /* pooled cache with attention pooling: the chart's baseline */
    mean: { t2v: 51.7, v2t: 51.6 },       /* pooled cache with mean pooling, named in the note */
    rows: [
      { label: 'Separate frame groups', sub: 'without feature change', t2v: 54.0, v2t: 53.6 },
      { label: 'Separate frame groups', sub: 'with feature change', t2v: 54.6, v2t: 54.6, ours: true }
    ]
  };
  (function structure() {
    var host = $('#chartStructure'); if (!host) return;
    var seg = $('#structureDir'), note = $('#structureNote');
    function draw(dir) {
      var base = STRUCTURE.attention[dir];
      var rows = STRUCTURE.rows.map(function (r) {
        return { label: r.label, sub: r.sub, from: base, to: r[dir], gain: r[dir] - base, ours: r.ours,
          tip: r.label + ', ' + r.sub + ' · ' + DIR_NAMES[dir] + ' · R@1 ' + one(r[dir]) + ' against ' + one(base) + ' with attention pooling (Table 4)' };
      });
      renderGains(host, rows, 4, [0, 1, 2, 3, 4]);
      note.innerHTML = '<strong>' + (dir === 't2v' ? 'Text to video.' : 'Video to text.') + '</strong> The pooled caches reach ' + one(STRUCTURE.attention[dir]) + ' R@1 with attention pooling and ' + one(STRUCTURE.mean[dir]) +
        ' with mean pooling. Separate frame groups reach ' + one(STRUCTURE.rows[0][dir]) + ', and ' + one(STRUCTURE.rows[1][dir]) + ' with feature-change supervision.';
    }
    seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-dir]'); if (!b) return; segSelect(seg, 'data-dir', b.getAttribute('data-dir')); draw(b.getAttribute('data-dir')); });
    draw('t2v');
  })();

  /* --- Table 3: R@1 without and with feature-change supervision, MSR-VTT, mean of three runs, as printed --- */
  var DELTA = [
    { label: '64 tokens', sub: '48\u00a0KiB per video', t2v: [54.0, 54.6], v2t: [53.6, 54.6] },
    { label: '16 tokens', sub: '12\u00a0KiB per video', t2v: [52.5, 54.4], v2t: [52.6, 53.1] }
  ];
  (function delta() {
    var host = $('#chartDelta'); if (!host) return;
    var seg = $('#deltaDir'), note = $('#deltaNote');
    function draw(dir) {
      var rows = DELTA.map(function (d) {
        var a = d[dir][0], b = d[dir][1];
        return { label: d.label, sub: d.sub, from: a, to: b, gain: b - a, tip: d.label + ' · ' + DIR_NAMES[dir] + ' · R@1 ' + one(a) + ' without and ' + one(b) + ' with feature-change supervision (Table 3)' };
      });
      renderGains(host, rows, 2, [0, 0.5, 1, 1.5, 2]);
      var g = rows.map(function (r) { return one(r.gain); });
      note.innerHTML = '<strong>' + (dir === 't2v' ? 'Text to video.' : 'Video to text.') + '</strong> The gain is ' + g[0] + ' points with 64 tokens and ' + g[1] + ' points with 16 tokens.';
    }
    seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-dir]'); if (!b) return; segSelect(seg, 'data-dir', b.getAttribute('data-dir')); draw(b.getAttribute('data-dir')); });
    draw('t2v');
  })();

  /* --- Table 5: scoring the same cache, MSR-VTT text-to-video R@1, VideoPrism first stage, 20 candidates, as printed --- */
  var SCORER = {
    stage1: 50.1,
    rows: [
      { label: 'Trained MaxSim', sub: 'with the prior', r1: 51.9, base: true },
      { label: 'Joint encoder', sub: 'with the prior, VEDJE', r1: 54.6, ours: true }
    ]
  };
  (function scorer() {
    var host = $('#chartScorer'); if (!host) return;
    var note = $('#scorerNote');
    var rows = SCORER.rows.map(function (r) {
      return { label: r.label, sub: r.sub, from: SCORER.stage1, to: r.r1, gain: r.r1 - SCORER.stage1, ours: r.ours, base: r.base,
        tip: r.label + ' ' + r.sub + ' · text to video · R@1 ' + one(r.r1) + ' against ' + one(SCORER.stage1) + ' for the first stage (Table 5)' };
    });
    renderGains(host, rows, 5, [0, 1, 2, 3, 4, 5]);
    note.innerHTML = 'Over the VideoPrism first stage at ' + one(SCORER.stage1) + ' R@1, trained MaxSim reaches ' + one(SCORER.rows[0].r1) + ' and the joint encoder ' + one(SCORER.rows[1].r1) + '.';
  })();

  /* --- Table 6: complete-query latency in ms, MSR-VTT text to video, VideoPrism first stage, as printed --- */
  var LATENCY = { cands: [20, 50, 100, 200], t64: [5.79, 6.21, 7.83, 10.24], t16: [5.82, 6.12, 6.33, 6.88], stage1: 4.3 };
  (function latency() {
    var host = $('#chartLatency'); if (!host) return;
    var note = $('#latencyNote');
    var legend = el('div', 'chart-legend');
    legend.innerHTML = '<span><i class="is-light"></i>64 tokens · 48 KiB</span><span><i class="is-ours"></i>16 tokens · 12 KiB</span>';
    host.parentNode.insertBefore(legend, host);
    function draw(W) {
      var narrow = W < 560, H = narrow ? Math.round(W * 0.78) : Math.round(Math.min(320, Math.max(240, W * 0.36)));
      var m = narrow ? { l: 40, r: 50, t: 14, b: 44 } : { l: 48, r: 64, t: 14, b: 46 };
      var lx0 = Math.log10(16), lx1 = Math.log10(240), y0 = 3, y1 = 11;
      function X(c) { return m.l + (Math.log10(c) - lx0) / (lx1 - lx0) * (W - m.l - m.r); }
      function Y(v) { return m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b); }
      var svg = mk('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'class': narrow ? 'is-narrow' : '', 'aria-label': 'Complete-query latency in milliseconds against the number of candidates for the 64-token and the 16-token cache' });
      [4, 6, 8, 10].forEach(function (v) {
        svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), 'class': 'grid' }));
        svg.appendChild(mk('text', { x: m.l - 8, y: Y(v) + 4, 'class': 'tick', 'text-anchor': 'end' }, v + ' ms'));
      });
      LATENCY.cands.forEach(function (c) { svg.appendChild(mk('text', { x: X(c), y: H - m.b + 17, 'class': 'tick', 'text-anchor': 'middle' }, String(c))); });
      svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, 'class': 'axis' }));
      svg.appendChild(mk('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'class': 'axis-title', 'text-anchor': 'middle' }, 'Candidates reranked per query'));
      /* first-stage retrieval alone, about 4.3 ms (Section 4.7) */
      svg.appendChild(mk('line', { x1: m.l, x2: W - m.r, y1: Y(LATENCY.stage1), y2: Y(LATENCY.stage1), 'class': 'grid', 'stroke-dasharray': '4 4', stroke: '#BDBDBD' }));
      svg.appendChild(mk('text', { x: W - m.r, y: Y(LATENCY.stage1) + 15, 'class': 'tick', 'text-anchor': 'end' }, 'first stage alone, about 4.3\u00a0ms'));
      [['t64', 'is-light'], ['t16', 'is-ours']].forEach(function (s) {
        var pts = LATENCY.cands.map(function (c, i) { return X(c).toFixed(1) + ',' + Y(LATENCY[s[0]][i]).toFixed(1); });
        svg.appendChild(mk('polyline', { points: pts.join(' '), 'class': 'line ' + s[1] }));
        LATENCY.cands.forEach(function (c, i) {
          var v = LATENCY[s[0]][i];
          var d = mk('circle', { cx: X(c), cy: Y(v), r: narrow ? 5 : 4.5, 'class': 'dot ' + s[1], tabindex: '0', 'data-tip': (s[0] === 't64' ? '64 tokens' : '16 tokens') + ' · ' + c + ' candidates · ' + v.toFixed(2) + ' ms (Table 6)' });
          svg.appendChild(d);
        });
        var last = LATENCY[s[0]][LATENCY.cands.length - 1];
        svg.appendChild(mk('text', { x: X(200) + 9, y: Y(last) + 4, 'class': 'val ' + s[1] }, last.toFixed(2) + ' ms'));
      });
      host.innerHTML = ''; host.appendChild(svg); bindTips(host);
    }
    onResize(host, draw);
    note.innerHTML = 'At 20 candidates the two caches take ' + LATENCY.t64[0].toFixed(2) + ' and ' + LATENCY.t16[0].toFixed(2) + ' ms; at 200 candidates, ' + LATENCY.t64[3].toFixed(2) + ' and ' + LATENCY.t16[3].toFixed(2) + ' ms. First-stage retrieval alone takes about 4.3\u00a0ms.';
  })();

  /* ------------------------------------------------------------------ */
  /* 3. Small screens: jump bar and the short version of each act         */
  /* ------------------------------------------------------------------ */
  (function smallScreens() {
    var main = $('.article-main'), toc = $$('.toc a');
    if (main && toc.length) {
      var bar = el('nav', 'jumpbar'); bar.setAttribute('aria-label', 'Jump to a section');
      toc.forEach(function (a) {
        var link = el('a', 'jump'); link.setAttribute('href', a.getAttribute('href'));
        var num = a.querySelector('small');
        if (num) link.appendChild(el('small', null, num.textContent));
        link.appendChild(document.createTextNode(a.textContent.replace(num ? num.textContent : '', '').trim()));
        bar.appendChild(link);
      });
      main.insertBefore(bar, main.firstChild);
    }
    var mq = window.matchMedia ? window.matchMedia('(max-width: 700px)') : null;
    if (!mq) return;
    $$('.text-block[id]').forEach(function (act) {
      var deep = $$('.deep', act); if (!deep.length) return;
      var btn = el('button', 'more-btn'); btn.type = 'button';
      var lbl = el('span'), chev = el('i', 'chev'); btn.appendChild(lbl); btn.appendChild(chev);
      var closing = act.querySelector(':scope > .article-closing-cta');
      if (closing) act.insertBefore(btn, closing); else act.appendChild(btn);
      function apply() {
        var narrow = mq.matches, open = act.classList.contains('is-open');
        deep.forEach(function (d) { if (narrow && !open) d.setAttribute('hidden', ''); else d.removeAttribute('hidden'); });
        btn.hidden = !narrow;
        lbl.textContent = open ? 'Show the short version' : 'Read the full section';
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
      btn.addEventListener('click', function () {
        act.classList.toggle('is-open'); apply();
        if (!act.classList.contains('is-open')) act.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
      });
      apply();
      if (mq.addEventListener) mq.addEventListener('change', apply); else if (mq.addListener) mq.addListener(apply);
    });
  })();

  /* ------------------------------------------------------------------ */
  /* 4. Quick-links rail, share, BibTeX copy, reveal                       */
  /* ------------------------------------------------------------------ */
  (function rail() {
    var links = $$('.toc a, .jumpbar a'); if (!links.length) return;
    var map = {}; links.forEach(function (a) { var k = a.getAttribute('href').slice(1); (map[k] = map[k] || []).push(a); });
    if ('IntersectionObserver' in window) {
      var current = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { if (en.isIntersecting) current = en.target.id; });
        links.forEach(function (a) { a.classList.toggle('is-active', !!(map[current] && map[current].indexOf(a) !== -1)); if (a.classList.contains('is-active') && a.closest('.jumpbar') && a.scrollIntoView) { try { a.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }); } catch (e) {} } });
      }, { rootMargin: '-25% 0px -60% 0px', threshold: 0 });
      Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
    }
    var share = $('#shareBtn'), toast = $('#toast');
    function showToast(msg) { toast.textContent = msg; toast.classList.add('is-on'); clearTimeout(showToast.t); showToast.t = setTimeout(function () { toast.classList.remove('is-on'); }, 2200); }
    if (share) share.addEventListener('click', function () {
      var url = location.href.split('#')[0], title = document.title;
      if (navigator.share) { navigator.share({ title: title, url: url }).catch(function () {}); return; }
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(function () { showToast('Link copied'); }, function () { showToast(url); });
      else showToast(url);
    });
    window.__toast = showToast;
  })();

  (function bib() {
    var btn = $('#bibtexCopy'), pre = $('#bibtexText'); if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var txt = pre.textContent;
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy BibTeX'; }, 1800); if (window.__toast) window.__toast('BibTeX copied'); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () { selectText(pre); });
      else selectText(pre);
    });
    function selectText(node) { var r = document.createRange(); r.selectNodeContents(node); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }
  })();

  (function reveal() {
    var items = $$('.reveal'); if (!items.length) return;
    if (reduced || !('IntersectionObserver' in window) || /[?&]static\b/.test(location.search)) { items.forEach(function (i) { i.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    items.forEach(function (i) { io.observe(i); });
    function revealAll() { items.forEach(function (i) { i.classList.add('in'); }); }
    setTimeout(revealAll, 3500);
    window.addEventListener('beforeprint', revealAll);
  })();

  /* ------------------------------------------------------------------ */
  /* 5. Colab links                                                       */
  /* ------------------------------------------------------------------ */
  /* On GitHub Pages the notebook lives in the code repository, so a placeholder Colab URL can be derived from
     the host (USER.github.io) and the first path segment (REPO). Anywhere else the link falls back to the
     notebook file. configure.py writes the real link, after which this does nothing. */
  (function colab() {
    var links = $$('a.js-colab').filter(function (a) { return /USER\/REPO/.test(a.getAttribute('href') || ''); });
    if (!links.length) return;
    var host = location.hostname, m = /^([a-z0-9-]+)\.github\.io$/i.exec(host);
    var href = null, noteText = null;
    if (m) {
      var user = m[1], seg = location.pathname.split('/').filter(Boolean)[0];
      var repo = seg && !/\.html?$/i.test(seg) ? seg : user + '.github.io';
      href = 'https://colab.research.google.com/github/' + user + '/' + repo + '/blob/main/colab/vedje_tour.ipynb';
    } else {
      href = 'colab/vedje_tour.ipynb';
      noteText = 'This preview is not served from GitHub Pages, so the link downloads the notebook; open it in Colab with File, Upload notebook.';
    }
    links.forEach(function (a) {
      if (href) a.setAttribute('href', href);
      if (noteText) { a.setAttribute('title', noteText); a.removeAttribute('target'); }
    });
  })();
})();
