/* =========================================================================
 * ml.js — Pure-JavaScript Multiple Linear Regression & Feature Selection
 * -------------------------------------------------------------------------
 * Everything runs in the browser (no server), so the page can be hosted on
 * GitHub Pages and users can upload their own CSV files.
 *
 *  - CSV parsing + automatic type inference + one-hot encoding (drop-first)
 *  - OLS via normal equations on standardized features (Gram-matrix cache)
 *  - K-fold (optionally repeated) cross-validation or hold-out evaluation
 *  - Feature selection: Lasso ranking, Forward SFS, RFE, ElasticNet ranking,
 *    Exhaustive search
 *  - Statsmodels-style OLS summary (SE, t, p-values, adj. R², F-test)
 * ========================================================================= */
(function (global) {
  'use strict';

  /* ---------------------------------------------------------------- utils */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffledCopy(arr, rand) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  const range = (n) => Array.from({ length: n }, (_, i) => i);

  function nCk(n, k) {
    if (k < 0 || k > n) return 0;
    let r = 1;
    for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
    return Math.round(r);
  }

  /* ------------------------------------------------------------ CSV parse */
  function parseCSV(text) {
    text = String(text).replace(/^\uFEFF/, '');
    const firstLine = text.split(/\r?\n/)[0] || '';
    let delim = ',', best = -1;
    for (const c of [',', ';', '\t']) {
      const cnt = firstLine.split(c).length;
      if (cnt > best) { best = cnt; delim = c; }
    }
    const rows = [];
    let row = [], field = '', inQ = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQ) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false;
        } else field += ch;
      } else if (ch === '"') inQ = true;
      else if (ch === delim) { row.push(field); field = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = ''; rows.push(row); row = [];
      } else field += ch;
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    const clean = rows.filter((r) => r.some((v) => String(v).trim() !== ''));
    if (clean.length < 2) throw new Error('CSV must contain a header row and at least one data row.');
    const headers = clean[0].map((h, i) => String(h).trim() || `col${i + 1}`);
    const data = clean.slice(1).map((r) => headers.map((_, i) => String(r[i] ?? '').trim()));
    return { headers, rows: data };
  }

  const MISSING = new Set(['', 'na', 'nan', 'null', 'none', '?', 'n/a', '#n/a']);
  const isMissing = (v) => MISSING.has(String(v).trim().toLowerCase());
  const toNum = (v) => {
    const s = String(v).trim();
    if (s === '') return NaN;
    return Number(s);
  };

  function inferColumns(parsed) {
    const n = parsed.rows.length;
    return parsed.headers.map((name, index) => {
      let nNum = 0, nMiss = 0;
      const uniq = new Set();
      for (const r of parsed.rows) {
        const v = r[index];
        if (isMissing(v)) { nMiss++; continue; }
        if (Number.isFinite(toNum(v))) nNum++;
        uniq.add(v);
      }
      const nonMiss = n - nMiss;
      const type = nonMiss > 0 && nNum === nonMiss ? 'numeric' : 'categorical';
      return { name, index, type, missing: nMiss, unique: uniq.size };
    });
  }

  /* -------------------------------------------- design matrix preparation */
  function prepare(parsed, target, features, colInfo) {
    const info = colInfo || inferColumns(parsed);
    const byName = Object.fromEntries(info.map((c) => [c.name, c]));
    if (!byName[target]) throw new Error(`Target column "${target}" not found.`);
    if (byName[target].type !== 'numeric') throw new Error(`Target "${target}" must be numeric.`);
    features = features.filter((f) => f !== target && byName[f]);
    if (!features.length) throw new Error('Please select at least one feature column.');

    const used = [target, ...features];
    const keep = parsed.rows.filter((r) => used.every((n) => !isMissing(r[byName[n].index])));
    const notes = [];
    const dropped = parsed.rows.length - keep.length;
    if (dropped) notes.push({ en: `${dropped} row(s) with missing values were removed.`, zh: `已移除 ${dropped} 筆含缺失值的資料。` });

    let names = [], cols = [];
    for (const f of features) {
      const c = byName[f];
      if (c.type === 'numeric') {
        names.push(f);
        cols.push(keep.map((r) => toNum(r[c.index])));
      } else {
        const levels = [...new Set(keep.map((r) => r[c.index]))].sort();
        if (levels.length > 25) {
          notes.push({ en: `"${f}" has ${levels.length} categories — skipped (too many levels).`, zh: `「${f}」有 ${levels.length} 個類別，類別過多已略過。` });
          continue;
        }
        if (levels.length < 2) continue;
        notes.push({ en: `"${f}" one-hot encoded (baseline = ${levels[0]}).`, zh: `「${f}」已進行 One-Hot 編碼（基準類別 = ${levels[0]}）。` });
        for (const lv of levels.slice(1)) {
          names.push(`${f}_${lv}`);
          cols.push(keep.map((r) => (r[c.index] === lv ? 1 : 0)));
        }
      }
    }

    const y = keep.map((r) => toNum(r[byName[target].index]));
    const n = y.length;
    if (n < 6) throw new Error('Need at least 6 complete rows to run the analysis.');

    // statistics & constant-column removal
    const means = [], sds = [], fNames = [], fCols = [];
    cols.forEach((col, j) => {
      const m = col.reduce((a, b) => a + b, 0) / n;
      const v = col.reduce((a, b) => a + (b - m) * (b - m), 0) / n;
      if (v <= 1e-18) {
        notes.push({ en: `"${names[j]}" is constant and was removed.`, zh: `「${names[j]}」為常數欄位，已移除。` });
        return;
      }
      means.push(m); sds.push(Math.sqrt(v)); fNames.push(names[j]); fCols.push(col);
    });
    names = fNames; cols = fCols;
    const p = names.length;
    if (!p) throw new Error('No usable feature columns after encoding.');

    const X = range(n).map((i) => cols.map((c) => c[i]));
    const Z = range(n).map((i) => {
      const z = new Float64Array(p);
      for (let j = 0; j < p; j++) z[j] = (cols[j][i] - means[j]) / sds[j];
      return z;
    });
    return { names, X, Z, y, means, sds, n, p, notes, target, rawRows: keep, features };
  }

  /* ------------------------------------------------------- linear algebra */
  function solve(A, b) {
    const m = b.length;
    const M = A.map((r, i) => { const row = Array.from(r); row.push(b[i]); return row; });
    for (let c = 0; c < m; c++) {
      let piv = c;
      for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (piv !== c) { const t = M[c]; M[c] = M[piv]; M[piv] = t; }
      if (Math.abs(M[c][c]) < 1e-12) M[c][c] = M[c][c] >= 0 ? 1e-12 : -1e-12;
      for (let r = c + 1; r < m; r++) {
        const f = M[r][c] / M[c][c];
        if (!f) continue;
        for (let k = c; k <= m; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(m).fill(0);
    for (let r = m - 1; r >= 0; r--) {
      let s = M[r][m];
      for (let k = r + 1; k < m; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  /** Gram matrix of augmented rows a = [1, z_1..z_p, y] over the given rows. */
  function gramOf(Z, y, rows) {
    const p = Z[0].length, P = p + 2;
    const G = Array.from({ length: P }, () => new Float64Array(P));
    const a = new Float64Array(P);
    for (const i of rows) {
      a[0] = 1;
      const zi = Z[i];
      for (let j = 0; j < p; j++) a[j + 1] = zi[j];
      a[P - 1] = y[i];
      for (let u = 0; u < P; u++) {
        const au = a[u];
        if (au === 0) continue;
        const Gu = G[u];
        for (let v = u; v < P; v++) Gu[v] += au * a[v];
      }
    }
    for (let u = 0; u < P; u++) for (let v = 0; v < u; v++) G[u][v] = G[v][u];
    return G;
  }

  const RIDGE = 1e-9; // tiny jitter for numerical stability (collinear dummies)

  /** OLS on a feature subset from a Gram matrix. Returns [b0, b_1..b_k] (standardized space). */
  function olsFromGram(G, subset) {
    const P = G.length;
    const idx = [0, ...subset.map((j) => j + 1)];
    const m = idx.length;
    const A = Array.from({ length: m }, (_, a) => {
      const row = new Array(m);
      for (let b = 0; b < m; b++) row[b] = G[idx[a]][idx[b]] + (a > 0 && a === b ? RIDGE * G[0][0] : 0);
      return row;
    });
    const rhs = idx.map((u) => G[u][P - 1]);
    return { beta: solve(A, rhs), A };
  }

  /* ------------------------------------------------------------ evaluator */
  function kfoldSplits(rows, K, repeats, seed) {
    K = Math.max(2, Math.min(K, rows.length));
    const splits = [];
    for (let r = 0; r < repeats; r++) {
      const perm = shuffledCopy(rows, mulberry32(seed + 9973 * r));
      for (let f = 0; f < K; f++) {
        const test = [], train = [];
        perm.forEach((v, i) => (i % K === f ? test : train).push(v));
        splits.push({ train, test, repeat: r });
      }
    }
    return splits;
  }

  class Evaluator {
    constructor(Z, y, splits) {
      this.Z = Z; this.y = y; this.splits = splits;
      this.grams = splits.map((s) => gramOf(Z, y, s.train));
      this.cache = new Map();
      let nT = 0, sum = 0;
      for (const s of splits) for (const i of s.test) { nT++; sum += y[i]; }
      this.nTest = nT;
      this.ybar = sum / nT;
      let sst = 0;
      for (const s of splits) for (const i of s.test) sst += (y[i] - this.ybar) ** 2;
      this.sst = sst;
    }

    score(subset, useCache = true) {
      const key = subset.slice().sort((a, b) => a - b).join(',');
      if (useCache && this.cache.has(key)) return this.cache.get(key);
      let sse = 0, sae = 0;
      const Z = this.Z, y = this.y, k = subset.length;
      for (let s = 0; s < this.splits.length; s++) {
        const { beta } = olsFromGram(this.grams[s], subset);
        for (const i of this.splits[s].test) {
          let pred = beta[0];
          const zi = Z[i];
          for (let t = 0; t < k; t++) pred += beta[t + 1] * zi[subset[t]];
          const e = y[i] - pred;
          sse += e * e; sae += Math.abs(e);
        }
      }
      const res = { rmse: Math.sqrt(sse / this.nTest), mae: sae / this.nTest, r2: 1 - sse / this.sst };
      if (useCache) this.cache.set(key, res);
      return res;
    }

    /** Out-of-sample predictions (averaged over repeats) for each evaluated row. */
    predictions(subset) {
      const acc = new Map();
      const Z = this.Z, k = subset.length;
      for (let s = 0; s < this.splits.length; s++) {
        const { beta } = olsFromGram(this.grams[s], subset);
        for (const i of this.splits[s].test) {
          let pred = beta[0];
          for (let t = 0; t < k; t++) pred += beta[t + 1] * Z[i][subset[t]];
          const o = acc.get(i) || { sum: 0, cnt: 0 };
          o.sum += pred; o.cnt++;
          acc.set(i, o);
        }
      }
      return [...acc.entries()].map(([i, o]) => ({ i, y: this.y[i], pred: o.sum / o.cnt }));
    }
  }

  /* --------------------------------------------- Lasso / ElasticNet paths */
  /**
   * sklearn-style objective on standardized features:
   *   (1/2n)||y - Xb||² + α·ρ·||b||₁ + α·(1-ρ)/2·||b||²
   * Ranking = |coef| at α (= alphaFrac·α_max), ties broken by path entry order.
   */
  function penalizedRanking(Z, y, rows, l1Ratio, alphaFrac, nAlphas = 80) {
    const n = rows.length, p = Z[0].length;
    const cols = [], norm = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const c = new Float64Array(n);
      let m = 0;
      for (let t = 0; t < n; t++) { c[t] = Z[rows[t]][j]; m += c[t]; }
      m /= n;
      let s = 0;
      for (let t = 0; t < n; t++) { c[t] -= m; s += c[t] * c[t]; }
      cols.push(c); norm[j] = s / n;
    }
    let ym = 0;
    for (const i of rows) ym += y[i];
    ym /= n;
    const yc = new Float64Array(n);
    rows.forEach((i, t) => (yc[t] = y[i] - ym));

    const rho = Math.max(l1Ratio, 1e-3);
    let alphaMax = 0;
    for (let j = 0; j < p; j++) {
      let d = 0;
      for (let t = 0; t < n; t++) d += cols[j][t] * yc[t];
      alphaMax = Math.max(alphaMax, Math.abs(d) / (n * rho));
    }
    alphaMax = alphaMax || 1;
    const soft = (z, g) => (z > g ? z - g : z < -g ? z + g : 0);

    function cd(alpha, b, r) {
      const l1 = alpha * rho, l2 = alpha * (1 - rho);
      for (let it = 0; it < 2000; it++) {
        let maxDelta = 0, maxB = 0;
        for (let j = 0; j < p; j++) {
          const c = cols[j], old = b[j];
          let d = 0;
          for (let t = 0; t < n; t++) d += c[t] * r[t];
          const z = d / n + norm[j] * old;
          const nb = soft(z, l1) / (norm[j] + l2);
          if (nb !== old) {
            const diff = nb - old;
            for (let t = 0; t < n; t++) r[t] -= c[t] * diff;
            b[j] = nb;
            maxDelta = Math.max(maxDelta, Math.abs(diff));
          }
          maxB = Math.max(maxB, Math.abs(nb));
        }
        if (maxDelta <= 1e-9 * Math.max(maxB, 1e-12) || maxDelta < 1e-14) break;
      }
    }

    // regularization path → entry order
    const b = new Float64Array(p), r = Float64Array.from(yc);
    const entry = new Array(p).fill(Infinity);
    for (let s = 0; s < nAlphas; s++) {
      const alpha = alphaMax * Math.pow(10, (-4 * s) / (nAlphas - 1));
      cd(alpha, b, r);
      for (let j = 0; j < p; j++) if (b[j] !== 0 && entry[j] === Infinity) entry[j] = s;
    }
    const pathEnd = Array.from(b);

    // coefficients at the chosen α (warm start from the path)
    const alpha = alphaMax * alphaFrac;
    const b2 = new Float64Array(p), r2 = Float64Array.from(yc);
    for (let s = 0; s < 40; s++) {
      const a = alphaMax * Math.pow(alphaFrac, s / 39);
      cd(a, b2, r2);
    }
    cd(alpha, b2, r2);
    const coefs = Array.from(b2);

    const order = range(p).sort((i, j) =>
      Math.abs(coefs[j]) - Math.abs(coefs[i]) ||
      entry[i] - entry[j] ||
      Math.abs(pathEnd[j]) - Math.abs(pathEnd[i]));
    return { order, coefs, entry, alphaMax, alpha, nonZero: coefs.filter((c) => c !== 0).length };
  }

  /* ------------------------------------------------------------------ RFE */
  function rfe(Z, y, rows) {
    const p = Z[0].length;
    const G = gramOf(Z, y, rows);
    let remaining = range(p);
    const eliminated = [];
    while (remaining.length) {
      const { beta } = olsFromGram(G, remaining);
      let minT = 0;
      for (let t = 1; t < remaining.length; t++)
        if (Math.abs(beta[t + 1]) < Math.abs(beta[minT + 1])) minT = t;
      eliminated.push(remaining[minT]);
      remaining = remaining.filter((_, t) => t !== minT);
    }
    return { order: eliminated.reverse() };
  }

  /* ---------------------------------------------------------- Forward SFS */
  function forwardSFS(crit, p, kmax) {
    const selected = [];
    while (selected.length < kmax) {
      let best = null;
      for (let j = 0; j < p; j++) {
        if (selected.includes(j)) continue;
        const s = crit.score([...selected, j]);
        if (!best || s.rmse < best.rmse) best = { j, rmse: s.rmse };
      }
      selected.push(best.j);
    }
    return { order: selected };
  }

  /* ---------------------------------------------------- Exhaustive search */
  const EXHAUSTIVE_LIMIT = 250000;

  function exhaustive(crit, p, kmax) {
    const bestByK = [];
    let evaluated = 0;
    for (let k = 1; k <= kmax; k++) {
      const idx = range(k);
      let best = null;
      while (true) {
        const s = crit.score(idx, false);
        evaluated++;
        if (!best || s.rmse < best.rmse) best = { subset: idx.slice(), rmse: s.rmse };
        let i = k - 1;
        while (i >= 0 && idx[i] === p - k + i) i--;
        if (i < 0) break;
        idx[i]++;
        for (let t = i + 1; t < k; t++) idx[t] = idx[t - 1] + 1;
      }
      bestByK.push(best.subset);
    }
    return { bestByK, evaluated };
  }

  /* --------------------------------------------------------- main runner */
  const METHOD_META = [
    { id: 'lasso', name: 'Lasso (L1 Ranking)', color: '#38bdf8' },
    { id: 'sfs', name: 'Forward SFS', color: '#34d399' },
    { id: 'rfe', name: 'RFE', color: '#fb923c' },
    { id: 'enet', name: 'ElasticNet Ranking', color: '#f472b6' },
    { id: 'exh', name: 'Exhaustive Search', color: '#a78bfa' },
  ];

  function runAnalysis(D, o) {
    const t0 = performance.now();
    const { Z, y, n, p } = D;
    const kmax = Math.max(1, Math.min(o.kmax || p, p));
    const all = range(n);
    let evalE, critE, selRows, split = null;

    if (o.mode === 'holdout') {
      const perm = shuffledCopy(all, mulberry32(o.seed));
      const nTest = Math.max(2, Math.round(n * o.testSize));
      const test = perm.slice(0, nTest), train = perm.slice(nTest);
      evalE = new Evaluator(Z, y, [{ train, test, repeat: 0 }]);
      critE = new Evaluator(Z, y, kfoldSplits(train, Math.min(o.folds, train.length), 1, o.seed + 1));
      selRows = train;
      split = { train, test };
    } else {
      evalE = new Evaluator(Z, y, kfoldSplits(all, o.folds, o.repeats, o.seed));
      critE = evalE;
      selRows = all;
    }

    const lasso = penalizedRanking(Z, y, selRows, 1, o.alphaFrac);
    const enet = penalizedRanking(Z, y, selRows, o.l1Ratio, o.alphaFrac);
    const rfeR = rfe(Z, y, selRows);
    const sfs = forwardSFS(critE, p, kmax);

    let totalCombos = 0;
    for (let k = 1; k <= kmax; k++) totalCombos += nCk(p, k);
    const exhOK = totalCombos <= EXHAUSTIVE_LIMIT;
    const exh = exhOK ? exhaustive(critE, p, kmax) : null;

    const subsetsFor = {
      lasso: (k) => lasso.order.slice(0, k),
      sfs: (k) => sfs.order.slice(0, k),
      rfe: (k) => rfeR.order.slice(0, k),
      enet: (k) => enet.order.slice(0, k),
      exh: (k) => exh.bestByK[k - 1],
    };
    const extras = { lasso, sfs, rfe: rfeR, enet, exh };

    const methods = METHOD_META.filter((m) => m.id !== 'exh' || exhOK).map((m) => {
      const perK = [];
      for (let k = 1; k <= kmax; k++) {
        const subset = subsetsFor[m.id](k);
        perK.push({ k, subset, ...evalE.score(subset) });
      }
      return { ...m, perK, order: m.id === 'exh' ? null : extras[m.id].order, extra: extras[m.id] };
    });

    return {
      methods, kmax, names: D.names, n, p,
      baseline: evalE.score([]),
      evaluator: evalE, selRows, split,
      exhaustive: { ok: exhOK, combos: totalCombos, limit: EXHAUSTIVE_LIMIT },
      options: { ...o },
      ms: performance.now() - t0,
    };
  }

  /* ---------------------------------------------- statsmodels-like summary */
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let yy = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (const ci of c) ser += ci / ++yy;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }
  function betacf(a, b, x) {
    const FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-14) break;
    }
    return h;
  }
  function betai(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
  }
  const tPValue = (t, df) => (Number.isFinite(t) ? betai(df / 2, 0.5, df / (df + t * t)) : NaN);
  const fPValue = (F, d1, d2) => (Number.isFinite(F) ? betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)) : NaN);

  function olsSummary(D, rows, subset) {
    const { Z, y, means, sds, names } = D;
    const G = gramOf(Z, y, rows);
    const { beta, A } = olsFromGram(G, subset);
    const m = beta.length, n = rows.length, k = subset.length;
    let sse = 0, ym = 0;
    for (const i of rows) ym += y[i];
    ym /= n;
    let sst = 0;
    for (const i of rows) {
      let pred = beta[0];
      for (let t = 0; t < k; t++) pred += beta[t + 1] * Z[i][subset[t]];
      sse += (y[i] - pred) ** 2;
      sst += (y[i] - ym) ** 2;
    }
    const df = n - k - 1;
    const sigma2 = df > 0 ? sse / df : NaN;
    const inv = range(m).map((c) => solve(A, range(m).map((r) => (r === c ? 1 : 0))));
    const cov = (a, b) => sigma2 * inv[b][a];

    const terms = [];
    // intercept in original units: b0 - Σ b_j m_j / s_j
    const cvec = [1, ...subset.map((j) => -means[j] / sds[j])];
    let b0 = 0, v0 = 0;
    for (let a = 0; a < m; a++) {
      b0 += cvec[a] * beta[a];
      for (let b = 0; b < m; b++) v0 += cvec[a] * cvec[b] * cov(a, b);
    }
    const mk = (name, coef, se, std) => {
      const t = coef / se;
      return { name, coef, se, t, p: df > 0 ? tPValue(t, df) : NaN, std };
    };
    terms.push(mk('Intercept', b0, Math.sqrt(v0), null));
    subset.forEach((j, t) => {
      terms.push(mk(names[j], beta[t + 1] / sds[j], Math.sqrt(cov(t + 1, t + 1)) / sds[j], beta[t + 1]));
    });
    const r2 = 1 - sse / sst;
    const F = k > 0 && df > 0 ? ((sst - sse) / k) / (sse / df) : NaN;
    return {
      terms, r2, adjR2: df > 0 ? 1 - ((1 - r2) * (n - 1)) / df : NaN,
      rmse: Math.sqrt(sse / n), F, Fp: k > 0 && df > 0 ? fPValue(F, k, df) : NaN,
      n, k, df,
      aic: n * Math.log(sse / n) + 2 * (k + 1),
      bic: n * Math.log(sse / n) + Math.log(n) * (k + 1),
    };
  }

  function correlation(cols) {
    const m = cols.length, n = cols[0].length;
    const st = cols.map((c) => {
      const mu = c.reduce((a, b) => a + b, 0) / n;
      const sd = Math.sqrt(c.reduce((a, b) => a + (b - mu) ** 2, 0)) || 1;
      return c.map((v) => (v - mu) / sd);
    });
    return range(m).map((a) => range(m).map((b) => {
      let s = 0;
      for (let i = 0; i < n; i++) s += st[a][i] * st[b][i];
      return s;
    }));
  }

  global.ML = {
    parseCSV, inferColumns, prepare, runAnalysis, olsSummary, correlation,
    METHOD_META, nCk, EXHAUSTIVE_LIMIT,
  };
})(window);
