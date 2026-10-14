/* =========================================================================
 * app.js — UI layer: dataset switching, upload, charts, insights, explorer
 * ========================================================================= */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);

  /* ------------------------------------------------------ built-in data */
  const BUILTIN = {
    startups: {
      csv: window.BUILTIN_CSV ? window.BUILTIN_CSV.startups : '',
      target: 'Profit',
      desc: {
        en: '50 US startups — predict <b>Profit</b> from R&D Spend, Administration, Marketing Spend and State. State is one-hot encoded (baseline California), giving <b>5 candidate features</b>.',
        zh: '50 家新創公司資料 — 以研發支出、管理費用、行銷支出與所在州預測 <b>Profit（利潤）</b>。State 經 One-Hot 編碼（基準為 California），共 <b>5 個候選特徵</b>。',
      },
    },
    boston: {
      csv: window.BUILTIN_CSV ? window.BUILTIN_CSV.boston : '',
      target: 'medv',
      desc: {
        en: 'Boston Housing (506 tracts) — predict <b>medv</b> (median home value, $1000s) from 13 features: crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat.',
        zh: '波士頓房價資料（506 筆）— 以 13 個特徵（犯罪率、住宅用地比例、非零售商業比例、河岸、NOx、房間數、屋齡、距就業中心距離、公路可達性、稅率、師生比、b、低收入比例）預測 <b>medv（房價中位數，千美元）</b>。',
      },
    },
  };

  const state = {
    key: 'startups',
    cfg: {},          // per-dataset { parsed, colInfo, target, selected:Set, kmax }
    D: null, R: null,
    lang: 'zh',
    charts: {},
    methodFilter: 'all',
  };

  /* ----------------------------------------------------- chart defaults */
  if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', 'Noto Sans TC', sans-serif";
    Chart.defaults.borderColor = 'rgba(148,163,184,0.10)';
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(10,15,32,0.95)';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(167,139,250,0.4)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.titleFont = { weight: '700' };
  }

  /* ------------------------------------------------------------ helpers */
  const fmt = (v) => {
    if (!Number.isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(3);
    if (a === 0) return '0';
    return v.toPrecision(3);
  };
  const fmtR2 = (v) => (Number.isFinite(v) ? v.toFixed(4) : '—');
  const fmtP = (p) => (!Number.isFinite(p) ? '—' : p < 1e-4 ? '<0.0001' : p.toFixed(4));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pct = (a, b) => ((a / b - 1) * 100);

  function toast(msg, ok = false) {
    const t = $('toast');
    if (!t) return;
    t.textContent = msg;
    t.className = 'toast' + (ok ? ' ok' : '');
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.add('hidden'), 4200);
  }

  function makeChart(id, config) {
    if (typeof Chart === 'undefined') return null;
    const canvas = $(id);
    if (!canvas) return null;
    if (state.charts[id]) state.charts[id].destroy();
    state.charts[id] = new Chart(canvas, config);
    return state.charts[id];
  }

  /* ----------------------------------------------------- dataset config */
  function initConfig(key, parsed, target) {
    const colInfo = ML.inferColumns(parsed);
    const n = parsed.rows.length;
    const numeric = colInfo.filter((c) => c.type === 'numeric');
    if (!target || !colInfo.some((c) => c.name === target && c.type === 'numeric')) {
      target = numeric.length ? numeric[numeric.length - 1].name : null;
    }
    const idLike = (c) => /^(id|index|unnamed.*|no\.?|row|#)$/i.test(c.name) ||
      (c.type === 'categorical' && c.unique >= n * 0.9);
    const selected = new Set(colInfo.filter((c) => c.name !== target && !idLike(c)).map((c) => c.name));
    state.cfg[key] = { parsed, colInfo, target, selected, kmax: null };
  }

  function loadDataset(key) {
    state.key = key;
    $('uploadCard').classList.toggle('hidden', key !== 'custom');
    if (key !== 'custom' && !state.cfg[key]) {
      initConfig(key, ML.parseCSV(BUILTIN[key].csv), BUILTIN[key].target);
    }
    const cfg = state.cfg[key];
    renderVariables();
    if (cfg) run();
    else {
      clearResults();
      $('datasetDesc').innerHTML = state.lang === 'zh'
        ? '請上傳 CSV 檔案，選擇目標欄位（數值）與候選特徵後即可自動分析。'
        : 'Upload a CSV file, then pick a numeric target and candidate features — the analysis runs automatically.';
    }
  }

  function clearResults() {
    Object.values(state.charts).forEach((c) => c && c.destroy());
    state.charts = {};
    ['kpis', 'insights', 'mapTable', 'rankings', 'equation', 'olsTable', 'olsStats',
      'heatmap', 'previewTable', 'resultsTable', 'prepNotes', 'targetSelect', 'featureChecks']
      .forEach((id) => { const el = $(id); if (el) el.innerHTML = ''; });
    state.R = null;
  }

  /* --------------------------------------------------------- variables */
  function renderVariables() {
    const cfg = state.cfg[state.key];
    const desc = BUILTIN[state.key]?.desc;
    $('datasetDesc').innerHTML = desc ? desc[state.lang] : (cfg
      ? `${cfg.parsed.rows.length} rows × ${cfg.parsed.headers.length} columns · ${esc(cfg.fileName || 'custom.csv')}`
      : '');
    if (!cfg) return;
    $('targetSelect').innerHTML = cfg.colInfo.filter((c) => c.type === 'numeric')
      .map((c) => `<option ${c.name === cfg.target ? 'selected' : ''}>${esc(c.name)}</option>`).join('');
    $('featureChecks').innerHTML = cfg.colInfo.filter((c) => c.name !== cfg.target).map((c) => `
      <label class="chip-check">
        <input type="checkbox" value="${esc(c.name)}" ${cfg.selected.has(c.name) ? 'checked' : ''}/>
        <span>${esc(c.name)} <em>${c.type === 'numeric' ? 'num' : `cat·${c.unique}`}</em></span>
      </label>`).join('');
  }

  /* -------------------------------------------------------------- run */
  function getOptions() {
    return {
      mode: $('modeSelect').value,
      folds: Math.max(2, +$('foldsInput').value || 5),
      repeats: Math.max(1, Math.min(20, +$('repeatsInput').value || 1)),
      testSize: +$('testSizeInput').value / 100,
      seed: +$('seedInput').value || 0,
      alphaFrac: +$('alphaInput').value / 100,
      l1Ratio: +$('l1Input').value / 100,
    };
  }

  let runTimer = null;
  function scheduleRun(delay = 300) {
    clearTimeout(runTimer);
    runTimer = setTimeout(run, delay);
  }

  function defaultKmax(p) {
    let k = p, total = 0;
    for (let i = 1; i <= p; i++) {
      total += ML.nCk(p, i);
      if (total > ML.EXHAUSTIVE_LIMIT) { k = i - 1; break; }
    }
    return Math.max(1, Math.min(p, Math.max(k, Math.min(p, 10))));
  }

  function run() {
    const cfg = state.cfg[state.key];
    if (!cfg) return;
    $('loader').classList.remove('hidden');
    setTimeout(() => {
      try {
        const D = ML.prepare(cfg.parsed, cfg.target, [...cfg.selected], cfg.colInfo);
        if (!cfg.kmax || cfg.kmax > D.p || cfg.p !== D.p) cfg.kmax = defaultKmax(D.p);
        cfg.p = D.p;
        const ks = $('kmaxInput');
        ks.max = D.p; ks.value = cfg.kmax;
        $('kmaxVal').textContent = `${cfg.kmax} / ${D.p}`;

        let R;
        const opt = getOptions();
        const isDefaultSetup = (opt.mode === 'cv' && opt.folds === 5 && opt.seed === 42 && opt.repeats === 1);
        if (state.key !== 'custom' && window.PRECOMPUTED && window.PRECOMPUTED[state.key] && isDefaultSetup && cfg.selected.size >= cfg.colInfo.filter((c) => c.name !== cfg.target).length) {
          const py = window.PRECOMPUTED[state.key];
          const all = Array.from({ length: D.n }, (_, i) => i);
          const evalE = new ML.Evaluator(D.Z, D.y, ML.kfoldSplits(all, 5, 1, 42));
          const methods = py.methods.map((m) => {
            const perK = m.per_k.slice(0, cfg.kmax).map((pk) => ({
              k: pk.k,
              subset: pk.subset_indices,
              rmse: pk.rmse,
              mae: pk.mae,
              r2: pk.r2
            }));
            const order = m.feature_ranking ? m.feature_ranking.map((name) => D.names.indexOf(name)).filter((i) => i >= 0) : null;
            const coefs = D.names.map((name) => (m.id === 'lasso' ? (py.lasso_coefficients[name] || 0) : (py.enet_coefficients[name] || 0)));
            return {
              id: m.id,
              name: m.name,
              color: m.color,
              perK,
              order,
              extra: { alpha: 0.05, coefs }
            };
          });
          R = {
            methods,
            kmax: Math.min(cfg.kmax, py.kmax),
            names: D.names,
            n: D.n,
            p: D.p,
            baseline: { rmse: py.baseline_rmse },
            evaluator: evalE,
            selRows: all,
            split: null,
            exhaustive: { ok: true, combos: 31, limit: ML.EXHAUSTIVE_LIMIT },
            options: { ...opt },
            ms: 0,
            engine: 'Python (scikit-learn)'
          };
        } else {
          R = ML.runAnalysis(D, { ...opt, kmax: cfg.kmax });
          R.engine = 'In-browser JS Engine';
        }

        state.D = D; state.R = R;
        renderAll();
        $('timing').textContent = `${R.engine} · ${R.ms.toFixed(0)} ms`;
      } catch (e) {
        console.error(e);
        toast('⚠ ' + e.message);
      } finally {
        $('loader').classList.add('hidden');
      }
    }, 30);
  }

  function renderAll() {
    renderNotes();
    renderChartNote();
    renderKPIs();
    renderMainCharts();
    renderInsights();
    renderMap();
    renderRankings();
    renderPenaltyChart();
    setupExplorer();
    renderHeatmap();
    renderPreview();
    renderResultsTable();
  }

  function renderNotes() {
    const { D, R } = state;
    if (!D || !R) return;
    const notes = [...D.notes];
    if (R.exhaustive && !R.exhaustive.ok) notes.push({
      en: `Exhaustive search skipped: ${R.exhaustive.combos.toLocaleString()} subsets exceed the limit of ${R.exhaustive.limit.toLocaleString()}. Lower "Max features" to enable it.`,
      zh: `窮舉搜尋需評估 ${R.exhaustive.combos.toLocaleString()} 種組合，超過上限 ${R.exhaustive.limit.toLocaleString()}，已略過。請調低「Max features」。`,
    });
    $('prepNotes').innerHTML = notes.map((n) => `<li>${esc(n[state.lang])}</li>`).join('');
  }

  /* ------------------------------------------------------------- stats */
  function summarize() {
    const { R } = state;
    if (!R || !R.methods) return { best: null, bestPerK: [], kPars: 1 };
    let best = null;
    const bestPerK = [];
    for (let k = 1; k <= R.kmax; k++) {
      let bk = null;
      for (const m of R.methods) {
        if (!m.perK || !m.perK[k - 1]) continue;
        const r = m.perK[k - 1];
        if (!bk || r.rmse < bk.rmse) bk = { ...r, method: m };
        if (!best || r.rmse < best.rmse - 1e-12) best = { ...r, method: m };
      }
      if (bk) bestPerK.push(bk);
    }
    const kParsItem = best && bestPerK.find((b) => b && b.rmse <= best.rmse * 1.01);
    const kPars = kParsItem ? kParsItem.k : (best ? best.k : 1);
    return { best, bestPerK, kPars };
  }

  function renderKPIs() {
    const { D, R } = state;
    if (!D || !R) return;
    const { best, bestPerK, kPars } = summarize();
    const last = bestPerK.length ? bestPerK[bestPerK.length - 1] : null;
    const evalTxt = R.options.mode === 'cv'
      ? `${R.options.folds}-fold CV × ${R.options.repeats}`
      : `hold-out ${Math.round(R.options.testSize * 100)}% test`;

    const bMethod = best && best.method ? best.method.name : '—';
    const bColor = best && best.method ? best.method.color : '#38bdf8';
    const bRmse = best ? fmt(best.rmse) : '—';
    const bR2 = best ? fmtR2(best.r2) : '—';
    const bMae = best ? fmt(best.mae) : '—';
    const bK = best ? best.k : '—';

    const lastRmse = last ? fmt(last.rmse) : '—';
    const lastR2 = last ? fmtR2(last.r2) : '—';
    const lastSub = (last && best && best.rmse > 0)
      ? `R² ${lastR2} · ${pct(last.rmse, best.rmse) >= 0 ? '+' : ''}${pct(last.rmse, best.rmse).toFixed(1)}% vs best`
      : `R² ${lastR2}`;

    const cards = [
      { label: 'Dataset', value: `${D.n} × ${D.p}`, sub: `target: ${esc(D.target)} · ${evalTxt}`, c1: '#64748b', c2: '#94a3b8' },
      { label: 'Best RMSE', value: bRmse, sub: `${bMethod} · k = ${bK}`, c1: bColor, c2: '#38bdf8' },
      { label: 'Best R²', value: bR2, sub: `MAE ${bMae}`, c1: '#34d399', c2: '#38bdf8' },
      { label: 'Recommended k', value: kPars, sub: 'fewest features within 1% of best RMSE', c1: '#a78bfa', c2: '#f472b6' },
      {
        label: R.kmax === D.p ? 'All features (k = p)' : `Largest k = ${R.kmax}`,
        value: lastRmse,
        sub: lastSub,
        c1: '#fb923c', c2: '#f472b6',
      },
      { label: 'Baseline (mean)', value: fmt(R.baseline ? R.baseline.rmse : 0), sub: 'RMSE of predicting ȳ only', c1: '#475569', c2: '#64748b' },
    ];
    $('kpis').innerHTML = cards.map((c, i) => `
      <div class="kpi" style="--c1:${c.c1};--c2:${c.c2};animation-delay:${i * 60}ms">
        <div class="label">${c.label}</div>
        <div class="value">${c.value}</div>
        <div class="sub">${c.sub}</div>
      </div>`).join('');
  }

  /* ------------------------------------------------------- main charts */
  const METHOD_STYLES = {
    exh:   { dash: [],            width: 3.8, radius: 8.0, hoverRadius: 10.0, style: 'star' },
    sfs:   { dash: [8, 4],        width: 2.6, radius: 6.2, hoverRadius: 8.5,  style: 'rectRot' },
    lasso: { dash: [],            width: 2.2, radius: 5.0, hoverRadius: 7.5,  style: 'circle' },
    rfe:   { dash: [3, 3],        width: 2.2, radius: 4.0, hoverRadius: 6.5,  style: 'triangle' },
    enet:  { dash: [10, 3, 2, 3], width: 2.2, radius: 3.2, hoverRadius: 5.5,  style: 'rect' },
  };

  function lineConfig(metric, yTitle) {
    const { R } = state;
    const labels = Array.from({ length: R.kmax }, (_, i) => i + 1);
    return {
      type: 'line',
      data: {
        labels,
        datasets: R.methods.map((m) => {
          const st = METHOD_STYLES[m.id] || { dash: [], width: 2.2, radius: 5, style: 'circle' };
          const isHidden = (state.methodFilter && state.methodFilter !== 'all' && state.methodFilter !== m.id);
          return {
            label: m.name,
            data: m.perK.map((r) => r[metric]),
            borderColor: m.color,
            backgroundColor: m.color,
            borderWidth: st.width,
            borderDash: st.dash,
            pointRadius: st.radius,
            pointHoverRadius: st.hoverRadius,
            pointStyle: st.style,
            tension: 0.25,
            hidden: isHidden,
            order: m.id === 'exh' ? 0 : (m.id === 'sfs' ? 1 : 2),
          };
        }),
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        animation: { duration: 600, easing: 'easeOutQuart' },
        scales: {
          x: { title: { display: true, text: 'Number of Features', font: { weight: '600' } }, grid: { display: false } },
          y: { title: { display: true, text: yTitle, font: { weight: '600' } }, ticks: { callback: (v) => (metric === 'r2' ? v.toFixed(3) : fmt(v)) } },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 10,
              boxWidth: 10,
              font: { size: 11.5, weight: '500' }
            }
          },
          tooltip: {
            callbacks: {
              title: (items) => `k = ${items[0].label} feature(s)`,
              label: (ctx) => ` ${ctx.dataset.label}: ${metric === 'r2' ? fmtR2(ctx.parsed.y) : fmt(ctx.parsed.y)}`,
              afterLabel: (ctx) => {
                const m = R.methods[ctx.datasetIndex];
                if (!m || !m.perK[ctx.dataIndex] || !m.perK[ctx.dataIndex].subset) return '';
                return '   ↳ ' + m.perK[ctx.dataIndex].subset.map((j) => R.names[j] || '').join(', ');
              },
            },
          },
        },
      },
    };
  }

  function setMethodFilter(methodId) {
    state.methodFilter = methodId;
    document.querySelectorAll('#methodFilterBar .filter-chip').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.method === methodId);
    });
    ['rmseChart', 'r2Chart'].forEach((id) => {
      const chart = state.charts[id];
      if (!chart || !chart.data || !chart.data.datasets) return;
      chart.data.datasets.forEach((ds, idx) => {
        const m = state.R?.methods[idx];
        if (!m) return;
        ds.hidden = (methodId !== 'all' && m.id !== methodId);
      });
      chart.update();
    });
  }

  function renderChartNote() {
    const el = $('chartNoteBanner');
    if (!el) return;
    if (state.key === 'startups') {
      el.innerHTML = state.lang === 'zh'
        ? `<b>💡 為什麼部分折線完全重疊？(Why do curves overlap?)</b><br />
          多元線性迴歸中，模型評估指標是由所選特徵子集建立的 OLS 迴歸決定。當多種特徵選取方法<b>選出完全相同的特徵組合</b>時（例如 50 Startups 在 <code>k=1</code> 皆選 <code>R&D Spend</code>、<code>k=2</code> 皆選 <code>R&D + Marketing Spend</code>），其 CV RMSE 與 R² <b>在數學上必定 100% 相同</b>！<br />
          兩組方法於 <code>k=3</code> 分歧：<b>Exhaustive & SFS</b> 找出 <code>State_Florida</code>（RMSE 9,381 較優），而 <b>Lasso / RFE / ElasticNet</b> 納入 <code>Administration</code>（RMSE 9,538 較差）。點擊上方按鈕可單獨切換各方法檢視！`
        : `<b>💡 Why do some curves overlap?</b><br />
          In Multiple Linear Regression, evaluation metrics (RMSE & R²) are calculated by fitting an OLS model on the selected feature subset. When multiple algorithms <b>select the exact same subset of features</b> (e.g. at <code>k=1</code> all 5 pick <code>R&D Spend</code>; at <code>k=2</code> all 5 pick <code>R&D Spend + Marketing Spend</code>), their cross-validation RMSE & R² <b>are mathematically identical</b>!<br />
          They diverge at <code>k=3</code>: <b>Exhaustive & SFS</b> discover <code>State_Florida</code> (better RMSE: 9,381), while <b>Lasso, RFE & ElasticNet</b> include <code>Administration</code> (worse RMSE: 9,538). Use the buttons above to isolate any individual method!`;
    } else if (state.key === 'boston') {
      el.innerHTML = state.lang === 'zh'
        ? `<b>💡 為什麼波士頓房價在前 4 個特徵數完全重疊？</b><br />
          在波士頓房價資料集中，所有 5 種方法在 <code>k=1~4</code> <b>完全一致地</b>依序選中了最強的 4 個特徵：<code>lstat</code> → <code>rm</code> → <code>ptratio</code> → <code>dis</code>！因此 <code>k=1~4</code> 的 RMSE 完全相同。<br />
          於 <code>k=5</code> 產生分歧：<b>Forward SFS、RFE、ElasticNet 與 Exhaustive</b> 皆選入 <code>nox</code>（RMSE 5.034 較優），而 <b>Lasso</b> 選入 <code>b</code>（RMSE 5.116 較差）。點擊上方按鈕可單獨查看！`
        : `<b>💡 Why do Boston Housing curves overlap at k = 1 to 4?</b><br />
          Across Boston Housing, all 5 methods <b>independently and unanimously agree</b> on the top 4 strongest predictors: <code>lstat</code> → <code>rm</code> → <code>ptratio</code> → <code>dis</code>! Because identical subsets produce identical OLS models, their RMSE & R² match perfectly for <code>k=1..4</code>.<br />
          At <code>k=5</code> they diverge: <b>Forward SFS, RFE, ElasticNet & Exhaustive</b> select <code>nox</code> (better RMSE: 5.034), whereas <b>Lasso</b> selects <code>b</code> (RMSE: 5.116). Use the buttons above to view any method!`;
    } else {
      el.innerHTML = state.lang === 'zh'
        ? `<b>💡 特徵子集相同即產生相同指標：</b>在同一資料集上，若不同選取演算法挑出相同的特徵組合，則配適之 OLS 迴歸的 RMSE 與 R² 必定相同。下方特徵地圖（Feature Selection Map）標註了每種方法在各特徵數所挑選的實際欄位。`
        : `<b>💡 Identical subsets produce identical metrics:</b> In linear regression, if different algorithms select the same feature subset, their OLS cross-validation metrics are identical. The Feature Selection Map below displays the exact columns selected at each step.`;
    }
  }

  function renderMainCharts() {
    makeChart('rmseChart', lineConfig('rmse', 'RMSE (Lower is Better)'));
    makeChart('r2Chart', lineConfig('r2', 'R-squared (Higher is Better)'));
  }

  /* ---------------------------------------------------------- insights */
  function renderInsights() {
    const { R, D } = state;
    if (!R || !D) return;
    const { best, bestPerK, kPars } = summarize();
    if (!best) return;
    const N = (s) => (s || []).map((j) => R.names[j] || '');
    const zh = [], en = [];
    const exh = R.methods.find((m) => m.id === 'exh');
    const bestSubset = N(best.subset).map((x) => `<code>${esc(x)}</code>`).join(', ');

    const bestMethodName = best.method ? best.method.name : 'Model';
    zh.push(['🏆', `整體最佳：<b>${bestMethodName}</b> 在 <b>k = ${best.k}</b> 時 RMSE 最低（<b>${fmt(best.rmse)}</b>），R² = <b>${fmtR2(best.r2)}</b>，使用特徵：${bestSubset}。`]);
    en.push(['🏆', `Overall best: <b>${bestMethodName}</b> with <b>k = ${best.k}</b> — lowest RMSE <b>${fmt(best.rmse)}</b>, R² = <b>${fmtR2(best.r2)}</b>, features: ${bestSubset}.`]);

    const parsObj = bestPerK[kPars - 1] || best;
    zh.push(['🎯', `最佳特徵數量：在最佳 RMSE 1% 以內的最少特徵數為 <b>k = ${kPars}</b>（RMSE ${fmt(parsObj.rmse)}，R² ${fmtR2(parsObj.r2)}），模型既精簡又具備良好預測力。`]);
    en.push(['🎯', `Optimal feature count: the most parsimonious model within 1% of the best RMSE uses <b>k = ${kPars}</b> (RMSE ${fmt(parsObj.rmse)}, R² ${fmtR2(parsObj.r2)}).`]);

    const last = bestPerK[R.kmax - 1];
    if (last && R.kmax === D.p && best.k < D.p) {
      const inc = pct(last.rmse, best.rmse);
      zh.push(['📈', `使用全部 ${D.p} 個特徵時 RMSE 上升至 <b>${fmt(last.rmse)}</b>（比最佳高 ${inc.toFixed(1)}%），R² 降至 <b>${fmtR2(last.r2)}</b>，顯示納入過多特徵會引入雜訊或造成過度擬合。`]);
      en.push(['📈', `With all ${D.p} features RMSE rises to <b>${fmt(last.rmse)}</b> (+${inc.toFixed(1)}% vs best) and R² drops to <b>${fmtR2(last.r2)}</b> — extra features add noise / overfitting.`]);
    } else if (R.kmax === D.p) {
      zh.push(['📈', `此資料集中使用全部 ${D.p} 個特徵即為最佳，表示每個特徵都提供了有用的資訊。`]);
      en.push(['📈', `Here the full model with all ${D.p} features is best — every feature carries useful signal.`]);
    }

    if (exh) {
      const agree = R.methods.filter((m) => m.id !== 'exh').map((m) => {
        const same = m.perK.filter((r, i) => exh.perK[i] && r.rmse <= exh.perK[i].rmse * (1 + 1e-9)).length;
        return { m, same };
      });
      const lst = agree.map((a) => `${a.m.name} ${a.same}/${R.kmax}`).join('、');
      zh.push(['🟣', `Exhaustive Search 遍歷所有組合，於每個 k 皆找到最優子集。其他方法達到最優解的次數：${lst}。`]);
      en.push(['🟣', `Exhaustive Search finds the optimal subset at every k. Times each method matched it: ${agree.map((a) => `${a.m.name} ${a.same}/${R.kmax}`).join(', ')}.`]);

      const gaps = agree.map(({ m }) => {
        let worst = { gap: 0, k: 0 };
        m.perK.forEach((r, i) => {
          if (exh.perK[i]) {
            const g = pct(r.rmse, exh.perK[i].rmse);
            if (g > worst.gap) worst = { gap: g, k: i + 1 };
          }
        });
        return { m, ...worst };
      }).filter((g) => g.gap > 0.5).sort((a, b) => b.gap - a.gap);
      if (gaps.length) {
        zh.push(['⚠️', `貪婪或懲罰項方法可能較早納入次優特徵：` + gaps.map((g) => `<b>${g.m.name}</b> 在 k = ${g.k} 時比最優解差 ${g.gap.toFixed(1)}%`).join('；') + '。']);
        en.push(['⚠️', `Greedy / penalty-based methods can pick sub-optimal features early: ` + gaps.map((g) => `<b>${g.m.name}</b> is ${g.gap.toFixed(1)}% worse than optimal at k = ${g.k}`).join('; ') + '.']);
      } else {
        zh.push(['✅', '所有方法在每個特徵數下都與窮舉搜尋結果幾乎一致，特徵重要性結構相當明確。']);
        en.push(['✅', 'All methods essentially match the exhaustive optimum at every k — the feature-importance structure is very clear.']);
      }
    }

    // most important single feature
    const first = {};
    R.methods.forEach((m) => {
      if (m.perK && m.perK[0] && m.perK[0].subset && m.perK[0].subset.length) {
        const f = R.names[m.perK[0].subset[0]];
        if (f) first[f] = (first[f] || 0) + 1;
      }
    });
    const entries = Object.entries(first).sort((a, b) => b[1] - a[1]);
    if (entries.length) {
      const [topF, topC] = entries[0];
      const bk0 = bestPerK[0] || best;
      zh.push(['🔑', `最關鍵的單一特徵：<code>${esc(topF)}</code>（${topC}/${R.methods.length} 種方法在 k = 1 時都選擇它，RMSE ${fmt(bk0.rmse)}，R² ${fmtR2(bk0.r2)}）。`]);
      en.push(['🔑', `Most important single feature: <code>${esc(topF)}</code> (chosen at k = 1 by ${topC}/${R.methods.length} methods; RMSE ${fmt(bk0.rmse)}, R² ${fmtR2(bk0.r2)}).`]);
    }

    if (R.baseline && R.baseline.rmse) {
      zh.push(['📊', `相較於只預測平均值的基準模型（RMSE ${fmt(R.baseline.rmse)}），最佳模型的誤差降低了 <b>${(100 - (best.rmse / R.baseline.rmse) * 100).toFixed(1)}%</b>。`]);
      en.push(['📊', `Compared with the mean-only baseline (RMSE ${fmt(R.baseline.rmse)}), the best model cuts error by <b>${(100 - (best.rmse / R.baseline.rmse) * 100).toFixed(1)}%</b>.`]);
    }

    const list = state.lang === 'zh' ? zh : en;
    $('insights').innerHTML = '<ul>' + list.map(([i, t]) => `<li data-icon="${i}">${t}</li>`).join('') + '</ul>';
  }

  /* --------------------------------------------------------------- map */
  function renderMap() {
    const { R } = state;
    if (!R) return;
    let html = '<thead><tr><th>Method</th>' +
      Array.from({ length: R.kmax }, (_, i) => `<th>k = ${i + 1}</th>`).join('') + '</tr></thead><tbody>';
    const colStats = Array.from({ length: R.kmax }, (_, i) => {
      const v = R.methods.map((m) => (m.perK && m.perK[i] ? m.perK[i].rmse : 0));
      return { min: Math.min(...v), max: Math.max(...v) };
    });
    for (const m of R.methods) {
      html += `<tr><td class="mname"><i style="background:${m.color}"></i>${m.name}</td>`;
      m.perK.forEach((r, i) => {
        const { min, max } = colStats[i];
        const t = max - min > 1e-12 ? (r.rmse - min) / (max - min) : 0;
        const bg = t < 0.5 ? `rgba(52,211,153,${0.22 * (1 - 2 * t)})` : `rgba(244,114,182,${0.22 * (2 * t - 1)})`;
        const prev = (i && m.perK[i - 1] && m.perK[i - 1].subset) ? new Set(m.perK[i - 1].subset) : new Set();
        const tags = (r && r.subset ? r.subset : []).map((j) => `<span class="tag ${prev.has(j) ? '' : 'new'}">${esc(R.names[j] || '')}</span>`).join('');
        const star = r.rmse <= min + 1e-9 ? '<span class="star">★</span> ' : '';
        html += `<td class="map-cell" style="background:${bg}" title="RMSE ${fmt(r.rmse)} · R² ${fmtR2(r.r2)}">${tags}
          <div class="metric">${star}RMSE ${fmt(r.rmse)} · R² ${fmtR2(r.r2)}</div></td>`;
      });
      html += '</tr>';
    }
    $('mapTable').innerHTML = html + '</tbody>';
  }

  /* ---------------------------------------------------------- rankings */
  function renderRankings() {
    const { R } = state;
    if (!R) return;
    $('rankings').innerHTML = R.methods.map((m) => {
      let order = m.order;
      let note = '';
      if (!order) {
        order = [];
        m.perK.forEach((r) => r && r.subset && r.subset.forEach((j) => { if (!order.includes(j)) order.push(j); }));
        note = ' <span class="muted small">(first appearance)</span>';
      }
      return `<div class="rank-row"><div class="rname" style="color:${m.color}">${m.name}${note}</div>
        <div class="rank-list">${order.map((j, i) => `<span class="tag"><b>${i + 1}</b>${esc(R.names[j] || '')}</span>`).join('')}</div></div>`;
    }).join('');
  }

  function renderPenaltyChart() {
    const { R } = state;
    if (!R) return;
    const lassoM = R.methods.find((m) => m.id === 'lasso');
    const enetM = R.methods.find((m) => m.id === 'enet');
    const lasso = lassoM ? lassoM.extra : null;
    const enet = enetM ? enetM.extra : null;
    if (!lasso || !enet) return;

    makeChart('penaltyChart', {
      type: 'bar',
      data: {
        labels: R.names,
        datasets: [
          { label: `Lasso (α=${fmt(lasso.alpha)})`, data: lasso.coefs, backgroundColor: 'rgba(56,189,248,0.75)', borderRadius: 4 },
          { label: `ElasticNet (α=${fmt(enet.alpha)}, ρ=${R.options.l1Ratio.toFixed(2)})`, data: enet.coefs, backgroundColor: 'rgba(244,114,182,0.75)', borderRadius: 4 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y)}` } } },
        scales: { y: { title: { display: true, text: 'Standardized coefficient' }, ticks: { callback: (v) => fmt(v) } }, x: { grid: { display: false } } },
      },
    });
  }

  /* ---------------------------------------------------------- explorer */
  function setupExplorer() {
    const { R } = state;
    if (!R || !R.methods || !R.methods.length) return;
    const { best } = summarize();
    const prevM = $('exMethod').value, prevK = +$('exK').value;
    $('exMethod').innerHTML = R.methods.map((m) => `<option value="${m.id}">${m.name}</option>`).join('');
    $('exK').innerHTML = Array.from({ length: R.kmax }, (_, i) => `<option value="${i + 1}">k = ${i + 1}</option>`).join('');
    const keep = state._explorerTouched && R.methods.some((m) => m.id === prevM) && prevK <= R.kmax;
    $('exMethod').value = keep ? prevM : (best && best.method ? best.method.id : R.methods[0].id);
    $('exK').value = keep ? prevK : (best ? best.k : 1);
    renderExplorer();
  }

  function renderExplorer() {
    const { R, D } = state;
    if (!R || !D) return;
    const m = R.methods.find((x) => x.id === $('exMethod').value) || R.methods[0];
    if (!m || !m.perK) return;
    const kIdx = Math.max(0, Math.min(+$('exK').value - 1, m.perK.length - 1));
    const r = m.perK[kIdx];
    if (!r) return;
    const S = ML.olsSummary(D, R.selRows, r.subset);

    // equation
    const terms = S.terms.slice(1).map((t) => {
      const sign = t.coef >= 0 ? '+' : '−';
      return ` <span class="${t.coef >= 0 ? 'pos' : 'neg'}">${sign} ${fmt(Math.abs(t.coef))}</span>·${esc(t.name)}`;
    }).join('');
    $('equation').innerHTML = `<span class="y">ŷ(${esc(D.target)})</span> = ${fmt(S.terms[0].coef)}${terms}`;

    // actual vs predicted
    if (R.evaluator) {
      const preds = R.evaluator.predictions(r.subset);
      const all = preds.flatMap((p) => [p.y, p.pred]);
      const lo = Math.min(...all), hi = Math.max(...all);
      makeChart('avpChart', {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Rows', data: preds.map((p) => ({ x: p.y, y: p.pred })), backgroundColor: m.color + 'cc', pointRadius: 4 },
            { label: 'y = x', type: 'line', data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: 'rgba(226,232,240,0.5)', borderDash: [6, 6], pointRadius: 0, borderWidth: 1.5 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { title: { display: true, text: 'Actual' }, ticks: { callback: fmt } }, y: { title: { display: true, text: 'Predicted' }, ticks: { callback: fmt } } },
        },
      });

      makeChart('residChart', {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Residual', data: preds.map((p) => ({ x: p.pred, y: p.y - p.pred })), backgroundColor: 'rgba(244,114,182,0.75)', pointRadius: 4 },
            { type: 'line', data: [{ x: Math.min(...preds.map((p) => p.pred)), y: 0 }, { x: Math.max(...preds.map((p) => p.pred)), y: 0 }], borderColor: 'rgba(226,232,240,0.5)', borderDash: [6, 6], pointRadius: 0, borderWidth: 1.5 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { title: { display: true, text: 'Predicted' }, ticks: { callback: fmt } }, y: { title: { display: true, text: 'Actual − Predicted' }, ticks: { callback: fmt } } },
        },
      });
    }

    const coefTerms = S.terms.slice(1).slice().sort((a, b) => Math.abs(b.std) - Math.abs(a.std));
    makeChart('coefChart', {
      type: 'bar',
      data: {
        labels: coefTerms.map((t) => t.name),
        datasets: [{
          data: coefTerms.map((t) => t.std),
          backgroundColor: coefTerms.map((t) => (t.std >= 0 ? 'rgba(52,211,153,0.8)' : 'rgba(244,114,182,0.8)')),
          borderRadius: 5,
        }],
      },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${fmt(c.parsed.x)}` } } },
        scales: { x: { ticks: { callback: fmt } }, y: { grid: { display: false } } },
      },
    });

    const stars = (p) => (p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : p < 0.1 ? '.' : '');
    $('olsTable').innerHTML = `<thead><tr><th>Term</th><th>Coef</th><th>Std. Err</th><th>t</th><th>P&gt;|t|</th><th>Sig.</th></tr></thead><tbody>` +
      S.terms.map((t) => `<tr><td class="text">${esc(t.name)}</td><td>${fmt(t.coef)}</td><td>${fmt(t.se)}</td><td>${Number.isFinite(t.t) ? t.t.toFixed(3) : '—'}</td>
        <td class="${t.p < 0.05 ? 'sig' : 'nsig'}">${fmtP(t.p)}</td><td class="${t.p < 0.05 ? 'sig' : 'nsig'}">${stars(t.p)}</td></tr>`).join('') + '</tbody>';

    const evalName = R.options.mode === 'cv' ? 'CV' : 'Test';
    $('olsStats').innerHTML = [
      [`${evalName} RMSE`, fmt(r.rmse)], [`${evalName} R²`, fmtR2(r.r2)], [`${evalName} MAE`, fmt(r.mae)],
      ['Train R²', fmtR2(S.r2)], ['Adj. R²', fmtR2(S.adjR2)], ['Train RMSE', fmt(S.rmse)],
      ['F-stat', fmt(S.F)], ['Prob(F)', fmtP(S.Fp)], ['AIC', fmt(S.aic)], ['BIC', fmt(S.bic)], ['n', S.n], ['df resid', S.df],
    ].map(([k, v]) => `<span>${k}: <b>${v}</b></span>`).join('') + '<span class="muted">Sig. codes: *** 0.001 ** 0.01 * 0.05 . 0.1</span>';
  }

  /* ----------------------------------------------------------- heatmap */
  function renderHeatmap() {
    const { D } = state;
    if (!D) return;
    const MAX = 25;
    const names = [...D.names.slice(0, MAX - 1), D.target];
    const cols = [...D.names.slice(0, MAX - 1).map((_, j) => D.X.map((r) => r[j])), D.y];
    const C = ML.correlation(cols);
    const color = (v) => {
      const a = 0.08 + 0.85 * Math.abs(v);
      return v >= 0 ? `rgba(56,189,248,${a})` : `rgba(244,114,182,${a})`;
    };
    let html = '<thead><tr><th></th>' + names.map((n) => `<th>${esc(n)}</th>`).join('') + '</tr></thead><tbody>';
    C.forEach((row, a) => {
      html += `<tr><th>${esc(names[a])}</th>` + row.map((v, b) =>
        `<td style="background:${color(v)};color:${Math.abs(v) > 0.55 ? '#0b1020' : '#e2e8f0'}" title="${esc(names[a])} × ${esc(names[b])}: ${v.toFixed(3)}">${v.toFixed(2)}</td>`).join('') + '</tr>';
    });
    $('heatmap').innerHTML = html + '</tbody>';
  }

  function renderPreview() {
    const cfg = state.cfg[state.key];
    if (!cfg) return;
    const cols = [cfg.target, ...cfg.colInfo.map((c) => c.name).filter((n) => cfg.selected.has(n) && n !== cfg.target)];
    const idx = cols.map((n) => cfg.colInfo.find((c) => c.name === n).index);
    const rows = cfg.parsed.rows.slice(0, 100);
    $('previewTable').innerHTML = `<thead><tr><th>#</th>${cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>` +
      rows.map((r, i) => `<tr><td>${i + 1}</td>${idx.map((j) => `<td>${esc(r[j])}</td>`).join('')}</tr>`).join('') + '</tbody>';
  }

  /* ----------------------------------------------------------- results */
  function resultRows() {
    const { R } = state;
    if (!R || !R.methods) return [];
    return R.methods.flatMap((m) => (m.perK || []).map((r) => ({
      method: m.name, k: r.k, features: (r.subset || []).map((j) => R.names[j] || '').join(' + '), rmse: r.rmse, mae: r.mae, r2: r.r2,
    })));
  }

  function renderResultsTable() {
    $('resultsTable').innerHTML = '<thead><tr><th>Method</th><th>k</th><th>Features</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>' +
      resultRows().map((r) => `<tr><td class="text">${r.method}</td><td>${r.k}</td><td class="text" style="text-align:left">${esc(r.features)}</td>
        <td>${fmt(r.rmse)}</td><td>${fmt(r.mae)}</td><td>${fmtR2(r.r2)}</td></tr>`).join('') + '</tbody>';
  }

  function exportCSV() {
    if (!state.R) return;
    const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
    const lines = ['method,k,features,rmse,mae,r2', ...resultRows().map((r) =>
      [q(r.method), r.k, q(r.features), r.rmse, r.mae, r.r2].join(','))];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `feature_selection_${state.key}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function downloadChart(id) {
    const src = $(id);
    if (!src) return;
    const c = document.createElement('canvas');
    c.width = src.width; c.height = src.height;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#0d1326';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(src, 0, 0);
    const a = document.createElement('a');
    a.href = c.toDataURL('image/png');
    a.download = `${id}_${state.key}.png`;
    a.click();
  }

  /* ------------------------------------------------------------ upload */
  function handleFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = ML.parseCSV(reader.result);
        initConfig('custom', parsed, null);
        state.cfg.custom.fileName = file.name;
        if (!state.cfg.custom.target) throw new Error('No numeric column found to use as target.');
        $('fileInfo').innerHTML = `✅ <b>${esc(file.name)}</b> — ${parsed.rows.length} rows × ${parsed.headers.length} columns. Target defaulted to <b>${esc(state.cfg.custom.target)}</b>; change it below if needed.`;
        toast(`Loaded ${file.name}`, true);
        renderVariables();
        run();
      } catch (e) {
        toast('⚠ ' + e.message);
      }
    };
    reader.readAsText(file);
  }

  /* ------------------------------------------------------------ events */
  function bind() {
    $('datasetSelect').addEventListener('change', (e) => { state._explorerTouched = false; loadDataset(e.target.value); });
    $('modeSelect').addEventListener('change', () => {
      const ho = $('modeSelect').value === 'holdout';
      $('testSizeField').classList.toggle('hidden', !ho);
      $('repeatsField').classList.toggle('hidden', ho);
      $('foldsField').querySelector('span').textContent = ho ? 'Inner K folds' : 'K folds';
      scheduleRun(50);
    });
    ['foldsInput', 'repeatsInput', 'seedInput'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('change', () => scheduleRun(50));
    });
    $('testSizeInput').addEventListener('input', (e) => { $('testSizeVal').textContent = e.target.value + '%'; scheduleRun(); });
    $('alphaInput').addEventListener('input', (e) => { $('alphaVal').textContent = e.target.value + '%'; scheduleRun(); });
    $('l1Input').addEventListener('input', (e) => { $('l1Val').textContent = (e.target.value / 100).toFixed(2); scheduleRun(); });
    $('kmaxInput').addEventListener('input', (e) => {
      const cfg = state.cfg[state.key];
      if (!cfg) return;
      cfg.kmax = +e.target.value;
      $('kmaxVal').textContent = `${cfg.kmax} / ${cfg.p ?? '?'}`;
      scheduleRun();
    });

    $('targetSelect').addEventListener('change', (e) => {
      const cfg = state.cfg[state.key];
      cfg.selected.add(cfg.target);
      cfg.target = e.target.value;
      cfg.selected.delete(cfg.target);
      cfg.kmax = null;
      renderVariables();
      scheduleRun(50);
    });
    $('featureChecks').addEventListener('change', (e) => {
      const cfg = state.cfg[state.key];
      if (e.target.checked) cfg.selected.add(e.target.value); else cfg.selected.delete(e.target.value);
      cfg.kmax = null;
      scheduleRun(400);
    });
    $('toggleAllBtn').addEventListener('click', () => {
      const cfg = state.cfg[state.key];
      if (!cfg) return;
      const allNames = cfg.colInfo.map((c) => c.name).filter((n) => n !== cfg.target);
      const allOn = allNames.every((n) => cfg.selected.has(n));
      cfg.selected = new Set(allOn ? [] : allNames);
      cfg.kmax = null;
      renderVariables();
      if (!allOn) scheduleRun(50);
    });
    $('runBtn').addEventListener('click', () => run());

    $('langSeg').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      state.lang = b.dataset.lang;
      [...$('langSeg').children].forEach((x) => x.classList.toggle('active', x === b));
      renderVariables();
      if (state.R) { renderInsights(); renderNotes(); renderChartNote(); }
    });

    const filterBar = $('methodFilterBar');
    if (filterBar) {
      filterBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.filter-chip');
        if (!btn || !btn.dataset.method) return;
        setMethodFilter(btn.dataset.method);
      });
    }

    $('exMethod').addEventListener('change', () => { state._explorerTouched = true; renderExplorer(); });
    $('exK').addEventListener('change', () => { state._explorerTouched = true; renderExplorer(); });
    $('exportBtn').addEventListener('click', exportCSV);
    document.querySelectorAll('[data-download]').forEach((b) => b.addEventListener('click', () => downloadChart(b.dataset.download)));

    // upload
    $('fileInput').addEventListener('change', (e) => handleFile(e.target.files[0]));
    const dz = $('dropzone');
    if (dz) {
      ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
      ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
      dz.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
    }
  }

  // Ensure DOM is ready before initial load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      bind();
      loadDataset('startups');
    });
  } else {
    bind();
    loadDataset('startups');
  }
})();
