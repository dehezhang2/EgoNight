import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarize_accuracy import (
    process_dataset,
    merge_accuracy,
    DIFFICULTY_LEVELS,
    load_and_filter_scores,
    OXFORD_DIFFICULTY,
    SYNTHETIC_DIFFICULTY,
    SOFIA_DIFFICULTY,
)

from flask import Flask, request, jsonify

app = Flask(__name__)

CONFIG = {
    "sofia_path": None,
    "oxford_path": None,
    "synthetic_path": None,
    "model": "gpt",
}

_DIFF_MAP = {
    "sofia": SOFIA_DIFFICULTY,
    "oxford": OXFORD_DIFFICULTY,
    "synthetic": SYNTHETIC_DIFFICULTY,
}


@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


def _configured_datasets():
    return [
        (name, CONFIG[f"{name}_path"])
        for name in ("sofia", "oxford", "synthetic")
        if CONFIG.get(f"{name}_path")
    ]


def _filter_datasets(dataset_param):
    all_ds = _configured_datasets()
    if dataset_param == "all":
        return all_ds
    return [(n, p) for n, p in all_ds if n == dataset_param]


def _splits(split_param):
    return ["day", "night"] if split_param == "all" else [split_param]


@app.route("/api/models")
def api_models():
    try:
        models = set()
        for ds_name, ds_path in _configured_datasets():
            if not os.path.isdir(ds_path):
                continue
            for scene in os.listdir(ds_path):
                for subdir in ("qa_result", "final_score"):
                    sdir = os.path.join(ds_path, scene, subdir)
                    if not os.path.isdir(sdir):
                        continue
                    for fname in os.listdir(sdir):
                        if fname.endswith(".json"):
                            if "_scores_" in fname:
                                models.add(fname.split("_scores_")[0])
                            elif fname.startswith("filtered_") and "_final_scores_" in fname:
                                models.add(fname[len("filtered_"):].split("_final_scores_")[0])
        default = CONFIG.get("model", "gpt")
        result = sorted(models)
        if default not in result:
            result.insert(0, default)
        return jsonify({"models": result, "default": default})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/summary")
def api_summary():
    try:
        model = request.args.get("model", CONFIG["model"])
        split_param = request.args.get("split", "all")
        dataset_param = request.args.get("dataset", "all")
        difficulty_param = request.args.get("difficulty", "all")

        splits = _splits(split_param)
        datasets = _filter_datasets(dataset_param)

        combined_overall = {}
        combined_by_difficulty = {lv: {} for lv in DIFFICULTY_LEVELS}
        dataset_results = {}

        for ds_name, ds_path in datasets:
            ds_overall = {}
            ds_by_diff = {lv: {} for lv in DIFFICULTY_LEVELS}

            for sp in splits:
                try:
                    result = process_dataset(ds_path, ds_name, model, sp)
                except Exception:
                    result = {"overall": {}, "by_difficulty": {lv: {} for lv in DIFFICULTY_LEVELS}}

                ds_overall = merge_accuracy(ds_overall, result.get("overall", {}))
                for lv in DIFFICULTY_LEVELS:
                    ds_by_diff[lv] = merge_accuracy(
                        ds_by_diff.get(lv, {}),
                        result.get("by_difficulty", {}).get(lv, {}),
                    )

            combined_overall = merge_accuracy(combined_overall, ds_overall)
            for lv in DIFFICULTY_LEVELS:
                combined_by_difficulty[lv] = merge_accuracy(
                    combined_by_difficulty.get(lv, {}),
                    ds_by_diff.get(lv, {}),
                )

            dataset_results[ds_name] = {"overall": ds_overall, "by_difficulty": ds_by_diff}

        if difficulty_param != "all":
            effective_overall = combined_by_difficulty.get(difficulty_param, {})
            effective_by_diff = {difficulty_param: combined_by_difficulty.get(difficulty_param, {})}
            for dn in dataset_results:
                dataset_results[dn]["overall"] = dataset_results[dn]["by_difficulty"].get(difficulty_param, {})
                dataset_results[dn]["by_difficulty"] = {
                    difficulty_param: dataset_results[dn]["by_difficulty"].get(difficulty_param, {})
                }
        else:
            effective_overall = combined_overall
            effective_by_diff = combined_by_difficulty

        return jsonify({
            "overall": effective_overall,
            "by_difficulty": effective_by_diff,
            "dataset_results": dataset_results,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/pairs")
def api_pairs():
    try:
        model = request.args.get("model", CONFIG["model"])
        split_param = request.args.get("split", "all")
        dataset_param = request.args.get("dataset", "all")
        difficulty_param = request.args.get("difficulty", "all")
        qatype_param = request.args.get("qatype", "all")
        page = max(1, int(request.args.get("page", 1)))
        per_page = max(1, int(request.args.get("per_page", 50)))

        splits = _splits(split_param)
        datasets = _filter_datasets(dataset_param)
        pairs = []

        for ds_name, ds_path in datasets:
            diff_map = _DIFF_MAP.get(ds_name, {})
            try:
                scenes = [d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))]
            except Exception:
                scenes = []

            for scene in scenes:
                scene_diff = diff_map.get(scene)
                if difficulty_param != "all" and scene_diff != difficulty_param:
                    continue

                for sp in splits:
                    score_path = os.path.join(ds_path, scene, "qa_result", f"{model}_scores_{sp}.json")
                    if not os.path.exists(score_path):
                        score_path = os.path.join(ds_path, scene, "final_score", f"filtered_{model}_final_scores_{sp}.json")
                    if not os.path.exists(score_path):
                        continue
                    try:
                        items = load_and_filter_scores(score_path, None)
                    except Exception:
                        continue

                    for pred_obj, qa_obj in items:
                        qa_type = qa_obj.get("M", "").strip().replace("_", " ").title()
                        if qatype_param != "all" and qa_type != qatype_param:
                            continue
                        pairs.append({
                            "dataset": ds_name,
                            "scene": scene,
                            "split": sp,
                            "question": qa_obj.get("Q", ""),
                            "ground_truth": qa_obj.get("C", ""),
                            "prediction": qa_obj.get("A", ""),
                            "qa_type": qa_type,
                            "difficulty": scene_diff,
                            "correct": pred_obj.get("pred", "") == "correct",
                            "score": pred_obj.get("score"),
                            "reason": pred_obj.get("reason"),
                        })

        total = len(pairs)
        pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, pages)
        start = (page - 1) * per_page
        return jsonify({
            "pairs": pairs[start: start + per_page],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>EgoNight Benchmark</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--bg-card:#1e2130;--bg-alt:#161928;--border:#2d3148;
  --accent:#7dd3fc;--accent2:#3b82f6;--text:#e2e8f0;--text-muted:#94a3b8;
  --green:#22c55e;--red:#ef4444;--radius:10px;
}
html,body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,-apple-system,sans-serif;font-size:14px;min-height:100vh}
header{background:linear-gradient(135deg,#12151f 0%,#1a1e2e 100%);border-bottom:1px solid var(--border);padding:16px 28px;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:14px;position:sticky;top:0;z-index:100;box-shadow:0 2px 20px rgba(0,0,0,.4)}
.logo{font-size:1.3rem;font-weight:700;letter-spacing:-.02em;background:linear-gradient(90deg,var(--accent),#a5f3fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;padding-top:6px;white-space:nowrap}
.filter-bar{display:flex;flex-wrap:wrap;gap:9px;align-items:flex-end}
.filter-group{display:flex;flex-direction:column;gap:3px}
.filter-group label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--text-muted)}
.filter-group input,.filter-group select{background:#252840;border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-size:13px;font-family:inherit;outline:none;min-width:105px;transition:border-color .15s}
.filter-group input:focus,.filter-group select:focus{border-color:var(--accent2)}
.btn{padding:7px 18px;border-radius:7px;border:none;cursor:pointer;font-family:inherit;font-size:13px;font-weight:600;background:linear-gradient(135deg,var(--accent2),#0ea5e9);color:#fff;transition:filter .15s,transform .15s;align-self:flex-end}
.btn:hover{filter:brightness(1.18);transform:translateY(-1px)}
main{max-width:1420px;margin:0 auto;padding:26px 28px;display:flex;flex-direction:column;gap:26px}
.section-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--text-muted);margin-bottom:12px;display:flex;align-items:center;gap:10px}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}
.stat-cards{display:flex;flex-wrap:wrap;gap:13px}
.stat-card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;min-width:130px;flex:1;position:relative;overflow:hidden;transition:border-color .2s,transform .2s}
.stat-card:hover{border-color:var(--accent2);transform:translateY(-2px)}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent2),var(--accent))}
.stat-card .lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--text-muted);margin-bottom:7px}
.stat-card .val{font-size:1.85rem;font-weight:700;background:linear-gradient(90deg,var(--accent),#c7d2fe);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1}
.stat-card .sub{font-size:11px;color:var(--text-muted);margin-top:4px}
.progress-list{display:flex;flex-direction:column;gap:9px}
.progress-row{display:grid;grid-template-columns:190px 1fr 58px;align-items:center;gap:12px}
.progress-name{font-size:12px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.progress-track{background:var(--bg-alt);border-radius:999px;height:12px;overflow:hidden;border:1px solid var(--border)}
.progress-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .5s cubic-bezier(.4,0,.2,1)}
.progress-pct{font-size:12px;font-weight:600;color:var(--accent);text-align:right}
.table-wrapper{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:13px}
thead tr{background:#13162280}
thead th{padding:10px 13px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);white-space:nowrap;border-bottom:1px solid var(--border)}
tbody tr{border-bottom:1px solid var(--border);transition:background .15s}
tbody tr:nth-child(odd){background:var(--bg-card)}
tbody tr:nth-child(even){background:var(--bg-alt)}
tbody tr:hover{background:#252840}
tbody tr.correct{border-left:3px solid var(--green)}
tbody tr.incorrect{border-left:3px solid var(--red)}
td{padding:8px 13px;vertical-align:top;max-width:260px}
td.narrow{max-width:100px;white-space:nowrap}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600;background:#252840;color:var(--text-muted);border:1px solid var(--border)}
.tag-ok{background:#14311f;color:var(--green);border-color:#1e5232}
.tag-fail{background:#2e1515;color:var(--red);border-color:#5c2020}
.pagination{display:flex;align-items:center;justify-content:center;gap:10px;padding:13px 0 3px;font-size:13px;color:var(--text-muted)}
.page-btn{padding:5px 13px;border-radius:6px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);cursor:pointer;font-size:13px;font-family:inherit;transition:border-color .15s}
.page-btn:hover:not(:disabled){border-color:var(--accent2);color:var(--accent)}
.page-btn:disabled{opacity:.35;cursor:not-allowed}
.empty{color:var(--text-muted);padding:30px;text-align:center;font-size:13px}
.loading{color:var(--text-muted);padding:30px;text-align:center;font-style:italic}
</style>
</head>
<body>
<header>
  <div class="logo">EgoNight Benchmark</div>
  <div class="filter-bar">
    <div class="filter-group">
      <label>Model</label>
      <select id="f-model"><option value="gpt">gpt</option></select>
    </div>
    <div class="filter-group">
      <label>Split</label>
      <select id="f-split">
        <option value="all">all</option>
        <option value="day">day</option>
        <option value="night">night</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Dataset</label>
      <select id="f-dataset">
        <option value="all">all</option>
        <option value="sofia">sofia</option>
        <option value="oxford">oxford</option>
        <option value="synthetic">synthetic</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Difficulty</label>
      <select id="f-difficulty">
        <option value="all">all</option>
        <option value="easy">easy</option>
        <option value="medium">medium</option>
        <option value="hard">hard</option>
      </select>
    </div>
    <div class="filter-group">
      <label>QA Type</label>
      <select id="f-qatype">
        <option value="all">all</option>
        <option>Object Recognition</option>
        <option>Spatial Reasoning</option>
        <option>Scene Sequence</option>
        <option>Non Common</option>
        <option>Counting</option>
        <option>Navigation</option>
        <option>Text Recognition</option>
        <option>Action</option>
      </select>
    </div>
    <button class="btn" onclick="applyFilters()">Apply Filters</button>
  </div>
</header>
<main>
  <section id="sec-overview">
    <div class="section-title">Accuracy Overview</div>
    <div class="stat-cards" id="stat-cards"><div class="loading">Loading...</div></div>
  </section>
  <section id="sec-breakdown">
    <div class="section-title">QA Type Breakdown</div>
    <div class="progress-list" id="progress-list"><div class="loading">Loading...</div></div>
  </section>
  <section id="sec-pairs">
    <div class="section-title">QA Pairs</div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Dataset</th><th>Scene</th><th>Split</th><th>Difficulty</th>
            <th>QA Type</th><th>Question</th><th>Ground Truth</th>
            <th>Prediction</th><th>Score</th><th>Correct</th>
          </tr>
        </thead>
        <tbody id="pairs-tbody"><tr><td colspan="10" class="loading">Loading...</td></tr></tbody>
      </table>
    </div>
    <div class="pagination" id="pagination"></div>
  </section>
</main>
<script>
const state = { model: 'gpt', split: 'all', dataset: 'all', difficulty: 'all', qatype: 'all', page: 1 };

async function fetchModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const sel = document.getElementById('f-model');
    sel.innerHTML = data.models.map(m => `<option value="${m}"${m===data.default?' selected':''}>${m}</option>`).join('');
    state.model = data.default;
  } catch(e) { console.error('fetchModels', e); }
}

function getFilters() {
  state.model      = document.getElementById('f-model').value.trim() || 'gpt';
  state.split      = document.getElementById('f-split').value;
  state.dataset    = document.getElementById('f-dataset').value;
  state.difficulty = document.getElementById('f-difficulty').value;
  state.qatype     = document.getElementById('f-qatype').value;
}

function qs(extra) {
  const p = new URLSearchParams({
    model: state.model, split: state.split, dataset: state.dataset,
    difficulty: state.difficulty, qatype: state.qatype
  });
  if (extra) Object.entries(extra).forEach(([k,v]) => p.set(k,v));
  return p.toString();
}

async function fetchSummary() {
  const el = document.getElementById('stat-cards');
  const pl = document.getElementById('progress-list');
  el.innerHTML = '<div class="loading">Loading...</div>';
  pl.innerHTML = '';
  try {
    const r = await fetch('/api/summary?' + qs());
    const data = await r.json();
    if (data.error) { el.innerHTML = `<div class="empty">Error: ${data.error}</div>`; return; }
    renderSummary(data);
  } catch(e) {
    el.innerHTML = `<div class="empty">Fetch error: ${e.message}</div>`;
  }
}

async function fetchPairs() {
  const tbody = document.getElementById('pairs-tbody');
  const pag   = document.getElementById('pagination');
  tbody.innerHTML = '<tr><td colspan="10" class="loading">Loading...</td></tr>';
  pag.innerHTML = '';
  try {
    const r = await fetch('/api/pairs?' + qs({ page: state.page, per_page: 50 }));
    const data = await r.json();
    if (data.error) { tbody.innerHTML = `<tr><td colspan="10" class="empty">Error: ${data.error}</td></tr>`; return; }
    renderPairs(data);
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty">Fetch error: ${e.message}</td></tr>`;
  }
}

function pct(v) {
  if (v === null || v === undefined) return 'N/A';
  return (v * 100).toFixed(1) + '%';
}

function renderSummary(data) {
  const overall = data.overall || {};
  const byDiff  = data.by_difficulty || {};

  const overallTotal   = Object.values(overall).reduce((s,v) => s + (v.total||0), 0);
  const overallCorrect = Object.values(overall).reduce((s,v) => s + (v.correct||0), 0);
  const overallAcc     = overallTotal > 0 ? overallCorrect / overallTotal : null;

  let cards = `<div class="stat-card">
    <div class="lbl">Overall</div>
    <div class="val">${overallAcc !== null ? (overallAcc*100).toFixed(1)+'%' : 'N/A'}</div>
    <div class="sub">${overallCorrect} / ${overallTotal}</div>
  </div>`;

  ['easy','medium','hard'].forEach(lv => {
    const lvData = byDiff[lv] || {};
    const t = Object.values(lvData).reduce((s,v) => s + (v.total||0), 0);
    const c = Object.values(lvData).reduce((s,v) => s + (v.correct||0), 0);
    const a = t > 0 ? c/t : null;
    cards += `<div class="stat-card">
      <div class="lbl">${lv}</div>
      <div class="val">${a !== null ? (a*100).toFixed(1)+'%' : 'N/A'}</div>
      <div class="sub">${c} / ${t}</div>
    </div>`;
  });

  document.getElementById('stat-cards').innerHTML = cards;

  const qaTypes = Object.keys(overall).filter(k => k !== 'OVERALL');
  if (!qaTypes.length) {
    document.getElementById('progress-list').innerHTML = '<div class="empty">No data</div>';
    return;
  }

  const rows = qaTypes.map(qt => {
    const v = overall[qt];
    const a = v.total > 0 ? v.correct / v.total : 0;
    const w = (a * 100).toFixed(1);
    return `<div class="progress-row">
      <div class="progress-name">${qt}</div>
      <div class="progress-track"><div class="progress-fill" style="width:${w}%"></div></div>
      <div class="progress-pct">${w}%</div>
    </div>`;
  }).join('');

  document.getElementById('progress-list').innerHTML = rows || '<div class="empty">No data</div>';
}

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderPairs(data) {
  const pairs = data.pairs || [];
  if (!pairs.length) {
    document.getElementById('pairs-tbody').innerHTML = '<tr><td colspan="10" class="empty">No pairs found.</td></tr>';
    document.getElementById('pagination').innerHTML = '';
    return;
  }

  const rows = pairs.map(p => {
    const cls  = p.correct ? 'correct' : 'incorrect';
    const tag  = p.correct
      ? '<span class="badge tag-ok">&#10003;</span>'
      : '<span class="badge tag-fail">&#10007;</span>';
    return `<tr class="${cls}">
      <td class="narrow">${esc(p.dataset)}</td>
      <td class="narrow">${esc(p.scene)}</td>
      <td class="narrow">${esc(p.split)}</td>
      <td class="narrow">${esc(p.difficulty)}</td>
      <td><span class="badge">${esc(p.qa_type)}</span></td>
      <td>${esc(p.question)}</td>
      <td>${esc(p.ground_truth)}</td>
      <td>${esc(p.prediction)}</td>
      <td class="narrow">${p.score !== null && p.score !== undefined ? esc(String(p.score)) : ''}</td>
      <td class="narrow" style="text-align:center">${tag}</td>
    </tr>`;
  }).join('');

  document.getElementById('pairs-tbody').innerHTML = rows;

  const { page, pages } = data;
  let pag = `<button class="page-btn" onclick="goPage(${page-1})" ${page<=1?'disabled':''}>&#8592; Prev</button>`;
  pag += `<span>Page ${page} of ${pages}</span>`;
  pag += `<button class="page-btn" onclick="goPage(${page+1})" ${page>=pages?'disabled':''}>Next &#8594;</button>`;
  document.getElementById('pagination').innerHTML = pag;
}

function goPage(n) {
  state.page = n;
  fetchPairs();
}

function applyFilters() {
  getFilters();
  state.page = 1;
  fetchSummary();
  fetchPairs();
}

window.addEventListener('DOMContentLoaded', async () => { await fetchModels(); applyFilters(); });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


parser = argparse.ArgumentParser(description="EgoNight benchmark visualizer")
parser.add_argument("--sofia_path",    type=str, default=None)
parser.add_argument("--oxford_path",   type=str, default=None)
parser.add_argument("--synthetic_path",type=str, default=None)
parser.add_argument("--model",         type=str, default="gpt")
parser.add_argument("--port",          type=int, default=5000)


if __name__ == "__main__":
    args = parser.parse_args()
    CONFIG["sofia_path"]    = args.sofia_path
    CONFIG["oxford_path"]   = args.oxford_path
    CONFIG["synthetic_path"]= args.synthetic_path
    CONFIG["model"]         = args.model
    app.run(host="0.0.0.0", port=args.port, debug=False)
