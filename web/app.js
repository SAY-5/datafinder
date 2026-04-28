// DataFinder · vanilla ES module UI.
const root = document.getElementById("app");
let state = { query: "", run: null, busy: false, error: "" };

const EXAMPLES = [
  "knee MRI cohort with cartilage thickness annotations and at least 50 subjects, age 40+",
  "lung CT studies with nodule annotations",
  "what's similar to the OAI knee dataset?",
  "show me ds_adni",
];

function el(t, a = {}, ...c) {
  const e = document.createElement(t);
  for (const [k, v] of Object.entries(a)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v != null) e.setAttribute(k, v);
  }
  for (const x of c.flat()) {
    if (x == null) continue;
    e.append(typeof x === "string" || typeof x === "number" ? document.createTextNode(String(x)) : x);
  }
  return e;
}

async function ask() {
  if (!state.query.trim() || state.busy) return;
  state.busy = true; state.error = ""; render();
  // Initial scaffold; streaming events fill it in as they arrive.
  const live = {
    id: "live", query: state.query, route: "—",
    normalized: { raw: state.query, text: state.query, hints: {} },
    answer: "", citations: [], grounded: false, refinements: 0,
    tool_calls: [],
  };
  state.run = live;
  try {
    const r = await fetch("/v1/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: state.query }),
    });
    if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        applyFrame(frame, live);
      }
    }
  } catch (e) {
    state.error = e.message;
  } finally {
    state.busy = false; render();
  }
}

function applyFrame(frame, run) {
  let event = "message"; let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  if (!data) return;
  let d; try { d = JSON.parse(data); } catch { return; }
  switch (event) {
    case "normalize":
      run.normalized = { raw: d.raw, text: d.text, hints: d.hints || {} };
      break;
    case "route":
      run.route = d.route;
      break;
    case "tool_call":
      run.tool_calls.push({
        idx: d.idx, tool: d.tool, args: d.args,
        result_summary: "(running…)", elapsed_ms: 0,
      });
      break;
    case "tool_result": {
      const tc = run.tool_calls.find((c) => c.idx === d.idx);
      if (tc) { tc.result_summary = d.summary; tc.elapsed_ms = d.elapsed_ms; }
      break;
    }
    case "refine":
      run.refinements = d.attempt;
      break;
    case "answer":
      run.answer = d.content;
      run.citations = d.citations || [];
      break;
    case "done":
      run.id = d.run_id;
      run.grounded = d.grounded;
      run.refinements = d.refinements;
      break;
    default: break;
  }
  render();
}

function render() {
  root.innerHTML = "";
  root.append(masthead(), main());
}

function masthead() {
  return el(
    "div",
    { class: "masthead" },
    el("div", { class: "brand" },
      el("div", {}, "data", el("b", {}, "Finder")),
      el("small", {}, "Research-Dataset Discovery Agent"),
    ),
    el("div", { class: "subtitle" },
      "An autonomous query-to-grounded-answer pipeline.",
    ),
    el("div", { class: "session-info" },
      "session",
      el("b", {}, state.run ? state.run.id.slice(-6).toUpperCase() : "—"),
    ),
  );
}

function main() {
  const m = el("main", {}, queryBox());
  if (state.error) m.append(el("div", { class: "empty" }, "Error: " + state.error));
  else if (state.busy) m.append(el("div", { class: "empty" }, el("div", { class: "eyebrow" }, "running"), "Routing query, calling tools, grounding answer…"));
  else if (state.run) m.append(runView(state.run));
  else m.append(emptyState());
  return m;
}

function queryBox() {
  return el(
    "div",
    { class: "query-box" },
    el("label", {}, "QUERY"),
    el("textarea", {
      placeholder: "e.g. knee MRI datasets with at least 100 subjects, age 40+, annotated for cartilage thickness",
      value: state.query,
      onInput: (e) => { state.query = e.target.value; },
      onKeydown: (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); },
    }),
    el("div", { class: "row" },
      el("div", { class: "examples" },
        EXAMPLES.map((x) =>
          el("button", { onClick: () => { state.query = x; render(); } }, x.slice(0, 60) + (x.length > 60 ? "…" : "")),
        ),
      ),
      el("button", { class: "submit", onClick: ask, disabled: state.busy ? "" : null },
        state.busy ? "Asking…" : "Ask  ⌘↩"),
    ),
  );
}

function emptyState() {
  return el(
    "div",
    { class: "empty" },
    el("div", { class: "eyebrow" }, "ready"),
    "Ask a question. The agent will route it, sequence its tools, and ground its answer with citations.",
  );
}

function runView(run) {
  return el(
    "div",
    { class: "run" },
    answerView(run),
    methodsView(run),
  );
}

function answerView(run) {
  const grounded = run.grounded;
  return el(
    "article",
    { class: "answer" },
    el("div", { class: "lede" }, "Route · " + run.route),
    el("h2", {},
      "Answer for ",
      el("em", {}, " " + (run.query.length > 60 ? run.query.slice(0, 60) + "…" : run.query)),
    ),
    el("div", { class: "body", html: linkifyDatasetIds(escapeHtml(run.answer)) }),
    el(
      "div",
      { class: "grounding-banner" + (grounded ? "" : " ungrounded") },
      grounded
        ? `grounded · ${run.citations.length} citation${run.citations.length === 1 ? "" : "s"}`
        : `ungrounded after ${run.refinements} refinement${run.refinements === 1 ? "" : "s"}`,
    ),
    run.citations.length > 0
      ? el("div", { class: "citations" },
          el("h3", {}, "Citations"),
          el("ol", {}, run.citations.map((c) => el("li", {}, el("code", {}, c)))))
      : null,
  );
}

function methodsView(run) {
  return el(
    "aside",
    { class: "methods" },
    el("h3", {}, "Methods · tool trace"),
    el(
      "div",
      { class: "meta" },
      "ROUTE ", el("b", {}, run.route), " · REFINEMENTS ", el("b", {}, run.refinements), " · TOOLS ", el("b", {}, run.tool_calls.length),
      run.normalized.hints && Object.keys(run.normalized.hints).length > 0
        ? el("div", { style: "margin-top: 6px" },
            "extracted hints: ",
            el("code", { style: "font-family: var(--mono); font-size: 10.5px;" },
              JSON.stringify(run.normalized.hints)))
        : null,
    ),
    el(
      "ol",
      { class: "trace" },
      run.tool_calls.map((tc) =>
        el("li", {},
          el("div", {}, el("span", { class: "tool" }, tc.tool), el("span", { class: "ms" }, tc.elapsed_ms + "ms")),
          el("div", { class: "args" }, JSON.stringify(tc.args)),
          el("div", { class: "summary" }, tc.result_summary.slice(0, 200) + (tc.result_summary.length > 200 ? "…" : "")),
        ),
      ),
    ),
  );
}

function linkifyDatasetIds(s) {
  return s.replace(/\b(ds[_-][a-zA-Z0-9_]+)\b/g, '<span class="ds-id">$1</span>');
}
function escapeHtml(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});

render();
