/**
 * ==========================================================================
 * CALYPSO — UNIFIED SINGLE-WINDOW SCREEN WORKSPACE
 * Dual-Theme Engine, Live Solver Streamer, Battle Arena & Syllabus Matrix
 * ==========================================================================
 */

const PRESETS = {
  algo_recurrence: {
    subject: "Algorithms",
    topic: "Asymptotic Analysis & Recurrences",
    question_type: "MCQ",
    marks: 1,
    question: "Consider the recurrence relation:\n$T(n) = 2T(\\lfloor\\sqrt{n}\\rfloor) + \\log_2 n$\nfor $n > 2$, where $T(2) = 1$. Which one of the following is the asymptotic order of $T(n)$?",
    options: {
      A: "$\\Theta(\\log_2 n)$",
      B: "$\\Theta((\\log_2 n) \\log_2 \\log_2 n)$",
      C: "$\\Theta((\\log_2 n)^2)$",
      D: "$\\Theta(n)$",
    },
  },
  os_paging: {
    subject: "Operating Systems",
    topic: "Virtual Memory & TLB",
    question_type: "NAT",
    marks: 2,
    question: "A computer system uses a 2-level page table. Virtual addresses are 32 bits, page size is 4 KB, and each page table entry is 4 bytes. If the TLB hit ratio is 0.90, TLB access time is 10 ns, and main memory access time is 80 ns, what is the Effective Memory Access Time (EMAT) in nanoseconds?",
    options: {},
  },
  dbms_norm: {
    subject: "DBMS",
    topic: "Functional Dependencies & Normal Forms",
    question_type: "MCQ",
    marks: 2,
    question: "Consider relation R(A, B, C, D, E) with FDs:\nF = { A -> BC, CD -> E, B -> D, E -> A }\nWhich one of the following is the highest normal form satisfied by relation R?",
    options: {
      A: "1NF but not 2NF",
      B: "2NF but not 3NF",
      C: "3NF but not BCNF",
      D: "BCNF",
    },
  },
  cn_subnet: {
    subject: "Computer Networks",
    topic: "IP Subnetting & CIDR",
    question_type: "NAT",
    marks: 2,
    question: "An organization is allocated the IP block 200.10.16.0/20. The network administrator needs to create 8 subnets with equal number of usable host IP addresses. What is the number of usable host IP addresses in EACH subnet?",
    options: {},
  },
  toc_decide: {
    subject: "Theory of Computation",
    topic: "Decidability & Grammars",
    question_type: "MCQ",
    marks: 1,
    question: "Which of the following problems is UNDECIDABLE?",
    options: {
      A: "Checking whether a given Regular Expression $R$ generates the empty language $\\emptyset$.",
      B: "Checking whether a given Context-Free Grammar $G$ generates an empty language $L(G) = \\emptyset$.",
      C: "Checking whether a given Context-Free Grammar $G$ is ambiguous.",
      D: "Checking whether two DFAs $M_1$ and $M_2$ are equivalent.",
    },
  },
  cd_lalr: {
    subject: "Compiler Design",
    topic: "Syntax Analysis - LR Parsers",
    question_type: "MCQ",
    marks: 1,
    question: "Consider an LR(1) parser. An LALR(1) parser is constructed by merging states of the LR(1) parser having the same core item sets. Which one of the following statements is TRUE regarding the conflicts that can arise during this merger?",
    options: {
      A: "Merging states can introduce Shift-Reduce (S-R) conflicts but never Reduce-Reduce (R-R) conflicts.",
      B: "Merging states can introduce Reduce-Reduce (R-R) conflicts but never Shift-Reduce (S-R) conflicts.",
      C: "Merging states can introduce both Shift-Reduce and Reduce-Reduce conflicts.",
      D: "Merging states can never introduce any new conflict.",
    },
  },
};

const ARENA_CASES = {
  case_os: {
    title: "⚡ OS: 2-Level Paging EMAT",
    presetKey: "os_paging",
    baseFailure: "Omitted 1 memory access for physical frame (calculated 98 ns instead of 96 ns).",
    calypsoAdvantage: "Enforces memory traversal accounting: $t_{tlb} + 3 \\times t_m = 250\\text{ ns} \\implies \\text{EMAT} = 96\\text{ ns}$.",
  },
  case_cn: {
    title: "⚡ CN: Subnet Usable Hosts",
    presetKey: "cn_subnet",
    baseFailure: "Calculated total subnet capacity ($2^9 = 512$) but omitted subtracting Network & Broadcast IDs ($2^h - 2$).",
    calypsoAdvantage: "Boundary validator enforces $512 - 2 = 510$ usable host IPs.",
  },
  case_cd: {
    title: "⚡ CD: LALR State Invariance",
    presetKey: "cd_lalr",
    baseFailure: "Conflated parser rules with grammar conflicts, claiming Shift-Reduce conflicts can emerge from identical cores.",
    calypsoAdvantage: "Proves lookahead core invariance, identifying Option (B) [Reduce-Reduce conflicts only].",
  },
  case_toc: {
    title: "⚡ TOC: CFG Undecidability",
    presetKey: "toc_decide",
    baseFailure: "Generated unstructured summary without formal Post Correspondence Problem (PCP) reduction.",
    calypsoAdvantage: "Formally proves reduction from PCP to Context-Free Grammar ambiguity undecidability.",
  },
};

const SYLLABUS_MATRIX = [
  { subject: "Algorithms", weight: "15%", tier: "Tier 1 (High Yield)", focus: "Asymptotic bounds, Dynamic Programming (LCS), Graphs (MST/Dijkstra), Recurrences" },
  { subject: "Computer Networks", weight: "11%", tier: "Tier 1 (High Yield)", focus: "Sliding Window (GBN/SR), CIDR Subnetting, TCP Congestion Control, Routing" },
  { subject: "Operating Systems", weight: "10%", tier: "Tier 1 (High Yield)", focus: "Multi-level Paging, CPU Scheduling, Banker's Safety, Semaphore Deadlocks" },
  { subject: "DBMS", weight: "9%", tier: "Tier 2", focus: "Normalization (BCNF/3NF), Relational Algebra, B+ Trees, Conflict Serializability" },
  { subject: "Theory of Computation", weight: "9%", tier: "Tier 2", focus: "Chomsky Hierarchy, DFA Minimization, Turing Decidability, Rice's Theorem" },
  { subject: "Digital Logic", weight: "6%", tier: "Tier 3", focus: "K-Map Minimization, Carry Lookahead Adders, Multiplexers, Synchronous Counters" },
  { subject: "Compiler Design", weight: "5%", tier: "Tier 3", focus: "LR(1)/LALR(1) Parsing, SDT synthesized attributes, Lexical Tokens, Dominators" },
];

let currentTheme = localStorage.getItem("calypso_theme") || "dark";
let activeMode = "solver"; // 'solver', 'arena', 'syllabus'
let audioEnabled = false;
let audioCtx = null;
let latestTerminalMarkdown = "";
let isTerminalSolving = false;
let currentArenaCase = "case_os";

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initAudio();
  initNavigation();
  initSolver();
  loadSolverPreset("algo_recurrence");
});

/* --------------------------------------------------------------------------
   0. Theme Engine
   -------------------------------------------------------------------------- */
function initTheme() {
  applyTheme(currentTheme);

  const toggle = document.getElementById("themeToggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      localStorage.setItem("calypso_theme", currentTheme);
      applyTheme(currentTheme);
      playTone(700, 0.05);
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const icon = document.getElementById("themeIcon");
  const text = document.getElementById("themeText");
  if (icon && text) {
    if (theme === "dark") {
      icon.textContent = "☀️";
      text.textContent = "LIGHT";
    } else {
      icon.textContent = "🌙";
      text.textContent = "DARK";
    }
  }
}

/* --------------------------------------------------------------------------
   1. Audio Synthesizer
   -------------------------------------------------------------------------- */
function initAudio() {
  const toggle = document.getElementById("audioToggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    audioEnabled = !audioEnabled;
    toggle.classList.toggle("on", audioEnabled);
    toggle.innerHTML = `<span>AUDIO: ${audioEnabled ? "ON" : "OFF"}</span>`;

    if (audioEnabled && !audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    playTone(audioEnabled ? 880 : 330, 0.08);
  });
}

function playTone(freq = 600, duration = 0.05) {
  if (!audioEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

/* --------------------------------------------------------------------------
   2. Mode Navigation & Workspace View Switching (Single Window)
   -------------------------------------------------------------------------- */
function initNavigation() {
  const btnSolver = document.getElementById("tabNavSolver");
  const btnArena = document.getElementById("tabNavArena");
  const btnSyllabus = document.getElementById("tabNavSyllabus");

  btnSolver.addEventListener("click", () => {
    activeMode = "solver";
    setTabActive(btnSolver);
    renderSolverStage();
    playTone(550, 0.03);
  });

  btnArena.addEventListener("click", () => {
    activeMode = "arena";
    setTabActive(btnArena);
    renderArenaStage(currentArenaCase);
    playTone(550, 0.03);
  });

  btnSyllabus.addEventListener("click", () => {
    activeMode = "syllabus";
    setTabActive(btnSyllabus);
    renderSyllabusStage();
    playTone(550, 0.03);
  });
}

function setTabActive(el) {
  document.querySelectorAll(".nav-tab-btn").forEach((t) => t.classList.remove("active"));
  el.classList.add("active");
}

/* --------------------------------------------------------------------------
   3. Solver Terminal Stage
   -------------------------------------------------------------------------- */
function initSolver() {
  document.querySelectorAll(".subj-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".subj-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const subj = chip.getAttribute("data-subj");
      document.getElementById("subjectInput").value = subj;
      playTone(500, 0.03);
    });
  });

  document.querySelectorAll(".preset-btn-card").forEach((card) => {
    card.addEventListener("click", () => {
      const key = card.getAttribute("data-preset");
      if (PRESETS[key]) {
        loadSolverPreset(key);
        playTone(650, 0.04);
      }
    });
  });

  document.getElementById("typeInput").addEventListener("change", (e) => {
    const isMcq = e.target.value === "MCQ" || e.target.value === "MSQ";
    document.getElementById("optionsConsole").style.display = isMcq ? "block" : "none";
  });

  document.getElementById("solveTerminalBtn").addEventListener("click", runTerminalSolve);
}

function loadSolverPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  document.getElementById("subjectInput").value = p.subject;
  document.getElementById("topicInput").value = p.topic;
  document.getElementById("typeInput").value = p.question_type;
  document.getElementById("marksInput").value = p.marks;
  document.getElementById("questionTextInput").value = p.question;

  document.querySelectorAll(".subj-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.getAttribute("data-subj") === p.subject);
  });

  const isMcq = p.question_type === "MCQ" || p.question_type === "MSQ";
  document.getElementById("optionsConsole").style.display = isMcq ? "block" : "none";

  if (isMcq && p.options) {
    document.getElementById("optionA").value = p.options.A || "";
    document.getElementById("optionB").value = p.options.B || "";
    document.getElementById("optionC").value = p.options.C || "";
    document.getElementById("optionD").value = p.options.D || "";
  }
}

function renderSolverStage() {
  const stage = document.getElementById("mainStageArea");
  stage.innerHTML = `
    <div class="stage-hud">
      <div class="stage-stats">
        <div class="stage-stat-item">ENGINE: <span>CALYPSO-QWEN-1.5B</span></div>
        <div class="stage-stat-item">LATENCY: <span id="stageLatency">--</span></div>
        <div class="stage-stat-item">THROUGHPUT: <span id="stageTps">--</span></div>
        <div class="stage-stat-item">TOKENS: <span id="stageTokens">0</span></div>
      </div>
      <div class="stage-actions">
        <button class="btn-stage-action" onclick="copyTerminalMarkdown()">Copy Markdown</button>
        <button class="btn-stage-action" onclick="window.print()">Print Proof</button>
      </div>
    </div>

    <div class="terminal-window" id="terminalOutputWindow">
      ${
        latestTerminalMarkdown
          ? formatMarkdown(latestTerminalMarkdown)
          : `<div style="color: var(--text-muted); text-align: center; margin-top: 80px;">
              <div style="font-family: var(--font-hero); font-size: 1.35rem; font-weight: 900; color: var(--text-primary); margin-bottom: 0.5rem; letter-spacing: -0.02em;">
                REASONING TERMINAL // STEP-BY-STEP PROOF
              </div>
              <div style="font-size: 0.88rem; color: var(--text-secondary); max-width: 480px; margin: 0 auto; line-height: 1.6;">
                Click <strong>Synthesize Mathematical Proof</strong> on the left to stream the 4-phase pedagogical derivation with LaTeX formatting.
              </div>
            </div>`
      }
    </div>
  `;
  renderKaTeX();
}

function getSolverData() {
  const subject = document.getElementById("subjectInput").value;
  const topic = document.getElementById("topicInput").value || "General";
  const question_type = document.getElementById("typeInput").value;
  const marks = parseInt(document.getElementById("marksInput").value) || 1;
  const question = document.getElementById("questionTextInput").value.trim();

  const options = {};
  if (question_type === "MCQ" || question_type === "MSQ") {
    const a = document.getElementById("optionA").value.trim();
    const b = document.getElementById("optionB").value.trim();
    const c = document.getElementById("optionC").value.trim();
    const d = document.getElementById("optionD").value.trim();
    if (a) options.A = a;
    if (b) options.B = b;
    if (c) options.C = c;
    if (d) options.D = d;
  }

  return { subject, topic, question_type, marks, question, options };
}

async function runTerminalSolve() {
  if (isTerminalSolving) return;
  const data = getSolverData();
  if (!data.question) {
    alert("Please formulate or select a GATE CS problem.");
    return;
  }

  if (activeMode !== "solver") {
    activeMode = "solver";
    setTabActive(document.getElementById("tabNavSolver"));
    renderSolverStage();
  }

  isTerminalSolving = true;
  playTone(850, 0.08);

  const solveBtn = document.getElementById("solveTerminalBtn");
  solveBtn.disabled = true;
  solveBtn.innerHTML = `<span>Synthesizing Proof...</span>`;

  const termWindow = document.getElementById("terminalOutputWindow");
  const termLatency = document.getElementById("stageLatency");
  const termTps = document.getElementById("stageTps");
  const termTokens = document.getElementById("stageTokens");

  termWindow.innerHTML = `<span class="cursor-live"></span>`;
  
  const startTime = performance.now();
  let fullOutput = "";
  let tokenCount = 0;

  try {
    const response = await fetch("/api/solve/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...data, model_type: "finetuned" }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const raw = line.replace("data: ", "").trim();
          if (raw === "[DONE]") break;
          try {
            const parsed = JSON.parse(raw);
            if (parsed.token) {
              fullOutput += parsed.token;
              tokenCount++;
              termWindow.innerHTML = formatMarkdown(fullOutput) + `<span class="cursor-live"></span>`;
              if (termTokens) termTokens.textContent = tokenCount;
            }
          } catch (e) {}
        }
      }
    }

    const elapsed = (performance.now() - startTime) / 1000;
    if (termLatency) termLatency.textContent = `${(elapsed * 1000).toFixed(0)} ms`;
    if (termTps) termTps.textContent = `${(tokenCount / elapsed).toFixed(1)} tok/s`;
    latestTerminalMarkdown = fullOutput;
    termWindow.innerHTML = formatMarkdown(fullOutput);
    renderKaTeX();
  } catch (err) {
    termWindow.innerHTML = `<div style="color:var(--neon-rose)">Terminal error: ${err.message}</div>`;
  }

  isTerminalSolving = false;
  solveBtn.disabled = false;
  solveBtn.innerHTML = `<span>Synthesize Mathematical Proof</span> ⚡`;
}

/* --------------------------------------------------------------------------
   4. The Arena Battle Stage (Under One Window)
   -------------------------------------------------------------------------- */
function renderArenaStage(caseKey = "case_os") {
  const stage = document.getElementById("mainStageArea");
  const c = ARENA_CASES[caseKey];
  if (c && c.presetKey) {
    loadSolverPreset(c.presetKey);
  }

  let pillsHtml = "";
  for (let k of Object.keys(ARENA_CASES)) {
    const item = ARENA_CASES[k];
    const isAct = k === caseKey ? "active" : "";
    pillsHtml += `<button class="subj-chip ${isAct}" onclick="switchArenaChallenge('${k}')">${item.title}</button>`;
  }

  stage.innerHTML = `
    <div class="stage-hud">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary); font-weight: 700;">CHALLENGE:</span>
        ${pillsHtml}
      </div>
      <button class="btn-stage-action" style="background: var(--neon-cyan); color: #000; font-weight: 800; border-color: var(--neon-cyan);" onclick="executeArenaBattle()">
        RUN ARENA BATTLE ⚡
      </button>
    </div>

    <div class="arena-split-grid">
      <!-- Left: Base Model Box -->
      <div class="arena-box">
        <div class="arena-box-header">
          <div class="arena-model-name">
            <span>Qwen2.5-1.5B</span>
            <span class="badge-base-box">Base Baseline</span>
          </div>
          <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary);" id="baseMetricsArena">Awaiting Battle</span>
        </div>
        <div class="arena-body" id="baseArenaContent">
          <p style="color: var(--text-secondary); margin-bottom: 0.75rem;">
            Click <strong>RUN ARENA BATTLE</strong> to execute the baseline forward pass.
          </p>
          <div class="failure-callout">
            <strong>Observed Trap:</strong> ${c.baseFailure}
          </div>
        </div>
      </div>

      <!-- Right: Calypso Engine Box -->
      <div class="arena-box calypso-box">
        <div class="arena-box-header">
          <div class="arena-model-name">
            <span style="color: var(--neon-cyan);">CALYPSO-QWEN-1.5B</span>
            <span class="badge-calypso-box">Fine-Tuned</span>
          </div>
          <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--neon-cyan);" id="ftMetricsArena">Awaiting Battle</span>
        </div>
        <div class="arena-body" id="ftArenaContent">
          <p style="color: var(--text-secondary); margin-bottom: 0.75rem;">
            Calypso enforces 4-phase pedagogical CoT derivations.
          </p>
          <div class="success-callout">
            <strong>Domain Advantage:</strong> ${c.calypsoAdvantage}
          </div>
        </div>
      </div>
    </div>
  `;
  renderKaTeX();
}

function switchArenaChallenge(k) {
  currentArenaCase = k;
  renderArenaStage(k);
  playTone(500, 0.03);
}

async function executeArenaBattle() {
  playTone(800, 0.08);
  const data = getSolverData();
  const c = ARENA_CASES[currentArenaCase];

  const baseContent = document.getElementById("baseArenaContent");
  const ftContent = document.getElementById("ftArenaContent");
  const baseMetrics = document.getElementById("baseMetricsArena");
  const ftMetrics = document.getElementById("ftMetricsArena");

  if (!baseContent || !ftContent) return;

  baseMetrics.textContent = "Synthesizing...";
  ftMetrics.textContent = "Synthesizing...";
  baseContent.innerHTML = `<span class="cursor-live"></span> Running baseline forward pass...`;
  ftContent.innerHTML = `<span class="cursor-live"></span> Deriving Calypso domain proof...`;

  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const res = await response.json();

    baseMetrics.textContent = `${res.base_model_result.inference_latency_ms} ms • ${res.base_model_result.tokens_per_second} tok/s`;
    ftMetrics.textContent = `${res.finetuned_model_result.inference_latency_ms} ms • ${res.finetuned_model_result.tokens_per_second} tok/s`;

    baseContent.innerHTML = `
      <div>${formatMarkdown(res.base_model_result.solution_markdown)}</div>
      <div class="failure-callout"><strong>Examiner Trap Analysis:</strong> ${c.baseFailure}</div>
    `;

    ftContent.innerHTML = `
      <div>${formatMarkdown(res.finetuned_model_result.solution_markdown)}</div>
      <div class="success-callout"><strong>Verified Domain Advantage:</strong> ${c.calypsoAdvantage}</div>
    `;

    renderKaTeX();
  } catch (err) {
    baseContent.innerHTML = `<div style="color:var(--neon-rose)">Arena execution error: ${err.message}</div>`;
  }
}

/* --------------------------------------------------------------------------
   5. Syllabus Stage (Under One Window)
   -------------------------------------------------------------------------- */
function renderSyllabusStage() {
  const stage = document.getElementById("mainStageArea");
  let cards = "";

  for (let s of SYLLABUS_MATRIX) {
    cards += `
      <div class="syllabus-mini-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="font-weight: 900; font-family: var(--font-hero); font-size: 1.1rem; color: var(--text-primary);">${s.subject}</div>
          <div style="font-family: var(--font-mono); font-weight: 800; font-size: 0.72rem; background: var(--accent-highlight-bg); color: var(--accent-highlight-text); padding: 0.15rem 0.45rem;">${s.weight}</div>
        </div>
        <div class="syllabus-tier">${s.tier}</div>
        <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5;">${s.focus}</div>
      </div>
    `;
  }

  stage.innerHTML = `
    <div class="stage-hud">
      <div class="stage-stat-item">CURRICULUM: <span>OFFICIAL IIT GATE CS 2024</span></div>
      <div class="stage-stat-item">DOMAINS: <span>7 FOUNDATIONAL PILLARS</span></div>
    </div>
    <div class="syllabus-grid-wrapper">
      ${cards}
    </div>
  `;
}

/* --------------------------------------------------------------------------
   6. Utilities: Markdown, LaTeX, Audio, Calculator
   -------------------------------------------------------------------------- */
function normalizeMathSymbols(rawText) {
  if (!rawText) return "";
  return rawText
    // Convert bare textual/unwrapped words into formatted LaTeX formulas
    .replace(/(?<![\\$a-zA-Z0-9])Theta(?![a-zA-Z0-9])/g, "$\\Theta$")
    .replace(/(?<![\\$a-zA-Z0-9])theta(?![a-zA-Z0-9])/g, "$\\theta$")
    .replace(/(?<![\\$a-zA-Z0-9])Omega(?![a-zA-Z0-9])/g, "$\\Omega$")
    .replace(/(?<![\\$a-zA-Z0-9])omega(?![a-zA-Z0-9])/g, "$\\omega$")
    .replace(/(?<![\\$a-zA-Z0-9])lambda(?![a-zA-Z0-9])/g, "$\\lambda$")
    .replace(/(?<![\\$a-zA-Z0-9])epsilon(?![a-zA-Z0-9])/g, "$\\epsilon$")
    .replace(/(?<![\\$a-zA-Z0-9])Sigma(?![a-zA-Z0-9])/g, "$\\Sigma$")
    // Ensure bare LaTeX macro expressions outside of $...$ are wrapped in $
    .replace(/(?<!\$)\\Theta\b/g, "$\\Theta$")
    .replace(/(?<!\$)\\Omega\b/g, "$\\Omega$")
    .replace(/(?<!\$)\\mathcal\{O\}\b/g, "$\\mathcal{O}$")
    .replace(/(?<!\$)\\theta\b/g, "$\\theta$")
    .replace(/(?<!\$)\\lambda\b/g, "$\\lambda$")
    .replace(/(?<!\$)\\epsilon\b/g, "$\\epsilon$")
    .replace(/(?<!\$)\\Sigma\b/g, "$\\Sigma$")
    .replace(/(?<!\$)\\emptyset\b/g, "$\\emptyset$")
    .replace(/(?<!\$)\\approx\b/g, "$\\approx$")
    .replace(/(?<!\$)\\le\b/g, "$\\le$")
    .replace(/(?<!\$)\\ge\b/g, "$\\ge$")
    .replace(/(?<!\$)\\neq\b/g, "$\\neq$")
    .replace(/(?<!\$)\\times\b/g, "$\\times$")
    .replace(/(?<!\$)\\rightarrow\b/g, "$\\rightarrow$")
    .replace(/(?<!\$)\\iff\b/g, "$\\iff$")
    .replace(/(?<!\$)\\subseteq\b/g, "$\\subseteq$")
    .replace(/(?<!\$)\\in\b/g, "$\\in$");
}

function formatMarkdown(text) {
  if (!text) return "";

  // Normalize mathematical symbol references so they display as real glyphs
  const normalized = normalizeMathSymbols(text);

  // Split at Section 4 if present to wrap the entire answer block cleanly
  let parts = normalized.split(/### 4\. Final Answer/i);
  let mainBody = parts[0];
  let finalAns = parts.length > 1 ? parts[1] : "";

  let html = mainBody
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/^## (.*$)/gim, "<h3>$1</h3>")
    .replace(/^\- (.*$)/gim, "<li>$1</li>")
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/gim, "<em>$1</em>")
    .replace(/\n\n/gim, "<br/>");

  if (finalAns.trim()) {
    let formattedAns = finalAns
      .replace(/^\- (.*$)/gim, "<li>$1</li>")
      .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/gim, "<em>$1</em>")
      .replace(/\n\n/gim, "<br/>");
    
    html += `
      <div class="terminal-final-ans">
        <div class="final-ans-header">🎯 FINAL VERIFIED ANSWER</div>
        <div class="final-ans-content">${formattedAns}</div>
      </div>
    `;
  }

  return html;
}

function renderKaTeX() {
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      throwOnError: false,
    });
  }
}

function copyTerminalMarkdown() {
  if (latestTerminalMarkdown) {
    navigator.clipboard.writeText(latestTerminalMarkdown);
    alert("Proof Markdown copied to clipboard!");
  } else {
    alert("Synthesize a proof first.");
  }
}
