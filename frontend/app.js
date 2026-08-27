/**
 * CALYPSO 2.0 Client Application
 * Features: 4-Phase CoT Rendering, KaTeX Math, OCR Ingestion, Multi-Turn Chat, Python Sandbox.
 */

// State
let activeTab = "solver";
let currentOcrBase64 = null;
let chatMessages = [];
let isGenerating = false;

// Presets Dictionary
const PRESETS = {
  algo_recurrence: {
    subject: "Algorithms",
    topic: "Asymptotic Analysis",
    type: "MCQ",
    marks: 2,
    question: "Consider the recurrence relation $T(n) = 2T(\\lfloor\\sqrt{n}\\rfloor) + \\log_2 n$ with $T(2) = 1$. What is the asymptotic order of $T(n)$?",
    options: {
      A: "$\\Theta((\\log_2 n) \\cdot (\\log_2 \\log_2 n))$",
      B: "$\\Theta(\\log_2 n)$",
      C: "$\\Theta((\\log_2 n)^2)$",
      D: "$\\Theta(n \\log_2 n)$",
    },
  },
  os_paging: {
    subject: "Operating Systems",
    topic: "Virtual Memory & Paging",
    type: "NAT",
    marks: 2,
    question: "A system uses 2-level paging. Main memory access time is $100\\text{ ns}$ and TLB access time is $10\\text{ ns}$. If the TLB hit ratio is $90\\%$, and on a TLB miss, all page table levels are in main memory, what is the Effective Memory Access Time (EMAT) in nanoseconds?",
    options: {},
  },
  dbms_norm: {
    subject: "DBMS",
    topic: "Relational Normalization",
    type: "MCQ",
    marks: 2,
    question: "Given a relation $R(A, B, C, D)$ with functional dependencies $FD = \\{AB \\rightarrow C, C \\rightarrow D, D \\rightarrow A\\}$. What is the highest normal form satisfied by $R$?",
    options: {
      A: "3NF but not BCNF",
      B: "BCNF",
      C: "2NF but not 3NF",
      D: "1NF only",
    },
  },
  cn_subnet: {
    subject: "Computer Networks",
    topic: "IPv4 Subnetting",
    type: "MCQ",
    marks: 2,
    question: "An ISP assigns the block $200.10.0.0/16$ to an organization. The organization wants to create subnets having up to 60 usable hosts each. What is the maximum number of such subnets that can be formed and what is the subnet mask?",
    options: {
      A: "$1024$ subnets with mask $255.255.255.192$",
      B: "$1024$ subnets with mask $255.255.255.128$",
      C: "$512$ subnets with mask $255.255.255.192$",
      D: "$2048$ subnets with mask $255.255.255.224$",
    },
  },
  toc_decide: {
    subject: "Theory of Computation",
    topic: "Decidability & CFGs",
    type: "MSQ",
    marks: 2,
    question: "Which of the following decision problems are UNDECIDABLE for Context-Free Grammars (CFGs)?",
    options: {
      A: "Is $L(G) = \\Sigma^*$ (Universality)?",
      B: "Is $L(G_1) \\cap L(G_2) = \\emptyset$ (Disjointness)?",
      C: "Is $L(G) = \\emptyset$ (Emptiness)?",
      D: "Is $G$ ambiguous?",
    },
  },
  cd_lalr: {
    subject: "Compiler Design",
    topic: "Bottom-Up Parsing",
    type: "MCQ",
    marks: 2,
    question: "When constructing an LALR(1) parser by merging LR(1) states with identical cores, which type of parsing conflict can potentially be introduced?",
    options: {
      A: "Reduce-Reduce (RR) conflict only",
      B: "Shift-Reduce (SR) conflict only",
      C: "Both Shift-Reduce and Reduce-Reduce conflicts",
      D: "Neither SR nor RR conflict",
    },
  },
};

// DOM Elements
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initTheme();
  initPresets();
  initOcrDropzone();
  initSolver();
  initChat();
  initArena();
  initPythonSandbox();
});

// 1. Tab Navigation
function initTabs() {
  const tabs = [
    { btn: "tabNavSolver", view: "solverTerminalView", name: "solver" },
    { btn: "tabNavChat", view: "chatTerminalView", name: "chat" },
    { btn: "tabNavArena", view: "arenaTerminalView", name: "arena" },
    { btn: "tabNavBench", view: "benchTerminalView", name: "bench" },
  ];

  tabs.forEach(({ btn, view, name }) => {
    const el = document.getElementById(btn);
    if (!el) return;
    el.addEventListener("click", () => {
      tabs.forEach((t) => {
        document.getElementById(t.btn)?.classList.remove("active");
        const v = document.getElementById(t.view);
        if (v) v.style.display = "none";
      });
      el.classList.add("active");
      const currentView = document.getElementById(view);
      if (currentView) currentView.style.display = "block";
      activeTab = name;
    });
  });
}

// 2. Theme Toggle
function initTheme() {
  const btn = document.getElementById("themeToggle");
  const icon = document.getElementById("themeIcon");
  const text = document.getElementById("themeText");

  btn?.addEventListener("click", () => {
    const isLight = document.body.classList.toggle("theme-light");
    if (icon) icon.textContent = isLight ? "🌙" : "☀️";
    if (text) text.textContent = isLight ? "DARK" : "LIGHT";
  });
}

// 3. Presets & Domain Selector
function initPresets() {
  document.querySelectorAll(".subj-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".subj-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const subj = chip.getAttribute("data-subj");
      const sel = document.getElementById("subjectInput");
      if (sel && subj) sel.value = subj;
    });
  });

  document.querySelectorAll(".preset-btn-card").forEach((card) => {
    card.addEventListener("click", () => {
      const key = card.getAttribute("data-preset");
      const p = PRESETS[key];
      if (!p) return;

      document.getElementById("subjectInput").value = p.subject;
      document.getElementById("topicInput").value = p.topic;
      document.getElementById("typeInput").value = p.type;
      document.getElementById("marksInput").value = p.marks;
      document.getElementById("questionTextInput").value = p.question;

      document.getElementById("optionA").value = p.options.A || "";
      document.getElementById("optionB").value = p.options.B || "";
      document.getElementById("optionC").value = p.options.C || "";
      document.getElementById("optionD").value = p.options.D || "";

      // Select corresponding chip
      document.querySelectorAll(".subj-chip").forEach((c) => {
        if (c.getAttribute("data-subj") === p.subject) c.classList.add("active");
        else c.classList.remove("active");
      });
    });
  });
}

// 4. Screenshot / Image OCR
function initOcrDropzone() {
  const dropzone = document.getElementById("ocrDropzone");
  const fileInput = document.getElementById("ocrFileInput");

  fileInput?.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });

  dropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone?.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  });

  // Global Clipboard Paste (Ctrl+V / Cmd+V)
  window.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          handleImageFile(file);
          break;
        }
      }
    }
  });
}

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = async (e) => {
    currentOcrBase64 = e.target.result;
    const preview = document.getElementById("ocrPreviewImg");
    const container = document.getElementById("ocrPreviewContainer");
    const promptText = document.getElementById("ocrPromptText");
    if (preview && container && promptText) {
      preview.src = currentOcrBase64;
      container.style.display = "block";
      promptText.style.display = "none";
    }

    // Auto-extract question parameters from image
    try {
      const resp = await fetch("/api/solve/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: currentOcrBase64 }),
      });
      const data = await resp.json();
      if (data.extracted_data) {
        const ext = data.extracted_data;
        if (ext.question) document.getElementById("questionTextInput").value = ext.question;
        if (ext.subject) document.getElementById("subjectInput").value = ext.subject;
        if (ext.question_type) document.getElementById("typeInput").value = ext.question_type;
        if (ext.options) {
          document.getElementById("optionA").value = ext.options.A || "";
          document.getElementById("optionB").value = ext.options.B || "";
          document.getElementById("optionC").value = ext.options.C || "";
          document.getElementById("optionD").value = ext.options.D || "";
        }
      }
    } catch (err) {
      console.log("OCR direct preview loaded.");
    }
  };
  reader.readAsDataURL(file);
}

function clearOcrImage() {
  currentOcrBase64 = null;
  const preview = document.getElementById("ocrPreviewImg");
  const container = document.getElementById("ocrPreviewContainer");
  const promptText = document.getElementById("ocrPromptText");
  const fileInput = document.getElementById("ocrFileInput");
  if (preview) preview.src = "";
  if (container) container.style.display = "none";
  if (promptText) promptText.style.display = "block";
  if (fileInput) fileInput.value = "";
}

// 5. Reasoning Solver
function initSolver() {
  const btn = document.getElementById("solveTerminalBtn");
  btn?.addEventListener("click", () => runSolverSynthesis());
}

async function runSolverSynthesis() {
  if (isGenerating) return;

  const subject = document.getElementById("subjectInput").value;
  const topic = document.getElementById("topicInput").value || "General";
  const questionType = document.getElementById("typeInput").value;
  const marks = parseInt(document.getElementById("marksInput").value) || 2;
  const question = document.getElementById("questionTextInput").value.trim();

  const options = {};
  const optA = document.getElementById("optionA").value.trim();
  const optB = document.getElementById("optionB").value.trim();
  const optC = document.getElementById("optionC").value.trim();
  const optD = document.getElementById("optionD").value.trim();
  if (optA) options["A"] = optA;
  if (optB) options["B"] = optB;
  if (optC) options["C"] = optC;
  if (optD) options["D"] = optD;

  if (!question && !currentOcrBase64) {
    alert("Please enter a question or upload a question image.");
    return;
  }

  isGenerating = true;
  const emptyState = document.getElementById("emptyStateMsg");
  const phasesContainer = document.getElementById("proofPhasesContainer");
  if (emptyState) emptyState.style.display = "none";
  if (phasesContainer) {
    phasesContainer.style.display = "block";
    phasesContainer.innerHTML = '<div class="loading-spinner">⚡ Synthesizing 4-Phase Mathematical Proof via Calypso...</div>';
  }

  // Switch to solver view if on another tab
  document.getElementById("tabNavSolver")?.click();

  const startTime = performance.now();

  try {
    let resp, data;
    if (currentOcrBase64 && !question) {
      resp = await fetch("/api/solve/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_data: currentOcrBase64,
          subject: subject,
          topic: topic,
        }),
      });
      const imgData = await resp.json();
      data = imgData.solution;
    } else {
      resp = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject,
          topic,
          question_type: questionType,
          marks,
          question,
          options,
          model_type: "finetuned",
        }),
      });
      data = await resp.json();
    }

    // Update Telemetry HUD
    document.getElementById("stageLatency").textContent = `${data.inference_latency_ms} ms`;
    document.getElementById("stageTps").textContent = `${data.tokens_per_second} tok/s`;
    document.getElementById("stageTokens").textContent = data.tokens_generated;

    // Render 4-Phase Cards
    renderStructuredPhases(data, question, subject, topic);

    // Add to chat history context
    chatMessages = [
      { role: "user", content: `Question (${subject} - ${topic}): ${question}` },
      { role: "assistant", content: data.solution_markdown },
    ];

  } catch (err) {
    if (phasesContainer) {
      phasesContainer.innerHTML = `<div class="error-msg">Error synthesizing proof: ${err.message}</div>`;
    }
  } finally {
    isGenerating = false;
  }
}

function renderStructuredPhases(data, question, subject, topic) {
  const container = document.getElementById("proofPhasesContainer");
  if (!container) return;

  const phases = data.structured_phases || {
    phase1_concept: "Core principles and formal syllabus anchors.",
    phase2_derivation: data.solution_markdown,
    phase3_elimination: "Candidate option analysis.",
    phase4_answer: data.extracted_answer || "Verified Answer.",
  };

  container.innerHTML = `
    <div class="proof-card">
      <div class="phase-header">
        <span class="phase-num">PHASE 1</span>
        <span class="phase-title">Conceptual Invariants & Core Principles</span>
      </div>
      <div class="phase-body markdown-render">${formatMarkdownContent(phases.phase1_concept)}</div>
    </div>

    <div class="proof-card">
      <div class="phase-header">
        <span class="phase-num">PHASE 2</span>
        <span class="phase-title">Step-by-Step Derivation & Calculations</span>
      </div>
      <div class="phase-body markdown-render">${formatMarkdownContent(phases.phase2_derivation)}</div>
    </div>

    <div class="proof-card">
      <div class="phase-header">
        <span class="phase-num">PHASE 3</span>
        <span class="phase-title">Option Elimination & Trap Analysis</span>
      </div>
      <div class="phase-body markdown-render">${formatMarkdownContent(phases.phase3_elimination)}</div>
    </div>

    <div class="proof-card answer-card">
      <div class="phase-header">
        <span class="phase-num">PHASE 4</span>
        <span class="phase-title">Verified GATE Answer</span>
      </div>
      <div class="phase-body markdown-render answer-highlight">
        ${formatMarkdownContent(phases.phase4_answer)}
      </div>
    </div>
  `;

  // Render LaTeX math via KaTeX
  if (window.renderMathInElement) {
    window.renderMathInElement(container, {
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

function formatMarkdownContent(text) {
  if (!text) return "";
  let html = text
    .replace(/^### (.*$)/gim, "<h4>$1</h4>")
    .replace(/^## (.*$)/gim, "<h3>$1</h3>")
    .replace(/^# (.*$)/gim, "<h2>$1</h2>")
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/gim, "<em>$1</em>")
    .replace(/`([^`]+)`/gim, "<code>$1</code>")
    .replace(/\n\n/gim, "<br><br>")
    .replace(/\n/gim, "<br>");
  return html;
}

// 6. Multi-Turn Interactive Doubt Chat
function initChat() {
  const sendBtn = document.getElementById("chatSendBtn");
  const input = document.getElementById("chatUserQuery");

  sendBtn?.addEventListener("click", () => sendChatMessage());
  input?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendChatMessage();
  });
}

async function sendChatMessage() {
  const input = document.getElementById("chatUserQuery");
  const text = input.value.trim();
  if (!text) return;

  const box = document.getElementById("chatHistoryBox");
  input.value = "";

  // Append user message
  const userMsgEl = document.createElement("div");
  userMsgEl.className = "chat-msg user-msg";
  userMsgEl.innerHTML = `<div class="chat-role-badge">YOU (STUDENT)</div><div class="chat-text">${text}</div>`;
  box.appendChild(userMsgEl);
  box.scrollTop = box.scrollHeight;

  chatMessages.push({ role: "user", content: text });

  // Bot loading placeholder
  const botMsgEl = document.createElement("div");
  botMsgEl.className = "chat-msg bot-msg";
  botMsgEl.innerHTML = `<div class="chat-role-badge">CALYPSO MENTOR</div><div class="chat-text loading-pulse">Thinking...</div>`;
  box.appendChild(botMsgEl);
  box.scrollTop = box.scrollHeight;

  try {
    const subject = document.getElementById("subjectInput")?.value || "Algorithms";
    const topic = document.getElementById("topicInput")?.value || "General";

    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatMessages,
        subject,
        topic,
      }),
    });
    const data = await resp.json();
    const replyText = data.reply || "Clarification synthesized.";

    botMsgEl.querySelector(".chat-text").innerHTML = formatMarkdownContent(replyText);
    chatMessages.push({ role: "assistant", content: replyText });

    if (window.renderMathInElement) {
      window.renderMathInElement(botMsgEl, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
      });
    }
  } catch (err) {
    botMsgEl.querySelector(".chat-text").textContent = `Error: ${err.message}`;
  }
  box.scrollTop = box.scrollHeight;
}

// 7. Arena Battle
function initArena() {
  const btn = document.getElementById("runArenaBtn");
  btn?.addEventListener("click", async () => {
    const subject = document.getElementById("subjectInput").value;
    const topic = document.getElementById("topicInput").value || "General";
    const question = document.getElementById("questionTextInput").value.trim();
    if (!question) {
      alert("Please formulate a question on the left dock first.");
      return;
    }

    const baseEl = document.getElementById("arenaBaseContent");
    const ftEl = document.getElementById("arenaFtContent");
    if (baseEl) baseEl.innerHTML = '<div class="loading-spinner">Evaluating base model...</div>';
    if (ftEl) ftEl.innerHTML = '<div class="loading-spinner">Synthesizing fine-tuned proof...</div>';

    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject,
          topic,
          question,
        }),
      });
      const data = await resp.json();
      if (baseEl) baseEl.innerHTML = formatMarkdownContent(data.base_model_result.solution_markdown);
      if (ftEl) ftEl.innerHTML = formatMarkdownContent(data.finetuned_model_result.solution_markdown);

      if (window.renderMathInElement) {
        window.renderMathInElement(document.getElementById("arenaTerminalView"), {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ],
          throwOnError: false,
        });
      }
    } catch (err) {
      if (baseEl) baseEl.textContent = `Error: ${err.message}`;
    }
  });
}

// 8. Python Sandbox Tool
function initPythonSandbox() {
  const runBtn = document.getElementById("executePythonBtn");
  runBtn?.addEventListener("click", async () => {
    const code = document.getElementById("sandboxPythonCode").value;
    const outBox = document.getElementById("sandboxPythonOutput");
    outBox.style.display = "block";
    outBox.innerHTML = "Executing in sandbox...";

    try {
      const resp = await fetch("/api/tools/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_name: "code_interpreter",
          code: code,
        }),
      });
      const res = await resp.json();
      if (res.success) {
        outBox.innerHTML = `<strong>Output (${res.elapsed_ms}ms):</strong><br><pre>${res.output || "(Code executed with no output)"}</pre>`;
      } else {
        outBox.innerHTML = `<span style="color:var(--accent-red)">Error: ${res.error}</span>`;
      }
    } catch (err) {
      outBox.innerHTML = `<span style="color:var(--accent-red)">Error: ${err.message}</span>`;
    }
  });
}

function openInteractivePythonRunner() {
  const m = document.getElementById("pythonModal");
  if (m) m.style.display = "flex";
}

function closeInteractivePythonRunner() {
  const m = document.getElementById("pythonModal");
  if (m) m.style.display = "none";
}

function copyTerminalMarkdown() {
  const container = document.getElementById("proofPhasesContainer");
  if (!container) return;
  navigator.clipboard.writeText(container.innerText).then(() => {
    alert("Proof Markdown copied to clipboard!");
  });
}

window.copyTerminalMarkdown = copyTerminalMarkdown;
window.openInteractivePythonRunner = openInteractivePythonRunner;
window.closeInteractivePythonRunner = closeInteractivePythonRunner;
window.clearOcrImage = clearOcrImage;
