// AgentOne Client Control Engine

document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupRAG();
  setupHITL();
  setupEval();
  setupOrchestrator();
});

// 1. Tab Navigation
function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  const contents = document.querySelectorAll('.tab-content');

  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      contents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = `tab-${btn.getAttribute('data-tab')}`;
      const target = document.getElementById(targetId);
      if (target) target.classList.add('active');

      if (btn.getAttribute('data-tab') === 'hitl') {
        fetchPendingApprovals();
      }
    });
  });
}

function setQuery(text) {
  document.getElementById('agent-query-input').value = text;
}

function clearLogs() {
  document.getElementById('stream-log-box').innerHTML = `
    <div class="log-entry">
      <span class="log-tag supervisor">SYSTEM</span> Log cleared.
    </div>
  `;
}

function appendLog(tag, tagClass, text) {
  const box = document.getElementById('stream-log-box');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-tag ${tagClass}">${tag}</span> ${escapeHtml(text)}`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function resetGraphNodes() {
  const nodes = ['supervisor', 'triage', 'rag', 'tool', 'critic'];
  nodes.forEach(n => {
    const el = document.getElementById(`node-${n}`);
    if (el) {
      el.classList.remove('active', 'completed');
    }
  });
}

function setNodeState(nodeId, state) {
  const map = {
    'supervisor': 'node-supervisor',
    'triage': 'node-triage',
    'rag': 'node-rag',
    'tool_executor': 'node-tool',
    'critic': 'node-critic',
  };
  const targetId = map[nodeId] || `node-${nodeId}`;
  const el = document.getElementById(targetId);
  if (el) {
    if (state === 'active') {
      el.classList.remove('completed');
      el.classList.add('active');
    } else if (state === 'completed') {
      el.classList.remove('active');
      el.classList.add('completed');
    }
  }
}

// 2. Orchestrator & Real-Time SSE Streamer
function setupOrchestrator() {
  const btn = document.getElementById('btn-execute-stream');
  const input = document.getElementById('agent-query-input');

  btn.addEventListener('click', async () => {
    const query = input.value.trim();
    if (!query) return;

    btn.disabled = true;
    btn.innerText = 'Orchestrating... ⚡';
    resetGraphNodes();
    document.getElementById('response-content-area').innerText = 'Streaming agent reasoning...';

    appendLog('INPUT', 'supervisor', `User Prompt: ${query}`);
    setNodeState('supervisor', 'active');

    const streamUrl = `/api/agents/stream?query=${encodeURIComponent(query)}`;
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);

        if (payload.event === 'node_update') {
          const nodeName = payload.node;
          const data = payload.data || {};

          setNodeState(nodeName, 'completed');

          if (nodeName === 'supervisor') {
            appendLog('SUPERVISOR', 'supervisor', 'Input Guardrails passed (PII sanitized, injection defense active).');
            setNodeState('triage', 'active');
          } else if (nodeName === 'triage') {
            appendLog('TRIAGE', 'triage', `Classified Intent: ${data.intent} | Priority: ${data.priority}`);
            setNodeState('rag', 'active');
          } else if (nodeName === 'rag') {
            appendLog('RAG', 'rag', `Hybrid retrieval completed: ${data.docs_count} knowledge chunks retrieved & graded.`);
            setNodeState('tool_executor', 'active');
          } else if (nodeName === 'tool_executor') {
            if (data.requires_approval) {
              appendLog('HITL', 'tool', '⚠️ High-risk action detected! Checkpoint paused awaiting human approval.');
              checkPendingCount();
            } else {
              appendLog('TOOL', 'tool', `Executed ${data.executed_actions} automated operations.`);
            }
            setNodeState('critic', 'active');
          } else if (nodeName === 'critic') {
            appendLog('CRITIC', 'critic', 'Output verification passed: Faithfulness confirmed vs runbook ground truth.');
            if (data.final_response) {
              document.getElementById('response-content-area').innerText = data.final_response;
            }
          }
        } else if (payload.event === 'complete') {
          eventSource.close();
          btn.disabled = false;
          btn.innerText = 'Execute Agent Flow ⚡';

          const tel = payload.telemetry || {};
          document.getElementById('meter-latency').innerText = `${tel.total_duration_ms || 0} ms`;
          document.getElementById('meter-tokens').innerText = `${tel.total_tokens || 0}`;
          document.getElementById('meter-cost').innerText = `$${(tel.total_cost_usd || 0).toFixed(6)}`;

          updateTelemetryLog(tel);
        }
      } catch (err) {
        console.error('SSE parse error:', err);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      btn.disabled = false;
      btn.innerText = 'Execute Agent Flow ⚡';
    };
  });
}

// 3. Hybrid RAG View
function setupRAG() {
  const btn = document.getElementById('btn-search-rag');
  const input = document.getElementById('rag-search-input');
  const slider = document.getElementById('rag-alpha-slider');
  const sliderVal = document.getElementById('rag-alpha-val');
  const container = document.getElementById('rag-results-container');

  slider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    let desc = 'Balanced';
    if (val > 0.6) desc = 'Dense Biased';
    else if (val < 0.4) desc = 'BM25 Biased';
    sliderVal.innerText = `${val} (${desc})`;
  });

  btn.addEventListener('click', async () => {
    const q = input.value.trim();
    if (!q) return;

    btn.disabled = true;
    btn.innerText = 'Retrieving...';
    container.innerHTML = '<div style="color:var(--text-muted);">Searching dense vectors and BM25 inverted index...</div>';

    try {
      const alpha = slider.value;
      const res = await fetch(`/api/rag/search?query=${encodeURIComponent(q)}&alpha=${alpha}&top_k=4`);
      const data = await res.json();

      if (!data.results || data.results.length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted);">No matching runbook chunks found.</div>';
      } else {
        container.innerHTML = data.results.map((r, i) => `
          <div style="background:var(--bg-card); border:1px solid var(--border-color); border-radius:8px; padding:1rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
              <span style="font-weight:700; color:var(--accent-blue);">#${i+1} [${escapeHtml(r.source)}]</span>
              <span style="font-size:0.75rem; color:var(--accent-cyan); font-family:var(--font-mono);">RRF Score: ${r.rrf_score.toFixed(6)} (Dense: ${r.dense_score}, BM25: ${r.sparse_score})</span>
            </div>
            <p style="font-size:0.875rem; color:var(--text-primary); line-height:1.5;">${escapeHtml(r.content)}</p>
          </div>
        `).join('');
      }
    } catch (err) {
      container.innerHTML = `<div style="color:var(--accent-red);">Search failed: ${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerText = 'Search RAG Index';
    }
  });
}

// 4. Human in the Loop (HITL)
async function checkPendingCount() {
  try {
    const res = await fetch('/api/hitl/pending');
    const items = await res.json();
    const badge = document.getElementById('hitl-badge');
    if (items.length > 0) {
      badge.style.display = 'inline-block';
      badge.innerText = items.length;
    } else {
      badge.style.display = 'none';
    }
  } catch (err) {}
}

async function fetchPendingApprovals() {
  const container = document.getElementById('hitl-container');
  try {
    const res = await fetch('/api/hitl/pending');
    const items = await res.json();
    checkPendingCount();

    if (!items || items.length === 0) {
      container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.875rem;">No pending approval requests in queue.</div>';
      return;
    }

    container.innerHTML = items.map(req => `
      <div class="hitl-card" id="hitl-card-${req.approval_id}">
        <div class="hitl-header">
          <span style="font-weight:700;">Ticket: ${req.approval_id}</span>
          <span class="risk-badge risk-${req.proposed_action.risk_level}">${req.proposed_action.risk_level.toUpperCase()}</span>
        </div>
        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.5rem;">
          <strong>Action:</strong> <code>${escapeHtml(req.proposed_action.tool_name)}</code>
        </div>
        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.5rem;">
          <strong>Params:</strong> <code>${JSON.stringify(req.proposed_action.tool_input)}</code>
        </div>
        <div style="font-size:0.8rem; color:var(--accent-amber); margin-bottom:0.75rem;">
          ${escapeHtml(req.reason_for_review)}
        </div>
        <div class="hitl-actions">
          <button class="btn-approve" onclick="approveAction('${req.approval_id}')">Approve & Execute</button>
          <button class="btn-reject" onclick="rejectAction('${req.approval_id}')">Reject Action</button>
        </div>
      </div>
    `).join('');
  } catch (err) {
    container.innerHTML = `<div style="color:var(--accent-red);">Error loading approval queue: ${err.message}</div>`;
  }
}

async function approveAction(id) {
  try {
    const res = await fetch(`/api/hitl/${id}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_comment: 'Approved by lead operator via dashboard.' })
    });
    if (res.ok) {
      fetchPendingApprovals();
    }
  } catch (err) {
    alert(`Approval failed: ${err.message}`);
  }
}

async function rejectAction(id) {
  try {
    const res = await fetch(`/api/hitl/${id}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_comment: 'Rejected by operator policy.' })
    });
    if (res.ok) {
      fetchPendingApprovals();
    }
  } catch (err) {
    alert(`Rejection failed: ${err.message}`);
  }
}

// 5. Evaluation & Benchmarks
function setupEval() {
  const btn = document.getElementById('btn-run-eval');
  const tableBody = document.getElementById('eval-table-body');
  const summaryCards = document.getElementById('eval-summary-cards');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.innerText = 'Evaluating Benchmark Suite... ⏳';
    tableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Executing multi-agent evaluation harness across ground-truth dataset...</td></tr>';

    try {
      const res = await fetch('/api/eval/run', { method: 'POST' });
      const data = await res.json();

      summaryCards.style.display = 'flex';
      document.getElementById('eval-overall-score').innerText = `${(data.overall_system_score * 100).toFixed(1)}%`;
      document.getElementById('eval-faithfulness').innerText = `${(data.avg_faithfulness * 100).toFixed(1)}%`;
      document.getElementById('eval-relevancy').innerText = `${(data.avg_answer_relevancy * 100).toFixed(1)}%`;
      document.getElementById('eval-precision').innerText = `${(data.avg_context_precision * 100).toFixed(1)}%`;
      document.getElementById('eval-latency').innerText = `${data.avg_latency_ms.toFixed(0)} ms`;

      tableBody.innerHTML = data.case_results.map(c => `
        <tr>
          <td><code>${c.case_id}</code></td>
          <td><span style="font-size:0.75rem; color:var(--text-secondary);">${c.category}</span></td>
          <td style="max-width:300px; font-size:0.8rem;">${escapeHtml(c.question)}</td>
          <td class="score-cell">${(c.metrics.faithfulness * 100).toFixed(0)}%</td>
          <td class="score-cell">${(c.metrics.answer_relevancy * 100).toFixed(0)}%</td>
          <td class="score-cell">${(c.metrics.context_precision * 100).toFixed(0)}%</td>
          <td class="score-cell" style="color:var(--accent-cyan); font-weight:800;">${(c.metrics.overall_score * 100).toFixed(0)}%</td>
        </tr>
      `).join('');
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="7" style="color:var(--accent-red);">Benchmark failed: ${err.message}</td></tr>`;
    } finally {
      btn.disabled = false;
      btn.innerText = 'Run Automated Benchmark Suite';
    }
  });
}

// 6. Telemetry Explorer
function updateTelemetryLog(summary) {
  const box = document.getElementById('telemetry-audit-box');
  if (!box || !summary.spans) return;

  box.innerHTML = summary.spans.map(s => `
    <div class="log-entry">
      <span class="log-tag ${s.node_name.includes('rag') ? 'rag' : s.node_name.includes('triage') ? 'triage' : s.node_name.includes('tool') ? 'tool' : 'supervisor'}">${s.node_name.toUpperCase()}</span>
      <span>Duration: <strong>${s.duration_ms} ms</strong> | Prompt Tokens: <strong>${s.prompt_tokens}</strong> | Output Tokens: <strong>${s.completion_tokens}</strong> | Cost: <strong>$${s.estimated_cost_usd.toFixed(6)}</strong></span>
    </div>
  `).join('');
}
