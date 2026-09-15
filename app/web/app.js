const form = document.querySelector("#research-form");
const query = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const emptyState = document.querySelector("#empty-state");
const results = document.querySelector("#results");
const summary = document.querySelector("#run-summary");
let pollTimer = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = query.value.trim();
  if (value.length < 3) return;
  setLoading(true);
  try {
    const response = await fetch("/research", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({query:value}) });
    const run = await response.json();
    if (!response.ok) throw new Error(run.detail || "研究请求失败");
    renderRun(run);
    beginPolling(run.run_id);
  } catch (error) {
    document.querySelector("#form-hint").textContent = `请求失败：${error.message}`;
  } finally { setLoading(false); }
});

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("is-active", item === tab));
  document.querySelectorAll("[data-result-panel]").forEach((panel) => panel.classList.toggle("is-hidden", panel.dataset.resultPanel !== tab.dataset.panel));
}));

function setLoading(active) { submitButton.disabled = active; submitButton.textContent = active ? "创建任务…" : "开始研究"; document.querySelector("#form-hint").textContent = active ? "正在创建可追踪的研究任务" : "建议聚焦一个企业或一个决策问题"; }
function text(node, value) { node.textContent = value || "—"; }
function duration(run) { const span = run.trace.find((item) => item.name === "research.run"); if (span) return `${(span.duration_ms / 1000).toFixed(1)} s`; return `${((Date.now() - new Date(run.created_at).getTime()) / 1000).toFixed(1)} s`; }
function statusLabel(status) { return ({pending:"等待执行", running:"研究中", completed:"已完成", failed:"执行失败"})[status] || status; }
function beginPolling(runId) { if (pollTimer) clearTimeout(pollTimer); const poll = async () => { try { const response = await fetch(`/runs/${runId}`); const run = await response.json(); if (!response.ok) throw new Error(run.detail || "无法获取研究进度"); renderRun(run); if (run.status === "pending" || run.status === "running") { pollTimer = setTimeout(poll, 1200); } else { pollTimer = null; } } catch (error) { document.querySelector("#form-hint").textContent = `进度刷新失败：${error.message}`; pollTimer = null; } }; pollTimer = setTimeout(poll, 300); }
function renderRun(run) {
  emptyState.classList.add("is-hidden"); results.classList.remove("is-hidden"); summary.classList.remove("is-hidden");
  text(document.querySelector("#run-id"), run.run_id); const status = document.querySelector("#run-status"); text(status, statusLabel(run.status)); status.className = `status ${run.status}`;
  text(document.querySelector("#evidence-count"), String(run.evidence.length)); text(document.querySelector("#run-duration"), duration(run)); text(document.querySelector("#report-goal"), run.goal || "正在生成研究计划");
  const error = document.querySelector("#run-error"); error.textContent = run.error ? `运行失败：${run.error}` : ""; error.classList.toggle("is-hidden", !run.error);
  renderReport(run.report || run.error || "暂无报告。"); renderTasks(run.tasks); renderEvidence(run.evidence); renderTrace(run.trace);
}
function renderReport(value) { const target = document.querySelector("#report-content"); target.replaceChildren(); value.split("\n").forEach((line) => { const node = line.startsWith("###") ? document.createElement("h3") : document.createElement("p"); node.textContent = line.replace(/^###\s*/, ""); target.append(node); }); }
function renderTasks(tasks) { const target = document.querySelector("#task-list"); target.replaceChildren(); tasks.forEach((task) => { const item=document.createElement("li"), title=document.createElement("strong"), meta=document.createElement("div"), badge=document.createElement("span"), result=document.createElement("p"); title.textContent=task.description; badge.className=`badge ${task.status}`; badge.textContent=task.status; meta.className="task-meta"; meta.append("Task " + task.task_id, badge); result.className="task-result"; result.textContent=task.result || task.error || "等待执行"; item.append(title,meta,result); target.append(item); }); }
function renderEvidence(evidence) { const target=document.querySelector("#evidence-list"); target.replaceChildren(); if (!evidence.length) { target.textContent="本次运行没有采集到可用证据。"; return; } evidence.forEach((item) => { const card=document.createElement("article"), title=document.createElement("h3"), content=document.createElement("p"), link=document.createElement("a"); card.className="evidence"; title.textContent=`[${item.evidence_id}] ${item.title}`; content.textContent=item.content; link.href=item.url; link.target="_blank"; link.rel="noreferrer"; link.textContent=item.url || "无 URL"; card.append(title,content,link); target.append(card); }); }
function renderTrace(trace) { const target=document.querySelector("#trace-list"); target.replaceChildren(); trace.forEach((item) => { const row=document.createElement("tr"); const details=Object.entries(item.metadata || {}).map(([key,value])=>`${key}=${value}`).join(", "); [item.name,item.component,item.task_id ?? "—",`${item.duration_ms} ms`,item.status,details || item.error || "—"].forEach((value,index)=>{ const cell=document.createElement("td"); cell.textContent=value; if (index===4 && item.status==="error") cell.className="trace-error"; row.append(cell); }); target.append(row); }); }
