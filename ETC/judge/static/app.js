"use strict";
const $ = id => document.getElementById(id);
const isDisplay = document.body.dataset.view === "display";
const query = new URLSearchParams(location.search);
const rankingKeys = ["1", "2", "3", "4"];
const rotationMs = 15000;
let board = null, page = 0, pageCount = 1, lastSuccess = 0;
let fetchingBoard = false, connectionFailed = false;
let autoRotate = query.has("rotate") ? query.get("rotate") === "1" : isDisplay && !matchMedia("(prefers-reduced-motion: reduce)").matches;
let nextRotation = Date.now() + rotationMs;
if (rankingKeys.includes(query.get("ranking"))) $("ranking").value = query.get("ranking");

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(path, {cache: "no-store", ...options, signal: controller.signal});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    return result;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The response timed out. Please check again shortly.");
    throw error;
  } finally { clearTimeout(timeout); }
}
function format(value) { return value === null || value === undefined ? "-" : value.toFixed(2); }
function stale() { return connectionFailed || (lastSuccess > 0 && Date.now() - lastSuccess > 15000); }
function rowsPerPage() {
  if (!isDisplay || window.innerWidth < 900) return 10;
  const top = document.querySelector("thead").getBoundingClientRect().bottom;
  const footer = document.querySelector(".board-footer").offsetHeight + document.querySelector(".footnote").offsetHeight + 50;
  const height = parseFloat(getComputedStyle(document.body).getPropertyValue("--row-height"));
  return Math.max(3, Math.min(10, Math.floor((window.innerHeight - top - footer) / height)));
}
function rememberView() {
  const url = new URL(location.href);
  url.search = "";
  url.searchParams.set("ranking", $("ranking").value);
  url.searchParams.set("rotate", autoRotate ? "1" : "0");
  history.replaceState(null, "", url);
}
function showLoading() {
  const key = $("ranking").value;
  board = null; page = 0; pageCount = 1; lastSuccess = 0; connectionFailed = false;
  $("challenge-title").textContent = `Challenge ${key}`;
  $("ranking-title").textContent = `Challenge ${key} standings`;
  $("ranking-description").textContent = "Best submission in this Challenge";
  $("score-heading").textContent = `C${key} score`;
  const tr = document.createElement("tr"), td = document.createElement("td");
  td.colSpan = 3; td.className = "empty";
  td.textContent = `Waiting for Challenge ${key} results...`;
  tr.append(td); $("standings").replaceChildren(tr);
  for (const status of ["people", "completed", "running", "queued"]) $("stat-" + status).textContent = "-";
  $("page-label").textContent = "Page 1 / 1";
  $("range-label").textContent = "Waiting for results";
  $("previous-page").disabled = true; $("next-page").disabled = true;
  updateConnection();
}
function drawBoard() {
  const key = $("ranking").value;
  if (!board || board.challenge !== key) return;
  const rank = row => row.challenge_ranks[key];
  const rows = [...board.participants].sort((a, b) => (rank(a) ?? Infinity) - (rank(b) ?? Infinity) || a.nickname.localeCompare(b.nickname, "en"));
  const size = rowsPerPage();
  pageCount = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pageCount - 1);
  const start = page * size;
  const fragment = document.createDocumentFragment();
  for (const row of rows.slice(start, start + size)) {
    const tr = document.createElement("tr");
    const position = rank(row);
    if (position === null) tr.classList.add("unranked");
    else if (position <= 3) tr.classList.add("leading");
    const values = [position ?? "-", row.nickname, format(row.scores[key])];
    values.forEach((value, index) => {
      const td = document.createElement("td");
      if (index === 0) {
        const badge = document.createElement("span"); badge.className = "rank"; badge.textContent = value; td.append(badge);
      } else { td.textContent = value; }
      if (index === 1) td.title = row.nickname;
      if (index === 2) td.classList.add("score-active");
      tr.append(td);
    });
    fragment.append(tr);
  }
  if (!rows.length) {
    const tr = document.createElement("tr"), td = document.createElement("td");
    td.colSpan = 3; td.className = "empty";
    td.textContent = "No participants registered yet. Standings will appear after registration.";
    tr.append(td); fragment.append(tr);
  }
  $("standings").replaceChildren(fragment);
  $("challenge-title").textContent = `Challenge ${key} · ${board.challenges[key].title}`;
  $("ranking-title").textContent = `Challenge ${key} standings`;
  $("ranking-description").textContent = `Best submission in this Challenge · out of ${board.rules.challenge_max}`;
  $("score-heading").textContent = `C${key} score`;
  $("stat-people").textContent = rows.length;
  for (const status of ["completed", "running", "queued"]) $("stat-" + status).textContent = board.queue[status] || 0;
  $("page-label").textContent = `Page ${page + 1} / ${pageCount}`;
  $("range-label").textContent = rows.length ? `Participants ${start + 1}–${Math.min(start + size, rows.length)} of ${rows.length}` : "Waiting for registration";
  $("previous-page").disabled = page === 0;
  $("next-page").disabled = page === pageCount - 1;
  updateRotation();
}
function updateConnection() {
  const disconnected = stale();
  const changed = $("stale-warning").hidden === disconnected;
  document.body.dataset.connection = disconnected ? "stale" : lastSuccess ? "live" : "waiting";
  const message = disconnected ? (lastSuccess ? "Connection delayed · last received scores" : "Connection delayed · waiting for this Challenge") : lastSuccess ? "Connected · updates every 5 seconds" : "Connecting";
  // Do not repeatedly announce an unchanged live region to screen readers.
  if ($("connection").textContent !== message) $("connection").textContent = message;
  $("stale-warning").hidden = !disconnected;
  $("updated").textContent = lastSuccess ? `Last received ${new Date(lastSuccess).toLocaleTimeString("en-GB", {hour12:false})}` : "No results received yet.";
  if (changed && board) drawBoard();
  updateRotation();
}
function updateRotation() {
  $("rotate-pages").setAttribute("aria-pressed", String(autoRotate));
  $("rotate-pages").textContent = autoRotate ? "Auto paging on" : "Auto paging off";
  $("rotation-note").textContent = stale() ? "Auto paging paused until reconnection" : pageCount === 1 ? "All participants are on this page." : autoRotate ? "Next page every 15 seconds · returns to the first page" : "Use the arrows to change page.";
}
async function refreshBoard() {
  if (fetchingBoard) return;
  fetchingBoard = true;
  const key = $("ranking").value;
  try {
    const next = await api(`/api/board?challenge=${encodeURIComponent(key)}`);
    if (key !== $("ranking").value) return;
    if (next.challenge !== key || !Array.isArray(next.participants) || !next.challenges?.[key] || !next.rules || !next.queue) throw new Error("Invalid board response");
    board = next;
    if (stale()) nextRotation = Date.now() + rotationMs;
    lastSuccess = Date.now(); connectionFailed = false;
    drawBoard();
  } catch (_) { if (key === $("ranking").value) connectionFailed = true; }
  finally {
    fetchingBoard = false;
    if (key !== $("ranking").value) refreshBoard();
    else updateConnection();
  }
}
function changePage(delta) {
  page = Math.max(0, Math.min(pageCount - 1, page + delta));
  nextRotation = Date.now() + rotationMs;
  drawBoard();
}
$("ranking").addEventListener("change", () => { nextRotation = Date.now() + rotationMs; rememberView(); showLoading(); refreshBoard(); });
$("previous-page").addEventListener("click", () => changePage(-1));
$("next-page").addEventListener("click", () => changePage(1));
$("rotate-pages").addEventListener("click", () => { autoRotate = !autoRotate; nextRotation = Date.now() + rotationMs; rememberView(); updateRotation(); });
window.addEventListener("resize", drawBoard);
document.addEventListener("visibilitychange", () => { if (!document.hidden) { nextRotation = Date.now() + rotationMs; refreshBoard(); } });

if (isDisplay) {
  // This page requests public board data only. No credentials or personal results.
  $("fullscreen").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else throw new Error("unsupported");
      $("display-notice").textContent = "";
    } catch (_) { $("display-notice").textContent = "Use your browser menu to enter full screen."; }
  });
  document.addEventListener("fullscreenchange", () => { $("fullscreen").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen"; drawBoard(); });
  document.addEventListener("keydown", event => {
    if (event.target.closest("button, select, input, a") || event.altKey || event.ctrlKey || event.metaKey) return;
    if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); changePage(event.key === "ArrowRight" ? 1 : -1); }
  });
}
showLoading();
refreshBoard();
setInterval(() => {
  updateConnection();
  if (!autoRotate || !board || stale() || document.hidden || $("board-panel").hidden || pageCount < 2) return;
  if (Date.now() >= nextRotation) { page = (page + 1) % pageCount; nextRotation = Date.now() + rotationMs; drawBoard(); }
}, 1000);
setInterval(async () => {
  if (document.hidden) return;
  await refreshBoard();
}, 5000);
