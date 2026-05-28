const templates = {
  C: {
    training_backend: "sa3_lora",
    run_id: "layer-c-sa3-worker-smoke",
    owner: "burger",
    base_model: "stable-audio-3 small-sfx-base",
    species: "horsfields_bronze_cuckoo",
    steps: 10,
    checkpoint_every: 10,
    demo_every: 999999,
    num_workers: 0,
    seed: 42,
    notes: "",
  },
  A: {
    training_backend: "adapter_pending",
    run_id: "layer-a-training-run",
    owner: "",
    base_model: "cvssp/audioldm2",
    species: "site_257_ambient",
    steps: 10,
    checkpoint_every: 10,
    demo_every: 999999,
    num_workers: 0,
    seed: 42,
    notes: "Layer A worker adapter is not connected yet.",
  },
  B: {
    training_backend: "asset_workflow_pending",
    run_id: "layer-b-weather-assets-run",
    owner: "",
    base_model: "none",
    species: "weather_assets",
    steps: 1,
    checkpoint_every: 1,
    demo_every: 999999,
    num_workers: 0,
    seed: 42,
    notes: "Layer B may not require GPU training.",
  },
};

const statusLabels = {
  queued: "Queued",
  claimed: "Claimed",
  running: "Running",
  uploading: "Uploading",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  cancel_requested: "Cancel requested",
};

const cancellableStatuses = new Set(["queued", "claimed", "running", "uploading"]);

const state = {
  jobs: [],
  selectedJob: null,
};

const els = {
  authStatus: document.querySelector("#auth-status"),
  loginForm: document.querySelector("#login-form"),
  jobForm: document.querySelector("#job-form"),
  layer: document.querySelector("#layer"),
  preview: document.querySelector("#payload-preview"),
  jobsList: document.querySelector("#jobs-list"),
  jobsCount: document.querySelector("#jobs-count"),
  refreshBtn: document.querySelector("#refresh-btn"),
  detailTitle: document.querySelector("#detail-title"),
  jobDetail: document.querySelector("#job-detail"),
  cancelBtn: document.querySelector("#cancel-btn"),
  toast: document.querySelector("#toast"),
};

function showToast(message, variant = "info") {
  els.toast.textContent = message;
  els.toast.className = `toast ${variant}`;
  els.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    els.toast.hidden = true;
  }, 4500);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || `Request failed (${response.status})`);
  }
  return data;
}

function getPayload() {
  const formData = new FormData(els.jobForm);
  const payload = {
    layer: formData.get("layer"),
    training_backend: formData.get("training_backend"),
    run_id: formData.get("run_id"),
    owner: formData.get("owner"),
    base_model: formData.get("base_model"),
    species: formData.get("species"),
    steps: Number(formData.get("steps") || 10),
    checkpoint_every: Number(formData.get("checkpoint_every") || 10),
    demo_every: Number(formData.get("demo_every") || 999999),
    num_workers: 0,
    seed: Number(formData.get("seed") || 42),
  };
  const notes = String(formData.get("notes") || "").trim();
  if (notes) payload.notes = notes;
  return payload;
}

function updatePreview() {
  els.preview.textContent = JSON.stringify({ type: "training", payload: getPayload() }, null, 2);
}

function fillTemplate(layer) {
  const template = templates[layer] || templates.C;
  for (const [key, value] of Object.entries(template)) {
    const input = els.jobForm.elements.namedItem(key);
    if (input) input.value = value;
  }
  updatePreview();
}

function formatDate(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function statusClass(status) {
  return `status status-${String(status || "unknown").replaceAll("_", "-")}`;
}

function renderJobs() {
  els.jobsCount.textContent = `${state.jobs.length} visible`;
  els.jobsList.innerHTML = "";

  if (!state.jobs.length) {
    els.jobsList.innerHTML = `<p class="empty">No jobs for this account yet.</p>`;
    return;
  }

  for (const job of state.jobs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-row${state.selectedJob?.id === job.id ? " active" : ""}`;
    button.innerHTML = `
      <span>#${job.id}</span>
      <strong>${job.payload?.run_id || job.type}</strong>
      <i class="${statusClass(job.status)}">${statusLabels[job.status] || job.status}</i>
      <small>${formatDate(job.updated_at)}</small>
    `;
    button.addEventListener("click", () => selectJob(job.id));
    els.jobsList.appendChild(button);
  }
}

function renderDetail() {
  const job = state.selectedJob;
  if (!job) {
    els.detailTitle.textContent = "Select a job";
    els.cancelBtn.hidden = true;
    els.jobDetail.className = "empty";
    els.jobDetail.textContent = "Select a job to inspect payload, result, artifact paths, and errors.";
    return;
  }

  els.detailTitle.textContent = `#${job.id}`;
  els.cancelBtn.hidden = !cancellableStatuses.has(job.status);
  els.cancelBtn.dataset.jobId = job.id;
  els.jobDetail.className = "detail";
  els.jobDetail.innerHTML = `
    <div class="stats">
      <div><span>Status</span><strong class="${statusClass(job.status)}">${statusLabels[job.status] || job.status}</strong></div>
      <div><span>Worker</span><strong>${job.claimed_by || "-"}</strong></div>
      <div><span>Updated</span><strong>${formatDate(job.updated_at)}</strong></div>
    </div>
    <div class="artifact"><span>Artifact</span><code>${job.artifact_uri || job.result?.checkpoint_dvc_path || "-"}</code></div>
    ${job.error_message ? `<p class="error-box">${job.error_message}</p>` : ""}
    <div class="json-grid">
      <div class="preview"><span>Payload</span><pre>${JSON.stringify(job.payload || {}, null, 2)}</pre></div>
      <div class="preview"><span>Result</span><pre>${JSON.stringify(job.result || {}, null, 2)}</pre></div>
    </div>
  `;
}

async function refreshJobs(selectedId = state.selectedJob?.id) {
  const data = await request("/api/jobs?limit=50");
  state.jobs = data.jobs || [];
  state.selectedJob = state.jobs.find((job) => String(job.id) === String(selectedId)) || state.jobs[0] || null;
  renderJobs();
  renderDetail();
}

async function selectJob(jobId) {
  try {
    const data = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
    state.selectedJob = data.job;
    renderJobs();
    renderDetail();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function checkSession() {
  try {
    const data = await request("/api/me");
    els.authStatus.textContent = `Logged in as ${data.user.username}`;
    await refreshJobs();
  } catch {
    els.authStatus.textContent = "Login required";
  }
}

els.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const account = document.querySelector("#account").value.trim();
  const password = document.querySelector("#password").value;
  try {
    const data = await request("/api/login", {
      method: "POST",
      body: JSON.stringify({ account, password }),
    });
    els.authStatus.textContent = `Logged in as ${data.user.username}`;
    showToast("Logged in", "success");
    await refreshJobs();
  } catch (error) {
    showToast(error.message, "error");
  }
});

els.jobForm.addEventListener("input", updatePreview);
els.layer.addEventListener("change", () => fillTemplate(els.layer.value));

els.jobForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const data = await request("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ type: "training", payload: getPayload() }),
    });
    showToast(`Job ${data.job.id} queued`, "success");
    await refreshJobs(data.job.id);
  } catch (error) {
    showToast(error.message, "error");
  }
});

els.refreshBtn.addEventListener("click", async () => {
  try {
    await refreshJobs();
    showToast("Jobs refreshed", "success");
  } catch (error) {
    showToast(error.message, "error");
  }
});

els.cancelBtn.addEventListener("click", async () => {
  const jobId = els.cancelBtn.dataset.jobId;
  if (!jobId) return;
  try {
    const data = await request(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    showToast(`Job ${data.job.id} ${data.job.status}`, "success");
    await refreshJobs(data.job.id);
  } catch (error) {
    showToast(error.message, "error");
  }
});

fillTemplate("C");
checkSession();
window.setInterval(() => refreshJobs().catch(() => {}), 10000);
