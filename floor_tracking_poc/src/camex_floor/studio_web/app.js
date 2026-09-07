const $ = (id) => document.getElementById(id);
let refreshGeneration = 0,
  selectionGeneration = 0,
  overlayGeneration = 0,
  projectGeneration = 0,
  displayedProject = null,
  displayedRevision = null,
  removalContext = null,
  sourceVideoKey = null,
  sourceVideoGeneration = 0,
  displayedReferenceKey = null,
  captureRequest = null;
let pid = new URLSearchParams(location.search).get("project"),
  site,
  cid,
  anchors = [],
  role = "fit",
  pending = null,
  dirty = false,
  floorImage,
  referenceImage,
  overlays = new Map(),
  measuring = false,
  measurePoints = [],
  moving = false,
  dragStart = null,
  regionMode = false,
  region = [],
  polling = false,
  activeTab = "calibration",
  shownOutput = null;
const colors = ["#168678", "#447dc0", "#d6764c", "#8b66b3", "#9c862d"];
const status = (text, error = false) => {
  $("status").textContent = text;
  $("status").className = error ? "error" : "";
};
const asset = (kind, camera = cid, job, project = pid) =>
  `/asset?project=${encodeURIComponent(project)}&kind=${kind}${camera ? "&camera=" + encodeURIComponent(camera) : ""}${job ? "&job=" + encodeURIComponent(job) : ""}`;

function clearMedia(id) {
  const media = $(id);
  if (id === "sourceVideo") {
    sourceVideoKey = null;
    ++sourceVideoGeneration;
    media.onloadedmetadata = null;
  }
  media.pause();
  media.removeAttribute("src");
  media.load();
}
function positionSourceVideo(seconds) {
  const video = $("sourceVideo"),
    generation = sourceVideoGeneration;
  const project = pid,
    camera = cid;
  const position = () => {
    if (
      generation !== sourceVideoGeneration ||
      project !== pid ||
      camera !== cid
    )
      return;
    video.pause();
    const maximum = Number.isFinite(video.duration)
      ? Math.max(0, video.duration - 0.001)
      : seconds;
    video.currentTime = Math.max(0, Math.min(seconds, maximum));
  };
  if (video.readyState >= 1) position();
  else video.onloadedmetadata = position;
}
function loadSourceVideo(camera, project, id) {
  const key = JSON.stringify([project, id, camera.video_sha256, camera.video]);
  if (key === sourceVideoKey) return;
  clearMedia("sourceVideo");
  sourceVideoKey = key;
  $("sourceVideo").src = asset("video", id, null, project);
  positionSourceVideo(
    camera.reference_actual_time_s ?? camera.reference_time_s ?? 0,
  );
}
function updateReferenceDetails() {
  const camera = site?.cameras?.[cid];
  if (!camera) {
    $("referenceDetails").textContent = "";
    return;
  }
  const actual = camera.reference_actual_time_s;
  const time = Number(actual ?? camera.reference_time_s ?? 0).toFixed(3);
  const kind =
    camera.calibration_state === "calibrated"
      ? "Undistorted reference"
      : "Original reference";
  $("referenceDetails").textContent =
    `${actual == null ? "Selected" : "Captured"} ${time} s | ${kind}`;
  $("referenceDetails").title =
    `Requested time: ${Number(camera.reference_time_s || 0).toFixed(3)} seconds`;
}
function showCameraView(view) {
  document
    .querySelectorAll("[data-view]")
    .forEach(
      (button) =>
        (button.className = button.dataset.view === view ? "active" : ""),
    );
  for (const name of ["reference", "video", "review"])
    $(name + "View").hidden = view !== name;
  if (view === "reference") {
    updateReferenceDetails();
    draw();
  }
  if (view === "review" && site?.cameras?.[cid]?.calibration_job)
    $("reviewImage").src = asset("calibration_review");
}
function clearOutput() {
  shownOutput = null;
  $("output").hidden = true;
  clearMedia("outputVideo");
  for (const id of ["downloadVideo", "downloadEvents", "downloadMetrics"])
    $(id).removeAttribute("href");
}
function cameraControls() {
  const camera = site?.cameras?.[cid];
  for (const id of [
    "saveScale",
    "measure",
    "layer",
    "opacity",
    "angle",
    "scale",
    "transform",
    "drag",
    "lock",
    "visibility",
    "floorRegion",
    "saveRegion",
    "fitRole",
    "checkRole",
    "undo",
    "fitHomography",
    "save",
    "approve",
    "snapshot",
    "useVideoTime",
  ]) {
    $(id).disabled = !camera;
  }
  $("fitLayer").disabled = !camera?.layer_to_image_h;
  $("snapshot").disabled =
    !camera || camera.locked || busy() || captureRequest?.project === pid;
  $("calibrate").disabled = !camera || busy();
  $("removeCamera").disabled = !camera || camera.locked || busy();
  $("removeCamera").title = camera?.locked
    ? "Unlock this camera before removing it"
    : "Remove this camera from the project; keep its files";
  const hasCameras = !!Object.keys(site?.cameras || {}).length;
  $("finish").disabled = !hasCameras || busy();
  $("finishPreview").disabled = !hasCameras || busy();
  $("run").disabled = !hasCameras || busy();
}
function clearCameraSelection() {
  ++selectionGeneration;
  ++overlayGeneration;
  cid = null;
  anchors = [];
  dirty = false;
  pending = null;
  referenceImage = null;
  displayedReferenceKey = null;
  overlays.clear();
  measuring = moving = regionMode = false;
  measurePoints = [];
  region = [];
  dragStart = null;
  role = "fit";
  $("fitRole").className = "active";
  $("checkRole").className = "";
  $("saveRegion").hidden = true;
  $("cameraList").replaceChildren();
  $("anchors").replaceChildren();
  for (const id of ["verification", "parameters", "cameraState", "mapMode"])
    $(id).textContent = "";
  $("cameraTitle").textContent = "Add the first camera";
  $("referenceCanvas")
    .getContext("2d")
    .clearRect(0, 0, $("referenceCanvas").width, $("referenceCanvas").height);
  clearMedia("sourceVideo");
  $("reviewImage").removeAttribute("src");
  $("referenceDetails").textContent = "";
  showCameraView("reference");
  $("referenceTime").value = 0;
  document.querySelector(".inspector").hidden = true;
  $("calibration").classList.add("no-cameras");
  $("lock").textContent = "Lock camera";
  $("visibility").textContent = "Hide camera";
  cameraControls();
}
function resetProject(next) {
  ++projectGeneration;
  ++refreshGeneration;
  pid = next || null;
  captureRequest = null;
  site = null;
  displayedProject = displayedRevision = null;
  floorImage = null;
  clearCameraSelection();
  clearOutput();
  $("syncVideos").replaceChildren();
  $("offsets").replaceChildren();
  for (const id of ["jobStatus", "jobLog", "runState", "runLog", "revision"])
    $(id).textContent = "";
  $("jobProgress").value = $("runProgress").value = 0;
  $("ppm").value = "";
  $("floorCanvas")
    .getContext("2d")
    .clearRect(0, 0, $("floorCanvas").width, $("floorCanvas").height);
  $("runStart").value = 0;
  $("runDuration").value = 120;
  $("runFps").value = 10;
  $("runWidth").value = 1280;
  $("detectorSize").value = "416";
  $("angle").value = 0;
  $("scale").value = 1;
  activeTab = "calibration";
  $("calibrationTab").className = "active";
  $("trackingTab").className = "";
  $("calibration").hidden = !pid;
  $("tracking").hidden = true;
  $("empty").hidden = !!pid;
  $("siteState").textContent = pid ? "Loading project" : "";
  removalContext = null;
  $("removeCameraDialog").close();
  history.replaceState(
    null,
    "",
    pid ? "?project=" + encodeURIComponent(pid) : "/",
  );
}
async function api(path, body) {
  const r = await fetch(
    path,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body, project: pid }),
        }
      : {},
  );
  const d = await r.json();
  if (!r.ok) throw Error(d.error || "Request failed");
  return d;
}
async function upload(file, where) {
  where.textContent = `Uploading ${file.name}...`;
  const r = await fetch("/api/upload?name=" + encodeURIComponent(file.name), {
    method: "POST",
    body: file,
  });
  const d = await r.json();
  if (!r.ok) throw Error(d.error);
  where.textContent = "Upload verified";
  return d.upload_id;
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(Error("Image unavailable"));
    im.src = url;
  });
}
async function projects() {
  const project = projectGeneration;
  const d = await api("/api/projects");
  if (project !== projectGeneration) return;
  $("projectSelect").replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Select project";
  $("projectSelect").append(blank);
  for (const p of d.projects) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    $("projectSelect").append(o);
  }
  $("projectSelect").value = pid || "";
  $("engine").textContent = d.engine.message;
  if (pid) await refresh(true);
  else status("Create a project or select an existing site");
}
async function refresh(force = false) {
  if (!pid) return;
  const requestedProject = pid,
    project = projectGeneration;
  const generation = ++refreshGeneration;
  const d = await api(
    "/api/site?project=" + encodeURIComponent(requestedProject),
  );
  if (
    project !== projectGeneration ||
    requestedProject !== pid ||
    d.project_id !== requestedProject ||
    generation !== refreshGeneration ||
    (site?.project_id === requestedProject && d.revision < site.revision)
  )
    return;
  const changed =
    !site ||
    site.project_id !== d.project_id ||
    site.revision !== d.revision ||
    displayedProject !== requestedProject ||
    displayedRevision !== d.revision;
  site = d;
  $("engine").textContent = site.engine.message;
  $("empty").hidden = true;
  $("calibration").hidden = activeTab !== "calibration";
  $("tracking").hidden = activeTab !== "tracking";
  $("revision").textContent = `Revision ${site.revision}`;
  $("siteState").textContent =
    site.state === "approved"
      ? "Calibration approved"
      : site.state === "preview"
        ? "Preview saved; verification pending"
        : "Calibration in progress";
  renderJobs();
  if (changed || force) {
    const loadedFloor = await loadImage(
      asset("floor", null, null, requestedProject),
    );
    if (
      project !== projectGeneration ||
      requestedProject !== pid ||
      site.revision !== d.revision
    )
      return;
    floorImage = loadedFloor;
    $("ppm").value = site.floor.pixels_per_metre || "";
    const ids = Object.keys(site.cameras);
    cid = ids.includes(cid) ? cid : ids[0];
    if (cid) {
      if (!dirty || force) await selectCamera(cid);
      else renderList();
    } else {
      clearCameraSelection();
      canvas($("floorCanvas"), floorImage);
      status(
        "This project has no cameras. Click Add to register the first camera.",
      );
    }
    if (project !== projectGeneration || requestedProject !== pid) return;
    renderOffsets();
    if (site.revision === d.revision) {
      displayedProject = requestedProject;
      displayedRevision = d.revision;
    }
  }
}
async function selectCamera(id) {
  if (!site?.cameras?.[id]) return;
  const requestedProject = pid,
    project = projectGeneration;
  const generation = ++selectionGeneration;
  cid = id;
  const c = site.cameras[id];
  anchors = structuredClone(c.anchors || []);
  dirty = false;
  pending = null;
  region = (c.floor_polygon || []).map((p) => [
    p[0] * c.image_size_wh[0],
    p[1] * c.image_size_wh[1],
  ]);
  referenceImage = null;
  const loadedReference = await loadImage(
    asset("reference", id, null, requestedProject) +
      "&image=" +
      encodeURIComponent(c.reference_sha256 || c.reference),
  );
  if (
    generation !== selectionGeneration ||
    project !== projectGeneration ||
    requestedProject !== pid ||
    cid !== id
  )
    return;
  referenceImage = loadedReference;
  document.querySelector(".inspector").hidden = false;
  $("calibration").classList.remove("no-cameras");
  $("cameraTitle").textContent = c.label;
  $("cameraState").textContent = c.approved_fingerprint
    ? "Approved"
    : c.calibration_state === "calibrated"
      ? c.local_geometry_status === "provisional"
        ? "Provisional geometry"
        : "Ready to align"
      : "Not calibrated";
  const referenceKey = JSON.stringify([
    requestedProject,
    id,
    c.reference_sha256,
    c.reference_time_s,
  ]);
  if (displayedReferenceKey !== referenceKey) {
    $("referenceTime").value = c.reference_time_s || 0;
    displayedReferenceKey = referenceKey;
  }
  $("referenceTime").max = c.video_duration_s;
  loadSourceVideo(c, requestedProject, id);
  if (c.calibration_job) $("reviewImage").src = asset("calibration_review");
  else $("reviewImage").removeAttribute("src");
  updateReferenceDetails();
  $("calibrate").disabled = busy();
  $("fitLayer").disabled = !c.layer_to_image_h;
  $("lock").textContent = c.locked ? "Unlock camera" : "Lock camera";
  $("visibility").textContent =
    c.visible === false ? "Show camera" : "Hide camera";
  $("parameters").textContent = JSON.stringify(
    {
      distortion: c.distortion,
      local_geometry: c.local_geometry_status,
      local_rotation: c.camera_to_local_rotation,
      camera_to_floor_R: c.camera_to_floor_R,
      camera_to_floor_t: c.camera_to_floor_t,
      local_floor_offset: c.local_floor_offset,
      global_pose: c.global_pose,
    },
    null,
    2,
  );
  await loadOverlays();
  if (
    generation !== selectionGeneration ||
    project !== projectGeneration ||
    requestedProject !== pid
  )
    return;
  draw();
  renderList();
  renderAnchors();
  cameraControls();
}
async function loadOverlays() {
  const generation = ++overlayGeneration,
    project = projectGeneration,
    requestedProject = pid,
    revision = site?.revision,
    kind = $("layer").value;
  const loaded = new Map();
  if (kind === "none") {
    overlays = loaded;
    return;
  }
  for (const [id, c] of Object.entries(site?.cameras || {})) {
    if (!c.image_to_floor_h || c.visible === false) continue;
    try {
      const im = await loadImage(
        `/overlay?project=${requestedProject}&camera=${id}&kind=${kind}&revision=${revision}`,
      );
      loaded.set(id, im);
    } catch {}
  }
  if (
    generation === overlayGeneration &&
    project === projectGeneration &&
    requestedProject === pid &&
    revision === site?.revision
  )
    overlays = loaded;
}
function canvas(el, im) {
  const b = el.getBoundingClientRect(),
    ratio = devicePixelRatio || 1;
  el.width = Math.max(1, b.width * ratio);
  el.height = Math.max(1, b.height * ratio);
  const ctx = el.getContext("2d");
  ctx.scale(ratio, ratio);
  const s = Math.min(b.width / im.width, b.height / im.height),
    x = (b.width - im.width * s) / 2,
    y = (b.height - im.height * s) / 2;
  ctx.drawImage(im, x, y, im.width * s, im.height * s);
  return { ctx, s, x, y };
}
function pointOnCanvas(e, el, im) {
  const r = el.getBoundingClientRect(),
    s = Math.min(r.width / im.width, r.height / im.height),
    p = [
      (e.clientX - r.left - (r.width - im.width * s) / 2) / s,
      (e.clientY - r.top - (r.height - im.height * s) / 2) / s,
    ];
  return p[0] >= 0 && p[1] >= 0 && p[0] <= im.width && p[1] <= im.height
    ? p
    : null;
}
function marker(ctx, p, text, color) {
  ctx.fillStyle = color;
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.font = "bold 12px system-ui";
  ctx.fillText(text, p[0] + 8, p[1] - 6);
}
function hpoint(h, p) {
  const q = h.map((row) => row[0] * p[0] + row[1] * p[1] + row[2]);
  return [q[0] / q[2], q[1] / q[2]];
}
function draw() {
  if (!floorImage || activeTab !== "calibration") return;
  const f = canvas($("floorCanvas"), floorImage);
  for (const [id, im] of overlays) {
    f.ctx.globalAlpha = Number($("opacity").value) * (id === cid ? 1 : 0.18);
    f.ctx.drawImage(
      im,
      f.x,
      f.y,
      floorImage.width * f.s,
      floorImage.height * f.s,
    );
  }
  f.ctx.globalAlpha = 1;
  if (!referenceImage || !site?.cameras?.[cid]) return;
  const r = canvas($("referenceCanvas"), referenceImage);
  anchors.forEach((a, i) => {
    const color = a.role === "check" ? "#bb802c" : "#168678";
    marker(
      r.ctx,
      [r.x + a.image_px[0] * r.s, r.y + a.image_px[1] * r.s],
      i + 1,
      color,
    );
    marker(
      f.ctx,
      [f.x + a.map_px[0] * f.s, f.y + a.map_px[1] * f.s],
      i + 1,
      color,
    );
    const h = site.cameras[cid].image_to_floor_h;
    if (h) {
      const p = hpoint(h, a.image_px);
      f.ctx.strokeStyle = color;
      f.ctx.beginPath();
      f.ctx.moveTo(f.x + a.map_px[0] * f.s, f.y + a.map_px[1] * f.s);
      f.ctx.lineTo(f.x + p[0] * f.s, f.y + p[1] * f.s);
      f.ctx.stroke();
    }
  });
  if (pending)
    marker(
      r.ctx,
      [r.x + pending[0] * r.s, r.y + pending[1] * r.s],
      "+",
      "#d17722",
    );
  measurePoints.forEach((p, i) =>
    marker(f.ctx, [f.x + p[0] * f.s, f.y + p[1] * f.s], i + 1, "#8158a8"),
  );
  if (regionMode && region.length) {
    r.ctx.strokeStyle = "#168678";
    r.ctx.fillStyle = "#16867844";
    r.ctx.beginPath();
    region.forEach((p, i) =>
      i
        ? r.ctx.lineTo(r.x + p[0] * r.s, r.y + p[1] * r.s)
        : r.ctx.moveTo(r.x + p[0] * r.s, r.y + p[1] * r.s),
    );
    r.ctx.closePath();
    r.ctx.fill();
    r.ctx.stroke();
    region.forEach((p, i) =>
      marker(r.ctx, [r.x + p[0] * r.s, r.y + p[1] * r.s], i + 1, "#168678"),
    );
  }
  $("mapMode").textContent = measuring
    ? "Measure known dimension"
    : moving
      ? "Drag selected layer"
      : pending
        ? "Choose matching floor-plan point"
        : role === "check"
          ? "Independent holdout check"
          : "Camera alignment";
}
function renderList() {
  $("cameraList").replaceChildren();
  Object.entries(site.cameras).forEach(([id, c], i) => {
    const b = document.createElement("button");
    b.className = "camera" + (id === cid ? " active" : "");
    b.style.borderLeftColor = colors[i % colors.length];
    const title = document.createElement("span");
    title.textContent = c.label;
    const small = document.createElement("small");
    small.textContent =
      (c.approved_fingerprint
        ? "Approved"
        : c.calibration_state === "calibrated"
          ? "Aligned / draft"
          : "Needs calibration") + (c.locked ? " / locked" : "");
    b.append(title, small);
    b.onclick = async () => {
      if (dirty) {
        if (!(await save())) return;
      }
      await selectCamera(id);
    };
    $("cameraList").append(b);
  });
}
function renderAnchors() {
  if (!cid) return;
  const c = site.cameras[cid],
    v = site.verification[cid];
  $("verification").textContent =
    `${anchors.filter((a) => a.role !== "check").length} fit / ${anchors.filter((a) => a.role === "check").length} holdout. ${v.ready ? "Holdout checks passed" : v.reasons.join(". ")}${c.fit_audit ? " | Inliers " + c.fit_audit.inlier_count + "/" + c.fit_audit.fit_count : ""}`;
  $("anchors").replaceChildren();
  anchors.forEach((a, i) => {
    const row = document.createElement("div");
    row.className = "point-row" + (a.role === "check" ? " holdout" : "");
    const l = document.createElement("span");
    l.textContent = `${i + 1}. ${a.role === "check" ? "Holdout" : "Fit"} (${a.image_px.map(Math.round).join(", ")})`;
    const value = document.createElement("span");
    value.textContent = c.image_to_floor_h
      ? Math.hypot(
          ...hpoint(c.image_to_floor_h, a.image_px).map(
            (x, j) => x - a.map_px[j],
          ),
        ).toFixed(1) + " px"
      : "--";
    row.append(l, value);
    $("anchors").append(row);
  });
}
function busy() {
  return site?.jobs?.some((j) => ["running", "queued"].includes(j.state));
}
function renderJobs() {
  const jobs = Object.keys(site?.cameras || {}).length ? site.jobs || [] : [],
    cal = jobs.find((j) => j.kind === "calibrate"),
    run = jobs.find((j) => j.kind === "infer");
  if (cal) {
    $("jobStatus").textContent = `${cal.state}: ${cal.message}`;
    if (cal.state === "complete") $("jobProgress").value = 1;
    else if (cal.progress) $("jobProgress").value = cal.progress.fraction;
    else $("jobProgress").removeAttribute("value");
    $("jobLog").textContent = cal.log_tail || "";
  } else {
    $("jobStatus").textContent = $("jobLog").textContent = "";
    $("jobProgress").value = 0;
  }
  if (run) {
    $("runState").textContent =
      `${run.state}: ${run.progress?.stage || run.message}${run.progress ? " / " + run.progress.completed_s.toFixed(1) + " of " + run.progress.total_s + " s" : ""}`;
    $("runProgress").value =
      run.state === "complete" ? 1 : run.progress?.fraction || 0.02;
    $("runLog").textContent = run.log_tail || "";
    if (run.state === "complete" && shownOutput !== run.id) {
      shownOutput = run.id;
      $("output").hidden = false;
      $("outputVideo").src = asset("demo", null, run.id);
      $("downloadVideo").href = asset("demo", null, run.id) + "&download=1";
      $("downloadEvents").href = asset("events", null, run.id) + "&download=1";
      $("downloadMetrics").href =
        asset("telemetry", null, run.id) + "&download=1";
    }
  } else {
    $("runState").textContent = $("runLog").textContent = "";
    $("runProgress").value = 0;
    if (shownOutput || $("outputVideo").hasAttribute("src")) clearOutput();
  }
  cameraControls();
}
function renderOffsets() {
  $("offsets").replaceChildren();
  for (const [id, c] of Object.entries(site.cameras)) {
    const l = document.createElement("label");
    l.textContent = c.label + " offset (s)";
    const input = document.createElement("input");
    input.type = "number";
    input.step = ".1";
    input.min = "0";
    input.value = c.video_start_seconds || 0;
    input.dataset.camera = id;
    l.append(input);
    $("offsets").append(l);
  }
}
async function save(action = "save", extra = {}) {
  if (!cid) return false;
  const requestedProject = pid,
    project = projectGeneration;
  try {
    const updated = await api("/api/edit", {
      camera_id: cid,
      anchors,
      revision: site.revision,
      action,
      ...extra,
    });
    if (requestedProject !== pid || project !== projectGeneration) return false;
    site = updated;
    dirty = false;
    await refresh(true);
    status(action === "approve" ? "Camera approved" : "Project saved");
    return true;
  } catch (e) {
    status(e.message, true);
    return false;
  }
}
async function tab(name) {
  if (dirty && !(await save())) return;
  activeTab = name;
  $("calibration").hidden = name !== "calibration" || !site;
  $("tracking").hidden = name !== "tracking" || !site;
  $("calibrationTab").className = name === "calibration" ? "active" : "";
  $("trackingTab").className = name === "tracking" ? "active" : "";
  if (name === "calibration") requestAnimationFrame(draw);
  else renderOffsets();
}
$("newProject").onclick = async () => {
  if (dirty && !(await save())) return;
  $("projectDialog").showModal();
};
$("emptyCreate").onclick = () => $("projectDialog").showModal();
document
  .querySelectorAll("[data-close]")
  .forEach((b) => (b.onclick = () => $(b.dataset.close).close()));
$("projectSelect").onchange = async () => {
  const next = $("projectSelect").value;
  if (dirty && !(await save())) {
    $("projectSelect").value = pid || "";
    return;
  }
  resetProject(next);
  if (pid) await refresh(true);
  else status("Create a project or select an existing site");
};
$("projectForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = e.target,
    fields = f.elements,
    msg = f.querySelector(".formStatus");
  try {
    const uid = await upload(fields.floor.files[0], msg);
    const created = await api("/api/create", {
      name: fields.name.value,
      upload_id: uid,
      page: Number(fields.page.value) - 1,
    });
    resetProject(created.project_id);
    $("projectDialog").close();
    await projects();
  } catch (error) {
    msg.textContent = error.message;
    msg.className = "formStatus error";
  }
};
$("addCamera").onclick = () => {
  if (site) $("cameraDialog").showModal();
};
$("cameraForm").onsubmit = async (e) => {
  e.preventDefault();
  const f = e.target,
    fields = f.elements,
    msg = f.querySelector(".formStatus");
  try {
    const uid = await upload(fields.video.files[0], msg);
    cid = fields.namedItem("id").value;
    site = await api("/api/camera", {
      camera_id: cid,
      label: fields.label.value,
      upload_id: uid,
    });
    $("cameraDialog").close();
    dirty = false;
    await refresh(true);
    f.reset();
    status("Camera added. Select a reference frame and calibrate.");
  } catch (error) {
    msg.textContent = error.message;
    msg.className = "formStatus error";
  }
};
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => showCameraView(b.dataset.view)));
$("useVideoTime").onclick = () =>
  ($("referenceTime").value = $("sourceVideo").currentTime.toFixed(2));
$("snapshot").onclick = async () => {
  const camera = site?.cameras?.[cid];
  if (!camera || camera.locked || busy() || captureRequest) return;
  if (
    (camera.calibration_state === "calibrated" || anchors.length || dirty) &&
    !window.confirm(
      "Replace this camera's reference? Its calibration and point alignment will need to be redone.",
    )
  )
    return;
  const context = {
    project: pid,
    camera: cid,
    generation: projectGeneration,
    seconds: Number($("referenceTime").value),
  };
  captureRequest = context;
  cameraControls();
  try {
    const updated = await api("/api/snapshot", {
      camera_id: context.camera,
      seconds: context.seconds,
    });
    if (
      pid !== context.project ||
      cid !== context.camera ||
      projectGeneration !== context.generation
    )
      return;
    site = updated;
    anchors = [];
    pending = null;
    region = [];
    measurePoints = [];
    regionMode = measuring = moving = false;
    $("saveRegion").hidden = true;
    dirty = false;
    await refresh(true);
    if (
      pid !== context.project ||
      cid !== context.camera ||
      projectGeneration !== context.generation
    )
      return;
    const captured =
      site.cameras[cid].reference_actual_time_s ?? context.seconds;
    positionSourceVideo(captured);
    showCameraView("reference");
    status(
      `Saved reference at ${Number(captured).toFixed(3)} s (requested ${context.seconds.toFixed(3)} s). Recalibration required.`,
    );
  } catch (e) {
    if (pid === context.project && projectGeneration === context.generation)
      status(e.message, true);
  } finally {
    if (captureRequest === context) captureRequest = null;
    cameraControls();
  }
};
$("calibrate").onclick = async () => {
  try {
    if (dirty && !(await save())) return;
    const j = await api("/api/calibrate", {
      camera_id: cid,
      level_method: $("level").value,
    });
    status(
      j.reused ? "Unchanged calibration reused" : "Camera calibration queued",
    );
    await refresh();
  } catch (e) {
    status(e.message, true);
  }
};
$("floorRegion").onclick = () => {
  if (site.cameras[cid].calibration_state !== "calibrated") {
    status(
      "Run the initial calibration to create the rectified reference first",
      true,
    );
    return;
  }
  regionMode = !regionMode;
  region = [];
  pending = null;
  $("saveRegion").hidden = !regionMode;
  status(
    regionMode
      ? "Select visible floor vertices in the rectified image"
      : "Floor selection closed",
  );
  draw();
  cameraControls();
};
$("saveRegion").onclick = async () => {
  const polygon = region.map((p) => [
    p[0] / referenceImage.width,
    p[1] / referenceImage.height,
  ]);
  if (await save("floor_polygon", { polygon })) {
    regionMode = false;
    $("saveRegion").hidden = true;
    status("Floor region saved. Click Calibrate camera to refine it.");
    draw();
    cameraControls();
  }
};
$("referenceCanvas").onclick = (e) => {
  if (!referenceImage || site.cameras[cid].locked) return;
  const p = pointOnCanvas(e, $("referenceCanvas"), referenceImage);
  if (!p) return;
  if (regionMode) {
    region.push(p);
    draw();
    return;
  }
  pending = p;
  status("Click the same floor landmark on the plan");
  draw();
};
$("floorCanvas").onclick = (e) => {
  if (!floorImage || moving) return;
  const p = pointOnCanvas(e, $("floorCanvas"), floorImage);
  if (!p) return;
  if (measuring) {
    measurePoints.push(p);
    if (measurePoints.length === 2) {
      const ppm =
        Math.hypot(...measurePoints[0].map((x, i) => x - measurePoints[1][i])) /
        Number($("knownDistance").value);
      measuring = false;
      save("save", { pixels_per_metre: ppm });
    }
    draw();
    return;
  }
  if (pending) {
    anchors.push({ image_px: pending, map_px: p, role });
    dirty = true;
    pending = null;
    renderAnchors();
    draw();
    status("Correspondence added; save or fit to commit it");
  }
};
$("floorCanvas").onpointerdown = (e) => {
  if (moving) {
    dragStart = pointOnCanvas(e, $("floorCanvas"), floorImage);
    $("floorCanvas").setPointerCapture(e.pointerId);
  }
};
$("floorCanvas").onpointerup = (e) => {
  if (moving && dragStart) {
    const p = pointOnCanvas(e, $("floorCanvas"), floorImage);
    if (p)
      save("transform", { dx: p[0] - dragStart[0], dy: p[1] - dragStart[1] });
    dragStart = null;
  }
};
$("fitRole").onclick = () => {
  role = "fit";
  $("fitRole").className = "active";
  $("checkRole").className = "";
  draw();
};
$("checkRole").onclick = () => {
  role = "check";
  $("checkRole").className = "active";
  $("fitRole").className = "";
  draw();
};
$("undo").onclick = () => {
  if (regionMode) region.pop();
  else {
    anchors.pop();
    dirty = true;
    renderAnchors();
  }
  draw();
};
$("fitHomography").onclick = () => save("fit");
$("fitLayer").onclick = () => save("fit_layer");
$("save").onclick = () => save();
$("approve").onclick = () => save("approve");
$("measure").onclick = () => {
  measuring = true;
  moving = false;
  measurePoints = [];
  draw();
};
$("saveScale").onclick = () =>
  save("save", { pixels_per_metre: Number($("ppm").value) });
$("transform").onclick = () =>
  save("transform", {
    rotation: Number($("angle").value),
    scale: Number($("scale").value),
  });
$("drag").onclick = () => {
  moving = !moving;
  measuring = false;
  draw();
};
$("lock").onclick = () => save(site.cameras[cid].locked ? "unlock" : "lock");
$("visibility").onclick = () => save("visibility");
$("removeCamera").onclick = () => {
  const camera = site?.cameras?.[cid];
  if (!camera || camera.locked || busy()) return;
  removalContext = {
    project: pid,
    camera_id: cid,
    revision: site.revision,
    generation: projectGeneration,
  };
  $("removeCameraMessage").textContent =
    `Remove "${camera.label}" (${cid}) from "${site.name}"?`;
  $("removeCameraForm").querySelector(".formStatus").textContent = "";
  $("removeCameraDialog").showModal();
};
$("removeCameraForm").onsubmit = async (e) => {
  e.preventDefault();
  const context = removalContext;
  const message = e.target.querySelector(".formStatus");
  if (
    !context ||
    context.project !== pid ||
    context.generation !== projectGeneration
  )
    return;
  $("confirmRemoveCamera").disabled = true;
  try {
    const updated = await api("/api/remove-camera", context);
    if (context.project !== pid || context.generation !== projectGeneration)
      return;
    site = updated;
    clearCameraSelection();
    clearOutput();
    $("syncVideos").replaceChildren();
    $("removeCameraDialog").close();
    removalContext = null;
    await refresh(true);
    status(
      "Camera removed from this project. Files retained; finish calibration or save preview before a new run.",
    );
  } catch (error) {
    message.textContent = error.message;
    message.className = "formStatus error";
  } finally {
    $("confirmRemoveCamera").disabled = false;
  }
};
$("layer").onchange = async () => {
  await loadOverlays();
  draw();
};
$("opacity").oninput = draw;
async function finish(preview) {
  try {
    if (dirty && !(await save())) return;
    site = await api("/api/finish", { preview });
    await refresh(true);
    status(
      preview
        ? "Preview saved; metric acceptance remains pending"
        : "All camera calibrations approved",
    );
    await tab("tracking");
  } catch (e) {
    status(e.message, true);
  }
}
$("finish").onclick = () => finish(false);
$("finishPreview").onclick = () => finish(true);
$("calibrationTab").onclick = () => tab("calibration");
$("trackingTab").onclick = () => tab("tracking");
$("showUndistortion").onclick = () =>
  ($("reviewImage").src = asset("undistortion_review"));
$("showGeometry").onclick = () =>
  ($("reviewImage").src = asset("calibration_review"));
$("syncPreview").onclick = () => {
  $("syncVideos").replaceChildren();
  const start = Number($("runStart").value);
  for (const [id, c] of Object.entries(site.cameras)) {
    const holder = document.createElement("div"),
      title = document.createElement("strong"),
      v = document.createElement("video");
    title.textContent = c.label;
    v.controls = true;
    v.preload = "metadata";
    v.src = asset("video", id);
    const offset = Number(
      document.querySelector(`[data-camera="${id}"]`).value,
    );
    v.onloadedmetadata = () => (v.currentTime = start + offset);
    holder.append(title, v);
    $("syncVideos").append(holder);
  }
};
$("run").onclick = async () => {
  try {
    const offsets = Object.fromEntries(
      [...document.querySelectorAll("[data-camera]")].map((e) => [
        e.dataset.camera,
        Number(e.value),
      ]),
    );
    const j = await api("/api/run", {
      start_s: Number($("runStart").value),
      duration_s: Number($("runDuration").value),
      fps: Number($("runFps").value),
      inference_width: Number($("runWidth").value),
      detector_size: Number($("detectorSize").value),
      offsets,
    });
    status(j.reused ? "Completed run reused" : "Fresh MCT run queued");
    await refresh();
  } catch (e) {
    status(e.message, true);
  }
};
window.addEventListener("beforeunload", (e) => {
  if (dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
new ResizeObserver(draw).observe($("floorCanvas"));
new ResizeObserver(draw).observe($("referenceCanvas"));
setInterval(async () => {
  if (pid && !polling) {
    polling = true;
    try {
      await refresh();
    } catch (e) {
      status(e.message, true);
    } finally {
      polling = false;
    }
  }
}, 2000);
projects().catch((e) => status(e.message, true));
