(function () {
  const DASHBOARD_STATE_KEY = "unagiConsole.dashboardState";
  const TAB_GROUP_STATE_KEY = "unagiConsole.tabGroups";
  const COLLAPSIBLE_GROUP_STATE_KEY = "unagiConsole.collapsibleGroups";
  const UNAGI_CATEGORICAL_COLORS = [
    "#4DBBD5",
    "#F39B7F",
    "#00A087",
    "#E64B35",
    "#3C5488",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
  ];
  let plotlyPromise = null;

  function ensureResult(resultId) {
    const element = document.getElementById(resultId);
    if (!element) {
      return null;
    }
    element.classList.remove("hidden-result");
    return element;
  }

  async function parseResponse(response) {
    const text = await response.text();
    try {
      return JSON.parse(text);
    } catch (_error) {
      return { ok: response.ok, raw: text };
    }
  }

  async function submitForm(formId, endpoint, resultId, options) {
    const form = document.getElementById(formId);
    const result = ensureResult(resultId);
    if (!form || !result) {
      return;
    }
    let pendingWindow = null;
    if (options && options.openNewTabOnSuccess) {
      pendingWindow = window.open("about:blank", "_blank");
    }
    result.textContent = "Working...";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
      body: new URLSearchParams(new FormData(form)),
    });
    const payload = await parseResponse(response);
    if (response.ok && options && options.successMessage) {
      const warnings = Array.isArray(payload && payload.warnings) ? payload.warnings.filter(Boolean) : [];
      result.textContent = warnings.length
        ? `${options.successMessage}\nWarnings: ${warnings.join("; ")}`
        : String(options.successMessage);
    } else {
      result.textContent = JSON.stringify(payload, null, 2);
    }
    if (pendingWindow && (!response.ok || !payload.detail_url)) {
      pendingWindow.close();
      pendingWindow = null;
    }
    if (response.ok && options && options.redirectOnSuccess && payload.detail_url) {
      window.location.href = payload.detail_url;
    }
    if (response.ok && options && options.refreshExplorerOnSuccess) {
      const refreshOptions = options.refreshExplorerOnSuccess || {};
      const selectedCustomJobId = refreshOptions.selectPayloadJobId ? payload.job_id : refreshOptions.selectedCustomJobId;
      await reloadAnalysisExplorer(refreshOptions.jobId, {
        mode: refreshOptions.mode || "perturbation",
        pert_panel_mode: refreshOptions.pertPanelMode || "custom",
        selected_custom_job_id: selectedCustomJobId || "",
      });
      return;
    }
    if (response.ok && options && options.openNewTabOnSuccess && payload.detail_url) {
      if (pendingWindow) {
        pendingWindow.location.href = payload.detail_url;
      } else {
        const opened = window.open(payload.detail_url, "_blank", "noopener");
        if (!opened) {
          window.location.href = payload.detail_url;
        }
      }
    }
  }

  function initializeDynamicScope(scope) {
    initTabGroups(scope);
    initCollapsibleGroups(scope);
    initExplorerForms(scope);
    initCustomPerturbationForms(scope);
    initAnalysisUmapWidgets(scope);
    initPlotlySpecCards(scope);
    initHierarchicalExplorerWidgets(scope);
    initTemporalGraphWidgets(scope);
    initStageCompositionWidgets(scope);
    initIdremTrackPanels(scope);
  }

  async function reloadAnalysisExplorer(jobId, params) {
    if (!jobId) {
      return false;
    }
    const current = document.getElementById(`analysis-explorer-${jobId}`);
    if (!current) {
      return false;
    }
    const url = new URL(`/partials/analysis/${jobId}/explorer`, window.location.origin);
    Object.entries(params || {}).forEach(function ([key, value]) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    });
    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
      return false;
    }
    const markup = await response.text();
    current.outerHTML = markup;
    const replacement = document.getElementById(`analysis-explorer-${jobId}`);
    if (replacement) {
      initializeDynamicScope(replacement);
    }
    return true;
  }

  function selectCustomPerturbation(jobId, perturbJobId) {
    reloadAnalysisExplorer(jobId, {
      mode: "perturbation",
      pert_panel_mode: "custom",
      selected_custom_job_id: perturbJobId,
    });
    return false;
  }

  function openCustomPerturbation(jobId, perturbJobId) {
    reloadAnalysisExplorer(jobId, {
      mode: "perturbation",
      pert_panel_mode: "custom",
      selected_custom_job_id: perturbJobId,
    });
  }

  async function refreshElementFromHxGet(elementId) {
    const element = document.getElementById(elementId);
    if (!element) {
      return;
    }
    const endpoint = element.getAttribute("hx-get");
    if (!endpoint) {
      return;
    }
    if (elementId === "dashboard-panel") {
      saveDashboardState();
    }
    const response = await fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      return;
    }
    element.innerHTML = await response.text();
    if (elementId === "dashboard-panel") {
      restoreDashboardState(element);
    }
  }

  async function postAction(endpoint, formId, resultId, options) {
    const result = ensureResult(resultId);
    if (!result) {
      return;
    }
    if (options && options.confirmMessage && !window.confirm(options.confirmMessage)) {
      return;
    }
    const form = formId ? document.getElementById(formId) : null;
    result.textContent = "Working...";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" },
      body: form ? new URLSearchParams(new FormData(form)) : "",
    });
    const payload = await parseResponse(response);
    if (response.ok && options && options.successMessage) {
      result.textContent = String(options.successMessage);
    } else {
      result.textContent = JSON.stringify(payload, null, 2);
    }
    if (response.ok && options && options.redirectTo) {
      window.location.href = options.redirectTo;
      return;
    }
    if (response.ok && options && options.reloadOnSuccess) {
      window.location.reload();
      return;
    }
    if (response.ok && options && options.refreshDashboardOnSuccess) {
      refreshElementFromHxGet("dashboard-panel");
      return;
    }
    if (response.ok && options && options.refreshExplorerOnSuccess) {
      const refreshOptions = options.refreshExplorerOnSuccess || {};
      reloadAnalysisExplorer(refreshOptions.jobId, {
        mode: refreshOptions.mode || "perturbation",
        pert_panel_mode: refreshOptions.pertPanelMode || "custom",
        selected_custom_job_id: refreshOptions.selectedCustomJobId || "",
      });
    }
  }

  function setPath(targetId, value) {
    const input = document.getElementById(targetId) || document.querySelector(`[name="${targetId}"]`);
    if (input) {
      input.value = value;
    }
  }

  function readTabGroupState() {
    try {
      const raw = window.sessionStorage.getItem(TAB_GROUP_STATE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_error) {
      return {};
    }
  }

  function writeTabGroupState(state) {
    window.sessionStorage.setItem(TAB_GROUP_STATE_KEY, JSON.stringify(state));
  }

  function readCollapsibleGroupState() {
    try {
      const raw = window.sessionStorage.getItem(COLLAPSIBLE_GROUP_STATE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_error) {
      return {};
    }
  }

  function writeCollapsibleGroupState(state) {
    window.sessionStorage.setItem(COLLAPSIBLE_GROUP_STATE_KEY, JSON.stringify(state));
  }

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function debounce(callback, delayMs) {
    let timerId = null;
    return function debounced() {
      const args = Array.from(arguments);
      const context = this;
      if (timerId) {
        window.clearTimeout(timerId);
      }
      timerId = window.setTimeout(function () {
        timerId = null;
        callback.apply(context, args);
      }, delayMs);
    };
  }

  function buildUmapCategoryPalette(labels) {
    const palette = new Map();
    Array.from(new Set((labels || []).map(function (label) { return String(label || "(blank)"); })))
      .sort(function (left, right) {
        return left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" });
      })
      .forEach(function (label, index) {
        palette.set(label, UNAGI_CATEGORICAL_COLORS[index % UNAGI_CATEGORICAL_COLORS.length]);
      });
    return palette;
  }

  function ensurePlotly() {
    if (window.Plotly) {
      return Promise.resolve(window.Plotly);
    }
    if (plotlyPromise) {
      return plotlyPromise;
    }
    plotlyPromise = new Promise(function (resolve, reject) {
      const existing = document.getElementById("unagi-plotly-script");
      if (existing) {
        existing.addEventListener("load", function () {
          if (window.Plotly) {
            resolve(window.Plotly);
            return;
          }
          reject(new Error("Plotly loaded, but window.Plotly is unavailable."));
        }, { once: true });
        existing.addEventListener("error", function () {
          reject(new Error("Failed to load Plotly."));
        }, { once: true });
        return;
      }
      const script = document.createElement("script");
      script.id = "unagi-plotly-script";
      script.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
      script.async = true;
      script.onload = function () {
        if (window.Plotly) {
          resolve(window.Plotly);
          return;
        }
        reject(new Error("Plotly loaded, but window.Plotly is unavailable."));
      };
      script.onerror = function () {
        reject(new Error("Failed to load Plotly from the CDN."));
      };
      document.head.appendChild(script);
    });
    return plotlyPromise;
  }

  function selectTabGroup(groupId, panelId) {
    const group = document.getElementById(groupId);
    if (!group) {
      return;
    }
    group.querySelectorAll("[data-tab-target]").forEach(function (button) {
      const active = button.getAttribute("data-tab-target") === panelId;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    group.querySelectorAll("[data-tab-panel]").forEach(function (panel) {
      panel.classList.toggle("is-active", panel.getAttribute("data-tab-panel") === panelId);
    });
    const state = readTabGroupState();
    state[groupId] = panelId;
    writeTabGroupState(state);
    window.setTimeout(function () {
      resizePlotlyWithin(group);
    }, 0);
  }

  function resizePlotlyWithin(scope) {
    if (!scope || !window.Plotly || !window.Plotly.Plots || typeof window.Plotly.Plots.resize !== "function") {
      return;
    }
    scope.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
      try {
        window.Plotly.Plots.resize(plot);
      } catch (_error) {
      }
    });
  }

  function initTabGroups(root) {
    const state = readTabGroupState();
    const scope = root || document;
    scope.querySelectorAll("[data-tab-group]").forEach(function (group) {
      const groupId = group.id;
      if (!groupId) {
        return;
      }
      if (group.dataset.tabGroupBound !== "true") {
        group.querySelectorAll("[data-tab-target]").forEach(function (button) {
          button.addEventListener("click", function () {
            const target = button.getAttribute("data-tab-target");
            if (target) {
              selectTabGroup(groupId, target);
            }
          });
        });
        group.dataset.tabGroupBound = "true";
      }
      const stored = state[groupId];
      const firstButton = group.querySelector("[data-tab-target]");
      const hasStoredTarget = stored && group.querySelector(`[data-tab-target="${stored}"]`);
      const target = hasStoredTarget ? stored : (firstButton ? firstButton.getAttribute("data-tab-target") : null);
      if (target) {
        selectTabGroup(groupId, target);
      }
    });
  }

  function persistCollapsibleGroupState(group) {
    if (!group) {
      return;
    }
    const groupId = group.getAttribute("data-collapsible-group") || group.id;
    if (!groupId) {
      return;
    }
    const state = readCollapsibleGroupState();
    state[groupId] = Array.from(group.querySelectorAll("details[data-collapsible-id][open]"))
      .map(function (item) { return item.getAttribute("data-collapsible-id"); })
      .filter(Boolean);
    writeCollapsibleGroupState(state);
  }

  function restoreCollapsibleGroupState(group) {
    if (!group) {
      return;
    }
    const groupId = group.getAttribute("data-collapsible-group") || group.id;
    if (!groupId) {
      return;
    }
    const state = readCollapsibleGroupState();
    const openIds = Array.isArray(state[groupId]) ? state[groupId] : [];
    group.querySelectorAll("details[data-collapsible-id]").forEach(function (item) {
      item.open = openIds.indexOf(item.getAttribute("data-collapsible-id")) !== -1;
    });
  }

  function initCollapsibleGroups(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-collapsible-group]").forEach(function (group) {
      restoreCollapsibleGroupState(group);
      if (group.dataset.collapsibleBound === "true") {
        return;
      }
      group.addEventListener("toggle", function (event) {
        if (event.target && event.target.matches("details[data-collapsible-id]")) {
          persistCollapsibleGroupState(group);
        }
      }, true);
      group.dataset.collapsibleBound = "true";
    });
  }

  function renderSimpleTable(target, columns, rows, emptyMessage) {
    if (!target) {
      return;
    }
    if (!rows.length) {
      target.innerHTML = `<p class="muted">${escapeHtml(emptyMessage)}</p>`;
      return;
    }
    const head = columns.map(function (column) {
      return `<th>${escapeHtml(column)}</th>`;
    }).join("");
    const body = rows.map(function (row) {
      return `<tr>${row.map(function (cell) {
        return `<td>${escapeHtml(cell)}</td>`;
      }).join("")}</tr>`;
    }).join("");
    target.innerHTML = `<table class="job-table compact-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function renderSimpleTableLimited(target, columns, rows, emptyMessage, limit) {
    const safeRows = Array.isArray(rows) ? rows : [];
    const maxRows = Number.isFinite(limit) ? Math.max(1, Math.floor(limit)) : safeRows.length;
    const truncated = safeRows.slice(0, maxRows);
    renderSimpleTable(target, columns, truncated, emptyMessage);
    if (target && safeRows.length > truncated.length) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = `Showing ${truncated.length} of ${safeRows.length} rows.`;
      target.appendChild(note);
    }
  }

  function buildUmapHoverText(point, payload) {
    const lines = [`Cell ID: ${escapeHtml(point.cell_id)}`];
    if (payload.expression_gene && typeof point.expression === "number" && Number.isFinite(point.expression)) {
      lines.push(`${escapeHtml(payload.expression_gene)}: ${point.expression.toFixed(4)}`);
    } else if (payload.color_by && point.color_value !== undefined) {
      lines.push(`${escapeHtml(payload.color_by)}: ${escapeHtml(point.color_value || "(blank)")}`);
    }
    return lines.join("<br>");
  }

  function buildUmapDensityText(payload) {
    if (!payload.density) {
      return [];
    }
    return payload.density.z.map(function (row, rowIndex) {
      return row.map(function (value, colIndex) {
        const count = payload.density.counts[rowIndex][colIndex];
        if (!count) {
          return "No cells in this bin";
        }
        if (payload.density.metric === "mean_expression" && payload.expression_gene) {
          return `Cells: ${count}<br>${escapeHtml(payload.expression_gene)} mean: ${Number(value || 0).toFixed(4)}`;
        }
        return `Cells: ${count}`;
      });
    });
  }

  function buildUmapTraces(payload) {
    if (payload.render_mode === "density" && payload.density) {
      return [
        {
          type: "heatmap",
          name: payload.density.metric === "mean_expression" && payload.expression_gene ? payload.expression_gene : "Cell density",
          x: payload.density.x,
          y: payload.density.y,
          z: payload.density.z,
          text: buildUmapDensityText(payload),
          hovertemplate: "%{text}<extra></extra>",
          colorscale: payload.density.metric === "mean_expression" ? "Viridis" : "YlOrRd",
          colorbar: {
            title: payload.density.metric === "mean_expression" && payload.expression_gene
              ? `${payload.expression_gene} mean`
              : "Cells",
          },
          hoverongaps: false,
          zsmooth: "best",
        },
      ];
    }
    if (payload.expression_gene) {
      return [
        {
          type: "scattergl",
          mode: "markers",
          name: payload.expression_gene,
          x: payload.points.map(function (point) { return Number(point.x); }),
          y: payload.points.map(function (point) { return Number(point.y); }),
          customdata: payload.points.map(function (point) { return Number(point.cell_id); }),
          text: payload.points.map(function (point) { return buildUmapHoverText(point, payload); }),
          hovertemplate: "%{text}<extra></extra>",
          marker: {
            size: 6,
            opacity: 0.82,
            color: payload.points.map(function (point) { return Number(point.expression || 0); }),
            colorscale: "Viridis",
            colorbar: { title: payload.expression_gene },
          },
          selected: { marker: { size: 8, opacity: 1, line: { width: 1, color: "#182024" } } },
          unselected: { marker: { opacity: 0.12 } },
        },
      ];
    }
    const groups = new Map();
    payload.points.forEach(function (point) {
      const groupKey = String(point.color_value || "(blank)");
      if (!groups.has(groupKey)) {
        groups.set(groupKey, { x: [], y: [], customdata: [], text: [] });
      }
      const bucket = groups.get(groupKey);
      bucket.x.push(Number(point.x));
      bucket.y.push(Number(point.y));
      bucket.customdata.push(Number(point.cell_id));
      bucket.text.push(buildUmapHoverText(point, payload));
    });
    const payloadPalette = payload && payload.color_palette && typeof payload.color_palette === "object"
      ? payload.color_palette
      : null;
    const colorPalette = payloadPalette || Object.fromEntries(buildUmapCategoryPalette(Array.from(groups.keys())));
    return Array.from(groups.entries())
      .sort(function (left, right) {
        return left[0].localeCompare(right[0], undefined, { numeric: true, sensitivity: "base" });
      })
      .map(function (entry) {
        return {
          entry: entry,
          color: colorPalette[entry[0]] || UNAGI_CATEGORICAL_COLORS[0],
        };
      })
      .map(function (entry) {
        return {
          type: "scattergl",
          mode: "markers",
          name: entry.entry[0],
          x: entry.entry[1].x,
          y: entry.entry[1].y,
          customdata: entry.entry[1].customdata,
          text: entry.entry[1].text,
          hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
          marker: { size: 6, opacity: 0.72, color: entry.color },
          selected: { marker: { size: 8, opacity: 1, line: { width: 1, color: "#182024" } } },
          unselected: { marker: { opacity: 0.12 } },
        };
      });
  }

  function getUmapViewport(plotDiv) {
    if (!plotDiv || !plotDiv.layout || !plotDiv.layout.xaxis || !plotDiv.layout.yaxis) {
      return null;
    }
    const xRange = plotDiv.layout.xaxis.range;
    const yRange = plotDiv.layout.yaxis.range;
    if (!Array.isArray(xRange) || !Array.isArray(yRange)) {
      return null;
    }
    const xMin = Number(xRange[0]);
    const xMax = Number(xRange[1]);
    const yMin = Number(yRange[0]);
    const yMax = Number(yRange[1]);
    if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) {
      return null;
    }
    return { x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax };
  }

  function normalizeUmapViewport(viewport) {
    if (!viewport) {
      return null;
    }
    const normalized = {
      x_min: Number(viewport.x_min),
      x_max: Number(viewport.x_max),
      y_min: Number(viewport.y_min),
      y_max: Number(viewport.y_max),
    };
    if (!Object.values(normalized).every(Number.isFinite)) {
      return null;
    }
    return normalized;
  }

  function viewportChanged(previousViewport, nextViewport) {
    const prev = normalizeUmapViewport(previousViewport);
    const next = normalizeUmapViewport(nextViewport);
    if (!prev && !next) {
      return false;
    }
    if (!prev || !next) {
      return true;
    }
    return ["x_min", "x_max", "y_min", "y_max"].some(function (key) {
      return Math.abs(prev[key] - next[key]) > 1e-6;
    });
  }

  function isViewportRelayout(eventData) {
    if (!eventData || typeof eventData !== "object") {
      return false;
    }
    return Object.keys(eventData).some(function (key) {
      return (
        key.indexOf("xaxis.range") === 0
        || key.indexOf("yaxis.range") === 0
        || key === "xaxis.autorange"
        || key === "yaxis.autorange"
      );
    });
  }

  function updateUmapRenderStatus(shell, payload) {
    if (!shell || !payload) {
      return;
    }
    const modeBadge = shell.querySelector("[data-umap-render-mode]");
    const modeNote = shell.querySelector("[data-umap-render-note]");
    const toolbarNote = shell.querySelector("[data-umap-toolbar-note]");
    const isDensity = payload.render_mode === "density";
    const isSampled = payload.render_mode === "sampled_points";
    if (modeBadge) {
      modeBadge.hidden = true;
    }
    if (modeNote) {
      modeNote.hidden = true;
    }
    if (toolbarNote) {
      if (isDensity) {
        toolbarNote.textContent = "Large views switch to density mode. Zoom into a smaller region to inspect exact cells and use cell selection.";
      } else if (isSampled) {
        toolbarNote.textContent = "Large metadata views use sampled colored points so labels remain visible. Zoom further to enable exact cell selection.";
      } else {
        toolbarNote.textContent = "Use the chart toolbar for pan, zoom, box select, and lasso select to inspect cell selections.";
      }
    }
  }

  function umapSupportsSelection(payload) {
    return Boolean(payload && payload.selection_supported);
  }

  function renderEmptyUmapSummary(shell, payload, message) {
    const selectedCount = shell ? shell.querySelector("[data-umap-selected-count]") : null;
    const selectionNote = shell ? shell.querySelector("[data-umap-selection-note]") : null;
    const summaryTarget = shell ? shell.querySelector("[data-umap-summary-table]") : null;
    if (selectedCount) {
      selectedCount.textContent = "0";
    }
    if (selectionNote) {
      selectionNote.textContent = message;
    }
    renderSimpleTable(
      summaryTarget,
      [],
      [],
      payload && payload.render_mode === "density"
        ? "Zoom further to inspect individual cells and request exact summaries."
        : (payload && payload.render_mode === "sampled_points"
          ? "Zoom further to switch from sampled metadata points to exact cell selection."
          : (payload && payload.expression_gene
          ? `Select cells to see ${payload.expression_gene} expression summary.`
          : "Select cells to see counts by the current color field."))
    );
  }

  async function restoreUmapSelection(widget, plotDiv, payload) {
    const shell = widget.closest(".analysis-umap-shell");
    const selected = Array.isArray(widget._selectedCellIds) ? widget._selectedCellIds.slice() : [];
    if (!shell) {
      return;
    }
    if (!umapSupportsSelection(payload)) {
      widget._selectedCellIds = [];
      renderEmptyUmapSummary(
        shell,
        payload,
        payload.render_mode === "sampled_points"
          ? "Sampled metadata mode is inspect-only. Zoom further to enable exact cell selection."
          : "Density mode is inspect-only. Zoom further to enable exact cell selection."
      );
      return;
    }
    if (!selected.length) {
      renderEmptyUmapSummary(shell, payload, "No cells selected.");
      return;
    }

    const selectedSet = new Set(selected.map(function (value) { return Number(value); }).filter(Number.isFinite));
    const traceSelection = [];
    let matchedCount = 0;
    (plotDiv.data || []).forEach(function (trace) {
      const traceCustom = Array.isArray(trace.customdata) ? trace.customdata : [];
      const traceIndices = [];
      traceCustom.forEach(function (cellId, index) {
        if (selectedSet.has(Number(cellId))) {
          traceIndices.push(index);
          matchedCount += 1;
        }
      });
      traceSelection.push(traceIndices.length ? traceIndices : null);
    });

    if (!matchedCount) {
      widget._selectedCellIds = [];
      renderEmptyUmapSummary(shell, payload, "Selected cells are outside the current view.");
      return;
    }

    await Promise.all(traceSelection.map(function (indices, traceIndex) {
      return window.Plotly.restyle(plotDiv, { selectedpoints: [indices] }, [traceIndex]);
    }));
    await updateUmapSelection(shell, widget, payload, Array.from(selectedSet));
  }

  async function updateUmapSelection(shell, widget, payload, cellIds) {
    if (!shell) {
      return;
    }
    const selectedCount = shell.querySelector("[data-umap-selected-count]");
    const selectionNote = shell.querySelector("[data-umap-selection-note]");
    const summaryTarget = shell.querySelector("[data-umap-summary-table]");
    const selected = Array.from(new Set((cellIds || []).map(function (cellId) {
      return Number(cellId);
    }).filter(Number.isFinite)));
    widget._selectedCellIds = selected;
    if (selectedCount) {
      selectedCount.textContent = String(selected.length);
    }
    if (!umapSupportsSelection(payload)) {
      renderEmptyUmapSummary(
        shell,
        payload,
        payload.render_mode === "sampled_points"
          ? "Sampled metadata mode is inspect-only. Zoom further to enable exact cell selection."
          : "Density mode is inspect-only. Zoom further to enable exact cell selection."
      );
      return;
    }
    if (!selected.length) {
      renderEmptyUmapSummary(shell, payload, "No cells selected.");
      return;
    }
    if (selectionNote) {
      selectionNote.textContent = `Loading summary for ${selected.length} selected cells...`;
    }
    if (summaryTarget) {
      summaryTarget.innerHTML = '<p class="muted">Loading selection summary...</p>';
    }
    widget._summaryRequestId = (widget._summaryRequestId || 0) + 1;
    const requestId = widget._summaryRequestId;
    try {
      const response = await fetch(widget.dataset.umapSummaryUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stage: payload.stage,
          color_by: payload.color_by,
          gene: payload.expression_gene || "",
          cell_ids: selected,
          render_mode: payload.render_mode,
        }),
      });
      const summary = await parseResponse(response);
      if (requestId !== widget._summaryRequestId) {
        return;
      }
      if (!response.ok) {
        throw new Error(summary.detail || "Failed to load selection summary.");
      }
      if (selectionNote) {
        selectionNote.textContent = summary.message || `Selected ${selected.length} cells.`;
      }
      renderSimpleTable(summaryTarget, summary.columns || [], summary.rows || [], summary.message || "No cells selected.");
    } catch (error) {
      if (requestId !== widget._summaryRequestId) {
        return;
      }
      if (selectionNote) {
        selectionNote.textContent = "Selection summary failed to load.";
      }
      if (summaryTarget) {
        summaryTarget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
      }
    }
  }

  function bindUmapSelectionEvents(widget, plotDiv, payload) {
    const shell = widget.closest(".analysis-umap-shell");
    if (!shell) {
      return;
    }
    plotDiv.on("plotly_selected", function (eventData) {
      const cellIds = eventData && eventData.points
        ? Array.from(new Set(eventData.points.map(function (point) { return Number(point.customdata); }))).filter(Number.isFinite)
        : [];
      updateUmapSelection(shell, widget, payload, cellIds);
    });

    plotDiv.on("plotly_click", function (eventData) {
      const cellIds = eventData && eventData.points && eventData.points.length
        ? [Number(eventData.points[0].customdata)].filter(Number.isFinite)
        : [];
      updateUmapSelection(shell, widget, payload, cellIds);
    });

    plotDiv.on("plotly_deselect", function () {
      updateUmapSelection(shell, widget, payload, []);
    });
  }

  function getUmapColorMode(shell) {
    if (!shell) {
      return "metadata";
    }
    const active = shell.querySelector("[data-umap-color-mode-option].is-active");
    return active ? String(active.getAttribute("data-umap-color-mode-option") || "metadata") : "metadata";
  }

  function setUmapColorMode(shell, colorMode) {
    if (!shell) {
      return;
    }
    shell.querySelectorAll("[data-umap-color-mode-option]").forEach(function (button) {
      const active = button.getAttribute("data-umap-color-mode-option") === colorMode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    const fieldGroup = shell.querySelector("[data-umap-field-group]");
    const geneGroup = shell.querySelector("[data-umap-gene-group]");
    const clearGeneButton = shell.querySelector("[data-umap-gene-clear]");
    if (fieldGroup) {
      fieldGroup.hidden = colorMode !== "metadata";
    }
    if (geneGroup) {
      geneGroup.hidden = colorMode !== "gene";
    }
    if (clearGeneButton) {
      clearGeneButton.hidden = colorMode !== "gene";
    }
  }

  function updateUmapPublicationLink(shell, colorBy, geneValue, colorMode) {
    if (!shell) {
      return;
    }
    const link = shell.querySelector("[data-umap-publication]");
    if (!link) {
      return;
    }
    const url = new URL(link.getAttribute("href"), window.location.origin);
    url.searchParams.set("umap_color_by", colorBy || "");
    if (colorMode === "gene" && geneValue) {
      url.searchParams.set("umap_gene", geneValue);
    } else {
      url.searchParams.delete("umap_gene");
    }
    link.setAttribute("href", `${url.pathname}${url.search}`);
  }

  function getUmapScreenSize(widget, plotDiv) {
    const width = Math.max(320, Math.round((plotDiv && plotDiv.clientWidth) || widget.clientWidth || 1200));
    const height = Math.max(420, Math.round((plotDiv && plotDiv.clientHeight) || 620));
    return { screen_width: width, screen_height: height };
  }

  async function loadAnalysisUmapPayload(widget, options) {
    const params = options || {};
    const existingSelection = Array.isArray(widget._selectedCellIds) ? widget._selectedCellIds.slice() : [];
    const url = new URL(widget.dataset.umapUrl, window.location.origin);
    if (params.colorBy) {
      url.searchParams.set("color_by", params.colorBy);
    }
    if (params.geneValue) {
      url.searchParams.set("gene", params.geneValue);
    } else {
      url.searchParams.delete("gene");
    }
    if (params.viewport) {
      url.searchParams.set("x_min", params.viewport.x_min);
      url.searchParams.set("x_max", params.viewport.x_max);
      url.searchParams.set("y_min", params.viewport.y_min);
      url.searchParams.set("y_max", params.viewport.y_max);
    } else {
      ["x_min", "x_max", "y_min", "y_max"].forEach(function (field) {
        url.searchParams.delete(field);
      });
    }
    if (params.screenWidth) {
      url.searchParams.set("screen_width", params.screenWidth);
    }
    if (params.screenHeight) {
      url.searchParams.set("screen_height", params.screenHeight);
    }
    if (params.currentRenderMode) {
      url.searchParams.set("current_render_mode", params.currentRenderMode);
    } else {
      url.searchParams.delete("current_render_mode");
    }
    widget._requestId = (widget._requestId || 0) + 1;
    const requestId = widget._requestId;
    const response = await fetch(`${url.pathname}${url.search}`, { method: "GET" });
    const payload = await parseResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to load UMAP data.");
    }
    if (requestId !== widget._requestId) {
      return null;
    }
    widget._payload = payload;
    widget._viewport = payload.viewport || null;
    widget._selectedCellIds = existingSelection;
    return payload;
  }

  function buildUmapLayout(payload) {
    const supportsSelection = umapSupportsSelection(payload);
    return {
      height: 620,
      dragmode: supportsSelection ? "lasso" : "pan",
      hovermode: "closest",
      margin: { l: 48, r: 18, t: 12, b: 48 },
      legend: payload.render_mode === "points" && !payload.expression_gene ? { orientation: "h", y: -0.18 } : { orientation: "h", y: -0.15 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        title: "UMAP 1",
        zeroline: false,
        range: [payload.viewport.x_min, payload.viewport.x_max],
      },
      yaxis: {
        title: "UMAP 2",
        zeroline: false,
        scaleanchor: "x",
        scaleratio: 1,
        range: [payload.viewport.y_min, payload.viewport.y_max],
      },
    };
  }

  function buildUmapConfig(payload) {
    const buttonsToRemove = ["autoScale2d"];
    if (!umapSupportsSelection(payload)) {
      buttonsToRemove.push("lasso2d", "select2d");
    }
    return {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: buttonsToRemove,
    };
  }

  function queueUmapViewportRefresh(widget, plotDiv) {
    if (!widget) {
      return;
    }
    if (widget._viewportRefreshTimer) {
      window.clearTimeout(widget._viewportRefreshTimer);
    }
    widget._viewportRefreshTimer = window.setTimeout(function () {
      const shell = widget.closest(".analysis-umap-shell");
      const colorSelect = shell ? shell.querySelector("[data-umap-control='color-by']") : null;
      const geneInput = shell ? shell.querySelector("[data-umap-control='gene']") : null;
      const colorMode = getUmapColorMode(shell);
      const viewport = getUmapViewport(plotDiv);
      const screen = getUmapScreenSize(widget, plotDiv);
      refreshAnalysisUmap(widget, {
        colorBy: colorSelect ? colorSelect.value : (widget._payload ? widget._payload.color_by : ""),
        colorMode: colorMode,
        geneValue: colorMode === "gene" && geneInput ? String(geneInput.value || "").trim() : "",
        viewport: viewport,
        screenWidth: screen.screen_width,
        screenHeight: screen.screen_height,
        currentRenderMode: widget._payload ? widget._payload.render_mode : "",
      }).catch(function (error) {
        widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
      });
    }, 180);
  }

  function bindUmapPlotEvents(widget, plotDiv, payload) {
    const shell = widget.closest(".analysis-umap-shell");
    if (!shell) {
      return;
    }
    if (typeof plotDiv.removeAllListeners === "function") {
      plotDiv.removeAllListeners("plotly_selected");
      plotDiv.removeAllListeners("plotly_click");
      plotDiv.removeAllListeners("plotly_deselect");
      plotDiv.removeAllListeners("plotly_relayout");
    }

    if (umapSupportsSelection(payload)) {
      bindUmapSelectionEvents(widget, plotDiv, payload);
    }

    plotDiv.on("plotly_relayout", function (_eventData) {
      const now = Date.now();
      if (widget._ignoreRelayoutUntil && now < widget._ignoreRelayoutUntil) {
        return;
      }
      if (!isViewportRelayout(_eventData)) {
        return;
      }
      const nextViewport = getUmapViewport(plotDiv);
      if (!viewportChanged(widget._payload ? widget._payload.viewport : null, nextViewport)) {
        return;
      }
      queueUmapViewportRefresh(widget, plotDiv);
    });
  }

  async function renderAnalysisUmapWidget(widget, colorBy, colorMode) {
    const payload = widget._payload;
    if (!payload || !window.Plotly) {
      return;
    }
    let plotDiv = widget.querySelector(".analysis-umap-plot");
    if (!plotDiv) {
      widget.innerHTML = '<div class="analysis-umap-plot"></div>';
      plotDiv = widget.querySelector(".analysis-umap-plot");
    }
    const traces = buildUmapTraces(payload);
    await window.Plotly.react(
      plotDiv,
      traces,
      buildUmapLayout(payload),
      buildUmapConfig(payload),
    );
    widget._ignoreRelayoutUntil = Date.now() + 220;
    bindUmapPlotEvents(widget, plotDiv, payload);
    const shell = widget.closest(".analysis-umap-shell");
    if (shell) {
      const geneInput = shell.querySelector("[data-umap-control='gene']");
      const activeMode = colorMode || getUmapColorMode(shell);
      const activeGene = activeMode === "gene" && geneInput ? String(geneInput.value || "").trim() : "";
      updateUmapPublicationLink(shell, colorBy, activeGene, activeMode);
      updateUmapRenderStatus(shell, payload);
      await restoreUmapSelection(widget, plotDiv, payload);
    }
  }

  async function refreshAnalysisUmap(widget, options) {
    const params = options || {};
    const shell = widget.closest(".analysis-umap-shell");
    const colorSelect = shell ? shell.querySelector("[data-umap-control='color-by']") : null;
    const geneInput = shell ? shell.querySelector("[data-umap-control='gene']") : null;
    const colorMode = params.colorMode || getUmapColorMode(shell);
    const geneValue = params.geneValue !== undefined
      ? String(params.geneValue || "").trim()
      : (colorMode === "gene" && geneInput ? String(geneInput.value || "").trim() : "");
    const colorBy = params.colorBy || (colorSelect ? colorSelect.value : widget.dataset.defaultColor || "");
    const plotDiv = widget.querySelector(".analysis-umap-plot");
    const screen = getUmapScreenSize(widget, plotDiv);
    const payload = await loadAnalysisUmapPayload(widget, {
      colorBy: colorBy,
      geneValue: geneValue,
      viewport: Object.prototype.hasOwnProperty.call(params, "viewport") ? params.viewport : widget._viewport,
      screenWidth: params.screenWidth || screen.screen_width,
      screenHeight: params.screenHeight || screen.screen_height,
      currentRenderMode: params.currentRenderMode || (widget._payload ? widget._payload.render_mode : ""),
    });
    if (!payload) {
      return;
    }
    await renderAnalysisUmapWidget(widget, colorBy, colorMode);
  }

  async function applyUmapGeneColoring(widget, colorSelect, geneInput) {
    const geneValue = geneInput ? String(geneInput.value || "").trim() : "";
    if (!geneValue) {
      if (geneInput) {
        geneInput.focus();
      }
      return;
    }
    await refreshAnalysisUmap(widget, {
      colorBy: colorSelect ? colorSelect.value : "",
      colorMode: "gene",
      geneValue: geneValue,
    });
  }

  async function initAnalysisUmapWidget(widget) {
    if (!widget || widget.dataset.initialized === "loading" || widget.dataset.initialized === "true") {
      return;
    }
    widget.dataset.initialized = "loading";
    try {
      await ensurePlotly();
      const shell = widget.closest(".analysis-umap-shell");
      let colorSelect = null;
      if (shell) {
        const defaultColorMode = widget.dataset.defaultColorMode || "metadata";
        colorSelect = shell.querySelector("[data-umap-control='color-by']");
        const geneInput = shell.querySelector("[data-umap-control='gene']");
        const clearGeneButton = shell.querySelector("[data-umap-gene-clear]");
        const colorModeButtons = shell.querySelectorAll("[data-umap-color-mode-option]");
        setUmapColorMode(shell, defaultColorMode);
        if (colorSelect) {
          if (!colorSelect.dataset.bound) {
            colorSelect.addEventListener("change", function () {
              if (getUmapColorMode(shell) === "metadata") {
                refreshAnalysisUmap(widget, {
                  colorBy: colorSelect.value,
                  colorMode: "metadata",
                }).catch(function (error) {
                  widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
                });
              }
            });
            colorSelect.dataset.bound = "true";
          }
        }
        colorModeButtons.forEach(function (button) {
          if (button.dataset.bound === "true") {
            return;
          }
          button.addEventListener("click", async function () {
            const mode = button.getAttribute("data-umap-color-mode-option") || "metadata";
            setUmapColorMode(shell, mode);
            try {
              if (mode === "gene") {
                if (geneInput && String(geneInput.value || "").trim()) {
                  await applyUmapGeneColoring(widget, colorSelect, geneInput);
                } else if (geneInput) {
                  geneInput.focus();
                }
                return;
              }
              await refreshAnalysisUmap(widget, {
                colorBy: colorSelect ? colorSelect.value : widget.dataset.defaultColor || "",
                colorMode: "metadata",
                geneValue: "",
              });
            } catch (error) {
              widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
            }
          });
          button.dataset.bound = "true";
        });
        const clearButton = shell.querySelector("[data-umap-clear]");
        if (clearButton && !clearButton.dataset.bound) {
          clearButton.addEventListener("click", function () {
            widget._selectedCellIds = [];
            renderAnalysisUmapWidget(widget, colorSelect ? colorSelect.value : widget.dataset.defaultColor || "", getUmapColorMode(shell));
          });
          clearButton.dataset.bound = "true";
        }
        if (clearGeneButton && geneInput && !clearGeneButton.dataset.bound) {
          clearGeneButton.addEventListener("click", async function () {
            geneInput.value = "";
            try {
              clearGeneButton.disabled = true;
              setUmapColorMode(shell, "metadata");
              await refreshAnalysisUmap(widget, {
                colorBy: colorSelect ? colorSelect.value : widget.dataset.defaultColor || "",
                colorMode: "metadata",
                geneValue: "",
              });
            } catch (error) {
              widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
            } finally {
              clearGeneButton.disabled = false;
            }
          });
          clearGeneButton.dataset.bound = "true";
        }
        if (geneInput && !geneInput.dataset.bound) {
          geneInput.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
              event.preventDefault();
              if (getUmapColorMode(shell) === "gene") {
                applyUmapGeneColoring(widget, colorSelect, geneInput).catch(function (error) {
                  widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
                });
              }
            }
          });
          geneInput.addEventListener("change", function () {
            if (getUmapColorMode(shell) === "gene" && String(geneInput.value || "").trim()) {
              applyUmapGeneColoring(widget, colorSelect, geneInput).catch(function (error) {
                widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
              });
            }
          });
          geneInput.dataset.bound = "true";
        }
      }
      await refreshAnalysisUmap(widget, {
        colorBy: widget.dataset.defaultColor || "",
        colorMode: widget.dataset.defaultColorMode || "metadata",
        geneValue: shell && getUmapColorMode(shell) === "gene"
          ? String((shell.querySelector("[data-umap-control='gene']") || {}).value || "").trim()
          : "",
        viewport: null,
        currentRenderMode: "",
      });
      widget.dataset.initialized = "true";
    } catch (error) {
      widget.dataset.initialized = "error";
      widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
    }
  }

  function initAnalysisUmapWidgets(root) {
    const scope = root || document;
    scope.querySelectorAll(".analysis-umap-widget[data-umap-url]").forEach(function (widget) {
      initAnalysisUmapWidget(widget);
    });
  }

  function buildTemporalGraphHoverText(node) {
    const lines = [
      `Stage: ${escapeHtml(node.stage)}`,
      `Cluster: ${escapeHtml(node.cluster)}`,
      `ident: ${escapeHtml(node.cell_type || "NA")}`,
      `Cells: ${escapeHtml(node.n_cells)}`,
    ];
    if (node.track_label) {
      lines.push(`Track: ${escapeHtml(node.track_label)}`);
    }
    return lines.join("<br>");
  }

  function buildTemporalGraphTraces(payload) {
    const traces = [];
    if (Array.isArray(payload.edges) && payload.edges.length) {
      const edgeX = [];
      const edgeY = [];
      payload.edges.forEach(function (edge) {
        edgeX.push(Number(edge.x0), Number(edge.x1), null);
        edgeY.push(Number(edge.y0), Number(edge.y1), null);
      });
      traces.push({
        type: "scattergl",
        mode: "lines",
        x: edgeX,
        y: edgeY,
        hoverinfo: "skip",
        line: { color: "rgba(106, 113, 118, 0.55)", width: 1.6 },
        showlegend: false,
      });
    }

    const groups = new Map();
    (payload.nodes || []).forEach(function (node, index) {
      const groupKey = String(node.cell_type || "NA");
      if (!groups.has(groupKey)) {
        groups.set(groupKey, {
          x: [],
          y: [],
          size: [],
          text: [],
          customdata: [],
          color: node.color || "#6E7C80",
        });
      }
      const bucket = groups.get(groupKey);
      bucket.x.push(Number(node.x));
      bucket.y.push(Number(node.y));
      bucket.size.push(Math.max(4, Math.min(16, Number(node.node_size || 0) / 42)));
      bucket.text.push(buildTemporalGraphHoverText(node));
      bucket.customdata.push(index);
    });

    Array.from(groups.entries()).forEach(function (entry) {
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: entry[0],
        x: entry[1].x,
        y: entry[1].y,
        customdata: entry[1].customdata,
        text: entry[1].text,
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
        marker: {
          size: entry[1].size,
          color: entry[1].color,
          opacity: 0.92,
          line: { width: 1, color: "#172124" },
        },
        selected: { marker: { opacity: 1, line: { width: 2, color: "#182024" } } },
        unselected: { marker: { opacity: 0.2 } },
      });
    });

    return traces;
  }

  function renderIdremTrackEmpty(panel, message) {
    if (!panel) {
      return;
    }
    panel.innerHTML = [
      '<div class="artifact-head"><strong>Track iDREM Summary</strong></div>',
      '<div class="idrem-track-panel-body">',
      `<p class="muted">${escapeHtml(message)}</p>`,
      "</div>",
    ].join("");
  }

  function findIdremPath(payload, pathKey) {
    if (!payload || !Array.isArray(payload.paths)) {
      return null;
    }
    return payload.paths.find(function (item) {
      return item.path_key === pathKey;
    }) || null;
  }

  function buildIdremPathPlotTraces(payload, selectedPathKey) {
    const plot = payload && payload.path_plot ? payload.path_plot : { paths: [] };
    return (plot.paths || []).map(function (path) {
      const active = path.path_key === selectedPathKey;
      return {
        type: "scatter",
        mode: "lines+markers",
        name: path.label,
        x: path.stage_labels,
        y: path.values,
        customdata: path.values.map(function () { return path.path_key; }),
        text: path.hover_texts || [],
        hovertemplate: "%{text}<extra></extra>",
        line: {
          width: active ? 3.4 : 1.8,
          color: active ? "#C15F31" : "#2D7D7C",
        },
        marker: {
          size: active ? 8 : 6,
          color: active ? "#C15F31" : "#2D7D7C",
        },
      };
    });
  }

  function renderIdremRegulatorSummary(panel, selected) {
    const target = panel.querySelector("[data-idrem-regulators-summary]");
    if (!target) {
      return;
    }
    const rows = selected && selected.regulators ? (selected.regulators.rows || []) : [];
    if (!rows.length) {
      target.innerHTML = '<p class="muted">No regulators were available for this path.</p>';
      return;
    }
    target.innerHTML = [
      '<div class="artifact-head compact-inline"><strong>Regulators</strong></div>',
      '<div class="idrem-regulator-chip-row">',
      rows.slice(0, 10).map(function (row) {
        const regulator = escapeHtml(row[0] || "");
        const type = escapeHtml(row[1] || "");
        const score = escapeHtml(row[2] || "");
        return `<span class="idrem-regulator-chip">${regulator}<em>${type}${score ? ` · ${score}` : ""}</em></span>`;
      }).join(""),
      "</div>",
    ].join("");
  }

  function buildCsvText(columns, rows) {
    const escapeCsv = function (value) {
      const text = String(value === null || value === undefined ? "" : value);
      if (/[",\n]/.test(text)) {
        return `"${text.replace(/"/g, '""')}"`;
      }
      return text;
    };
    const lines = [];
    lines.push(columns.map(escapeCsv).join(","));
    rows.forEach(function (row) {
      lines.push(row.map(escapeCsv).join(","));
    });
    return lines.join("\n");
  }

  function revokeObjectUrl(url) {
    if (url && String(url).startsWith("blob:")) {
      try {
        window.URL.revokeObjectURL(url);
      } catch (_error) {
        // Ignore stale blob revocation errors.
      }
    }
  }

  function setIdremDownloadLink(panel, payload) {
    const link = panel ? panel.querySelector("[data-idrem-detail-download]") : null;
    if (!link) {
      return;
    }
    revokeObjectUrl(link.dataset.objectUrl || "");
    if (!payload || !Array.isArray(payload.columns) || !payload.columns.length || !Array.isArray(payload.rows) || !payload.rows.length) {
      link.hidden = true;
      link.removeAttribute("href");
      link.removeAttribute("download");
      link.dataset.objectUrl = "";
      return;
    }
    const blob = new Blob([buildCsvText(payload.columns, payload.rows)], { type: "text/csv;charset=utf-8" });
    const url = window.URL.createObjectURL(blob);
    link.href = url;
    link.download = payload.filename || "idrem_table.csv";
    link.dataset.objectUrl = url;
    link.hidden = false;
  }

  function getIdremEnrichmentDisplayState(selected, categoryFilter) {
    const columns = selected && selected.enrichment ? (selected.enrichment.columns || []) : [];
    const allRows = selected && selected.enrichment ? (selected.enrichment.rows || []) : [];
    const allHoverTexts = selected && selected.enrichment ? (selected.enrichment.hover_texts || []) : [];
    const message = selected && selected.enrichment ? String(selected.enrichment.message || "") : "";
    const categoryIndex = columns.indexOf("Category");
    const filteredRows = [];
    const filteredHoverTexts = [];
    allRows.forEach(function (row, rowIndex) {
      if (!categoryFilter || categoryIndex < 0 || String(row[categoryIndex] || "") === categoryFilter) {
        filteredRows.push(row);
        filteredHoverTexts.push(allHoverTexts[rowIndex] || "");
      }
    });
    const hiddenColumns = new Set(["Genes in query"]);
    if (categoryFilter) {
      hiddenColumns.add("Category");
    }
    if (categoryFilter === "Pathway") {
      hiddenColumns.add("Term ID");
    }
    const displayColumns = columns.filter(function (column) {
      return !hiddenColumns.has(String(column || ""));
    });
    const displayRows = filteredRows.map(function (row) {
      return row.filter(function (_cell, cellIndex) {
        return !hiddenColumns.has(String(columns[cellIndex] || ""));
      });
    });
    return {
      columns: columns,
      rows: filteredRows,
      hover_texts: filteredHoverTexts,
      message: message,
      display_columns: displayColumns,
      display_rows: displayRows,
    };
  }

  function renderIdremEnrichmentTable(target, selected, categoryFilter) {
    if (!target) {
      return null;
    }
    const state = getIdremEnrichmentDisplayState(selected, categoryFilter);
    const filteredRows = state.rows;
    const filteredHoverTexts = state.hover_texts;
    const displayColumns = state.display_columns;
    const displayRows = state.display_rows;
    if (!filteredRows.length) {
      target.innerHTML = `<p class="muted">${escapeHtml(state.message || "No enrichment rows were available for this path.")}</p>`;
      return state;
    }
    const head = displayColumns.map(function (column) {
      return `<th>${escapeHtml(column)}</th>`;
    }).join("");
    const body = displayRows.map(function (row, rowIndex) {
      const termIndex = displayColumns.indexOf("Term Name");
      return `<tr>${row.map(function (cell, cellIndex) {
        const tooltip = cellIndex === termIndex && filteredHoverTexts.length ? ` title="${escapeHtml(filteredHoverTexts[rowIndex] || "")}"` : "";
        return `<td${tooltip}>${escapeHtml(cell || "")}</td>`;
      }).join("")}</tr>`;
    }).join("");
    target.innerHTML = `<table class="job-table compact-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    return state;
  }

  function setIdremDetailTab(panel, tabName) {
    if (!panel) {
      return;
    }
    const selected = panel._selectedPathData || null;
    const nextTab = ["genes", "biological_process", "pathways"].indexOf(tabName) !== -1 ? tabName : "genes";
    panel._idremDetailTab = nextTab;
    panel.querySelectorAll("[data-idrem-detail-tab]").forEach(function (button) {
      const active = button.getAttribute("data-idrem-detail-tab") === nextTab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    const title = panel.querySelector("[data-idrem-detail-title]");
    const target = panel.querySelector("[data-idrem-detail-table]");
    let downloadPayload = null;
    if (!target || !selected) {
      return;
    }
    if (nextTab === "biological_process") {
      if (title) {
        title.textContent = "Biological Process";
      }
      const state = renderIdremEnrichmentTable(target, selected, "GO Biological Process");
      downloadPayload = state && state.rows.length ? {
        columns: state.columns.filter(function (column) { return String(column || "") !== "Category"; }),
        rows: state.rows.map(function (row) {
          return row.filter(function (_cell, index) { return String(state.columns[index] || "") !== "Category"; });
        }),
        filename: `${String(panel._selectedPathKey || "path")}_biological_process.csv`,
      } : null;
      setIdremDownloadLink(panel, downloadPayload);
      return;
    }
    if (nextTab === "pathways") {
      if (title) {
        title.textContent = "Pathways";
      }
      const state = renderIdremEnrichmentTable(target, selected, "Pathway");
      downloadPayload = state && state.rows.length ? {
        columns: state.columns.filter(function (column) { return String(column || "") !== "Category"; }),
        rows: state.rows.map(function (row) {
          return row.filter(function (_cell, index) { return String(state.columns[index] || "") !== "Category"; });
        }),
        filename: `${String(panel._selectedPathKey || "path")}_pathways.csv`,
      } : null;
      setIdremDownloadLink(panel, downloadPayload);
      return;
    }
    if (title) {
      title.textContent = "Genes";
    }
    renderSimpleTableLimited(
      target,
      selected.genes.columns || [],
      selected.genes.rows || [],
      "No genes were assigned to this path.",
      120,
    );
    setIdremDownloadLink(panel, null);
  }

  async function renderIdremPathPlot(panel, payload, pathKey) {
    const plotDiv = panel.querySelector("[data-idrem-path-plot]");
    if (!plotDiv) {
      return;
    }
    await ensurePlotly();
    await window.Plotly.react(
      plotDiv,
      buildIdremPathPlotTraces(payload, pathKey),
      {
        height: 280,
        hovermode: "closest",
        dragmode: "pan",
        margin: { l: 48, r: 18, t: 12, b: 42 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Stage", zeroline: false },
        yaxis: { title: "Mean expression", zeroline: false, gridcolor: "rgba(31, 42, 46, 0.12)" },
        legend: { orientation: "h", y: -0.22 },
      },
      {
        responsive: true,
        displaylogo: false,
      },
    );
    if (typeof plotDiv.removeAllListeners === "function") {
      plotDiv.removeAllListeners("plotly_click");
    }
    plotDiv.on("plotly_click", function (eventData) {
      if (!eventData || !Array.isArray(eventData.points) || !eventData.points.length) {
        return;
      }
      const clickedPath = String(eventData.points[0].customdata || "");
      if (clickedPath) {
        updateIdremPathSelection(panel, payload, clickedPath);
      }
    });
  }

  function updateIdremPathSelection(panel, payload, pathKey) {
    if (!panel || !payload) {
      return;
    }
    const selected = findIdremPath(payload, pathKey) || findIdremPath(payload, payload.selected_path);
    if (!selected) {
      return;
    }
    panel.querySelectorAll("[data-idrem-path-chip]").forEach(function (button) {
      const active = button.getAttribute("data-path-key") === selected.path_key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    const selectedChip = panel.querySelector("[data-idrem-selected-path]");
    if (selectedChip) {
      selectedChip.textContent = selected.label;
    }
    panel._selectedPathKey = selected.path_key;
    panel._selectedPathData = selected;
    renderIdremRegulatorSummary(panel, selected);
    setIdremDetailTab(panel, panel._idremDetailTab || "genes");
    renderIdremPathPlot(panel, payload, selected.path_key).catch(function (error) {
      const plotDiv = panel.querySelector("[data-idrem-path-plot]");
      if (plotDiv) {
        plotDiv.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
      }
    });
  }

  async function renderIdremTrackPanel(panel, payload) {
    if (!panel) {
      return;
    }
    panel.innerHTML = [
      '<div class="artifact-head">',
      '<strong>Track iDREM Summary</strong>',
      '<div class="link-stack artifact-actions">',
      '<a class="button-link secondary" href="#" data-idrem-detail-download hidden>table</a>',
      "</div>",
      "</div>",
      '<div class="idrem-track-panel-body">',
      `<p><strong>${escapeHtml(payload.track_label || payload.track_key || "Track")}</strong></p>`,
      '<div class="idrem-track-main-grid">',
      '<section class="idrem-track-plot-section">',
      `<p class="muted">Selected path <span class="status-pill" data-idrem-selected-path>${escapeHtml((findIdremPath(payload, payload.selected_path) || {}).label || "None")}</span></p>`,
      '<div class="analysis-hierarchical-plot" data-idrem-path-plot></div>',
      '<div class="idrem-path-chip-row">',
      (payload.paths || []).map(function (path) {
        const active = path.path_key === payload.selected_path;
        return `<button class="idrem-path-chip${active ? " is-active" : ""}" type="button" data-idrem-path-chip data-path-key="${escapeHtml(path.path_key)}" aria-pressed="${active ? "true" : "false"}">${escapeHtml(path.label)}</button>`;
      }).join(""),
      "</div>",
      '<section class="idrem-path-section idrem-regulator-section">',
      '<div data-idrem-regulators-summary></div>',
      '</section>',
      "</section>",
      '<section class="idrem-track-detail-section">',
      '<div class="module-tab-nav idrem-detail-tab-row">',
      '<button class="module-tab-button is-active" type="button" data-idrem-detail-tab="genes" aria-pressed="true">Genes</button>',
      '<button class="module-tab-button" type="button" data-idrem-detail-tab="biological_process" aria-pressed="false">Biological Process</button>',
      '<button class="module-tab-button" type="button" data-idrem-detail-tab="pathways" aria-pressed="false">Pathways</button>',
      "</div>",
      '<section class="idrem-path-section">',
      '<div class="artifact-head"><strong data-idrem-detail-title>Genes</strong></div>',
      '<div class="table-wrap" data-idrem-detail-table></div>',
      "</section>",
      "</section>",
      "</div>",
      "</div>",
    ].join("");
    panel._idremDetailTab = panel._idremDetailTab || "genes";
    panel.querySelectorAll("[data-idrem-path-chip]").forEach(function (button) {
      button.addEventListener("click", function () {
        updateIdremPathSelection(panel, payload, String(button.getAttribute("data-path-key") || ""));
      });
    });
    panel.querySelectorAll("[data-idrem-detail-tab]").forEach(function (button) {
      button.addEventListener("click", function () {
        setIdremDetailTab(panel, String(button.getAttribute("data-idrem-detail-tab") || "genes"));
      });
    });
    updateIdremPathSelection(panel, payload, payload.selected_path);
  }

  async function loadIdremTrackPanel(panel, trackUrl) {
    if (!panel || !trackUrl) {
      return;
    }
    panel.dataset.loading = "true";
    renderIdremTrackEmpty(panel, "Loading track iDREM summary...");
    try {
      const response = await fetch(trackUrl, { method: "GET" });
      const payload = await parseResponse(response);
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load the track iDREM summary.");
      }
      await renderIdremTrackPanel(panel, payload);
      panel.dataset.initialized = "true";
    } finally {
      panel.dataset.loading = "false";
    }
  }

  function initIdremTrackPanels(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-idrem-track-panel][data-track-url]").forEach(function (panel) {
      if (panel.dataset.initialized === "true" || panel.dataset.initialized === "loading") {
        return;
      }
      panel.dataset.initialized = "loading";
      loadIdremTrackPanel(panel, panel.dataset.trackUrl).catch(function (error) {
        panel.dataset.initialized = "error";
        renderIdremTrackEmpty(panel, error.message || String(error));
      });
    });
  }

  async function initTemporalGraphWidget(widget) {
    if (!widget || widget.dataset.initialized === "loading" || widget.dataset.initialized === "true") {
      return;
    }
    widget.dataset.initialized = "loading";
    try {
      await ensurePlotly();
      const response = await fetch(widget.dataset.graphUrl, { method: "GET" });
      const payload = await parseResponse(response);
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load temporal dynamics graph.");
      }
      widget._payload = payload;
      let plotDiv = widget.querySelector(".analysis-temporal-graph-plot");
      if (!plotDiv) {
        widget.innerHTML = '<div class="analysis-temporal-graph-plot"></div>';
        plotDiv = widget.querySelector(".analysis-temporal-graph-plot");
      }
      const shell = widget.closest(".overview-graph-shell");
      const statusTarget = shell ? shell.querySelector("[data-temporal-graph-status]") : null;
      const popupBase = widget.dataset.trackPopupBase || "";
      await window.Plotly.react(
        plotDiv,
        buildTemporalGraphTraces(payload),
        {
          height: 340,
          hovermode: "closest",
          dragmode: "pan",
          margin: { l: 68, r: 24, t: 18, b: 88 },
          legend: {
            orientation: "h",
            x: 0,
            xanchor: "left",
            y: -0.18,
            yanchor: "top",
            font: { size: 11 },
          },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          xaxis: {
            visible: false,
            zeroline: false,
            showgrid: false,
          },
          yaxis: {
            title: "Stage",
            zeroline: false,
            tickvals: payload.stage_tick_values || [],
            ticktext: payload.stage_tick_labels || [],
            gridcolor: "rgba(31, 42, 46, 0.12)",
          },
        },
        {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        },
      );
      resizePlotlyWithin(widget);
      if (typeof plotDiv.removeAllListeners === "function") {
        plotDiv.removeAllListeners("plotly_click");
      }
      plotDiv.on("plotly_click", function (eventData) {
        if (!eventData || !Array.isArray(eventData.points) || !eventData.points.length) {
          return;
        }
        const point = eventData.points[0];
        const nodeIndex = Number(point.customdata);
        if (!Number.isFinite(nodeIndex) || !Array.isArray(payload.nodes) || !payload.nodes[nodeIndex]) {
          return;
        }
        const node = payload.nodes[nodeIndex];
        if (!node.track_key) {
          return;
        }
        if (!popupBase) {
          return;
        }
        const url = new URL(popupBase, window.location.origin);
        url.searchParams.set("track_key", String(node.track_key));
        if (node.track_label) {
          url.searchParams.set("track_label", String(node.track_label));
        }
        const popup = window.open(
          `${url.pathname}${url.search}`,
          "unagi_idrem_track",
          "popup=yes,width=1600,height=1040,scrollbars=yes,resizable=yes",
        );
        if (!popup) {
          if (statusTarget) {
            statusTarget.textContent = "The popup window was blocked by the browser. Allow popups for this site and try again.";
          }
          return;
        }
        popup.focus();
      });
      widget.dataset.initialized = "true";
    } catch (error) {
      widget.dataset.initialized = "error";
      widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
    }
  }

  function initTemporalGraphWidgets(root) {
    const scope = root || document;
    scope.querySelectorAll(".analysis-temporal-graph-widget[data-graph-url]").forEach(function (widget) {
      initTemporalGraphWidget(widget);
    });
  }

  async function initStageCompositionWidget(widget) {
    if (!widget || widget.dataset.initialized === "loading" || widget.dataset.initialized === "true") {
      return;
    }
    widget.dataset.initialized = "loading";
    try {
      await ensurePlotly();
      const response = await fetch(widget.dataset.stageCompositionUrl, { method: "GET" });
      const payload = await parseResponse(response);
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load interactive stage composition.");
      }
      let plotDiv = widget.querySelector(".analysis-stage-composition-plot");
      if (!plotDiv) {
        widget.innerHTML = '<div class="analysis-stage-composition-plot"></div>';
        plotDiv = widget.querySelector(".analysis-stage-composition-plot");
      }
      await window.Plotly.react(
        plotDiv,
        payload.data || [],
        payload.layout || {},
        {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        },
      );
      resizePlotlyWithin(widget);
      widget.dataset.initialized = "true";
    } catch (error) {
      widget.dataset.initialized = "error";
      widget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
    }
  }

  function initStageCompositionWidgets(root) {
    const scope = root || document;
    scope.querySelectorAll(".analysis-stage-composition-widget[data-stage-composition-url]").forEach(function (widget) {
      initStageCompositionWidget(widget);
    });
  }

  async function initHierarchicalExplorerWidget(shell) {
    if (!shell || shell.dataset.hierarchicalBound === "true") {
      return;
    }
    shell.dataset.hierarchicalBound = "true";
    await ensurePlotly();

    const explorerRoot = shell.closest(".analysis-explorer");
    const form = explorerRoot ? explorerRoot.querySelector(".explorer-form") : null;
    const clusterSelect = form ? form.querySelector("select[name='hier_cluster']") : null;
    const dendrogramCard = shell.querySelector("[data-dendrogram-url]");
    const heatmapCard = shell.querySelector("[data-heatmap-url]");
    const dendrogramTarget = shell.querySelector("[data-hierarchical-dendrogram]");
    const heatmapTarget = shell.querySelector("[data-hierarchical-heatmap]");
    const selectedGeneChip = shell.querySelector("[data-hierarchical-selected-gene]");

    if (dendrogramCard && dendrogramTarget) {
      try {
        dendrogramTarget.innerHTML = "";
        const response = await fetch(dendrogramCard.dataset.dendrogramUrl, { method: "GET" });
        const payload = await parseResponse(response);
        if (!response.ok) {
          throw new Error(payload.detail || "Failed to load interactive dendrogram.");
        }
        await window.Plotly.react(
          dendrogramTarget,
          payload.data || [],
          payload.layout || {},
          { responsive: true, displaylogo: false },
        );
        if (typeof dendrogramTarget.removeAllListeners === "function") {
          dendrogramTarget.removeAllListeners("plotly_click");
        }
        dendrogramTarget.on("plotly_click", function (eventData) {
          if (!clusterSelect || !eventData || !Array.isArray(eventData.points) || !eventData.points.length) {
            return;
          }
          const cluster = String(eventData.points[0].customdata || "");
          if (!cluster) {
            return;
          }
          clusterSelect.value = cluster;
          if (typeof form.requestSubmit === "function") {
            form.requestSubmit();
          } else {
            form.submit();
          }
        });
      } catch (error) {
        dendrogramTarget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
      }
    }

    if (heatmapCard && heatmapTarget) {
      try {
        heatmapTarget.innerHTML = "";
        const response = await fetch(heatmapCard.dataset.heatmapUrl, { method: "GET" });
        const payload = await parseResponse(response);
        if (!response.ok) {
          throw new Error(payload.detail || "Failed to load interactive branch heatmap.");
        }
        await window.Plotly.react(
          heatmapTarget,
          payload.data || [],
          payload.layout || {},
          { responsive: true, displaylogo: false },
        );
        if (typeof heatmapTarget.removeAllListeners === "function") {
          heatmapTarget.removeAllListeners("plotly_click");
        }
        heatmapTarget.on("plotly_click", function (eventData) {
          if (!selectedGeneChip || !eventData || !Array.isArray(eventData.points) || !eventData.points.length) {
            return;
          }
          const gene = String(eventData.points[0].y || "");
          if (!gene) {
            return;
          }
          selectedGeneChip.hidden = false;
          selectedGeneChip.textContent = gene;
        });
      } catch (error) {
        heatmapTarget.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
      }
    }
  }

  function initHierarchicalExplorerWidgets(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-hierarchical-shell]").forEach(function (shell) {
      initHierarchicalExplorerWidget(shell);
    });
  }

  function setExplorerModeVisualState(form, selectedValue) {
    if (!form) {
      return;
    }
    form.querySelectorAll(".segment-option").forEach(function (option) {
      const input = option.querySelector("input[name='mode']");
      const active = !!input && input.value === selectedValue;
      option.classList.toggle("active", active);
    });
  }

  function initExplorerForms(root) {
    const scope = root || document;
    scope.querySelectorAll(".explorer-form").forEach(function (form) {
      if (form.dataset.explorerBound === "true") {
        return;
      }
      form.addEventListener("change", function (event) {
        const target = event.target;
        if (!target) {
          return;
        }
        const shouldSubmitImmediately = target.matches("input[name='mode'][data-explorer-mode], select[name='dyn_view_mode'], select[name='gene_scope'], select[name='pert_panel_mode']");
        if (!shouldSubmitImmediately) {
          return;
        }
        if (target.matches("input[name='mode'][data-explorer-mode]")) {
          setExplorerModeVisualState(form, target.value);
        }
        if (typeof form.requestSubmit === "function") {
          form.requestSubmit();
          return;
        }
        form.submit();
      });
      const checked = form.querySelector("input[name='mode']:checked");
      if (checked) {
        setExplorerModeVisualState(form, checked.value);
      }
      form.dataset.explorerBound = "true";
    });
  }

  function syncCustomPerturbationForm(form) {
    if (!form) {
      return;
    }
    const modeSelect = form.querySelector("[data-custom-perturbation-mode]");
    const scopeSelect = form.querySelector("[data-custom-perturbation-scope]");
    const mode = modeSelect ? modeSelect.value : "gene";
    const scopeKind = scopeSelect ? scopeSelect.value : "overall";

    const targetName = form.querySelector("[data-custom-perturbation-target-name]");
    const targetLabel = form.querySelector("[data-custom-perturbation-target-label]");
    const trackGroup = form.querySelector("[data-custom-perturbation-track-group]");
<<<<<<< HEAD
    const stageGroup = form.querySelector("[data-custom-perturbation-stage-group]");
=======
>>>>>>> 12a1cc490713a3a5cdb0cc508f0140b98087e2e0
    const geneGroup = form.querySelector("[data-custom-perturbation-gene-group]");
    const drugGroup = form.querySelector("[data-custom-perturbation-drug-group]");
    const signedLabel = form.querySelector("[data-custom-perturbation-signed-label]");
    const signedHelp = form.querySelector("[data-custom-perturbation-signed-help]");
    const signedTextarea = form.querySelector("[data-custom-perturbation-signed-gene-text]");

    if (targetName) {
      targetName.hidden = mode === "gene";
    }
    if (targetLabel) {
      if (mode === "pathway") {
        targetLabel.textContent = "Pathway Name";
      } else {
        targetLabel.textContent = "Drug Name";
      }
    }
    if (trackGroup) {
      trackGroup.hidden = scopeKind !== "track";
    }
<<<<<<< HEAD
    if (stageGroup) {
      stageGroup.hidden = scopeKind !== "stage";
    }
=======
>>>>>>> 12a1cc490713a3a5cdb0cc508f0140b98087e2e0
    if (geneGroup) {
      geneGroup.hidden = mode !== "pathway";
    }
    if (drugGroup) {
      drugGroup.hidden = !(mode === "gene" || mode === "drug");
    }
    if (signedLabel) {
      signedLabel.textContent = mode === "gene" ? "Gene Set Signature" : "Signed Gene Signature";
    }
    if (signedHelp) {
      signedHelp.textContent = mode === "gene"
        ? "Use GENE:+, GENE-, or plain genes. Plain genes default to +."
        : "Use GENE:+ or GENE-.";
    }
    if (signedTextarea) {
      signedTextarea.placeholder = mode === "gene" ? "GENE1:+, GENE2-, GENE3" : "GENE:+, GENE-";
    }
  }

  function initPlotlySpecCards(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-plotly-widget]").forEach(function (container) {
      const script = container.querySelector("script[data-plotly-spec]");
      if (!script) {
        return;
      }
      let spec = null;
      try {
        spec = JSON.parse(script.textContent || "{}");
      } catch (_error) {
        container.innerHTML = '<div class="notice notice-bad">Failed to load interactive plot.</div>';
        return;
      }
      ensurePlotly().then(function (Plotly) {
        container.innerHTML = "";
        const target = document.createElement("div");
        target.className = "analysis-plotly-stage";
        container.appendChild(target);
        Plotly.react(
          target,
          Array.isArray(spec.data) ? spec.data : [],
          spec.layout || {},
          spec.config || { responsive: true, displayModeBar: false },
        );
      }).catch(function (error) {
        container.innerHTML = `<div class="notice notice-bad">${escapeHtml(error.message || error)}</div>`;
      });
    });
  }

  function hideCustomPerturbationSuggestions(container) {
    if (!container) {
      return;
    }
    container.innerHTML = "";
    container.hidden = true;
  }

  async function fetchCustomPerturbationSuggestions(form, kind, query, limit) {
    const endpoint = form.getAttribute("data-custom-perturbation-suggest-url");
    if (!endpoint) {
      return [];
    }
    const url = new URL(endpoint, window.location.origin);
    url.searchParams.set("kind", kind);
    url.searchParams.set("q", query || "");
    url.searchParams.set("limit", String(limit || 12));
    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
      return [];
    }
    const payload = await response.json();
    return Array.isArray(payload.suggestions) ? payload.suggestions : [];
  }

  function renderCustomPerturbationSuggestions(container, suggestions, onSelect) {
    if (!container) {
      return;
    }
    if (!Array.isArray(suggestions) || !suggestions.length) {
      hideCustomPerturbationSuggestions(container);
      return;
    }
    container.innerHTML = suggestions.map(function (item) {
      const meta = Number.isFinite(item.gene_count) ? `<span class="muted">${escapeHtml(String(item.gene_count))} genes</span>` : "";
      return (
        `<button class="autocomplete-suggestion-button" type="button" data-suggestion-value="${escapeHtml(item.value)}">`
        + `<span>${escapeHtml(item.label || item.value)}</span>${meta}</button>`
      );
    }).join("");
    container.hidden = false;
    Array.from(container.querySelectorAll("[data-suggestion-value]")).forEach(function (button, index) {
      button.addEventListener("click", function () {
        const suggestion = suggestions[index];
        if (suggestion) {
          onSelect(suggestion);
        }
        hideCustomPerturbationSuggestions(container);
      });
    });
  }

  function getTextareaTokenState(textarea, signedMode) {
    const value = textarea.value || "";
    const cursor = Number.isFinite(textarea.selectionStart) ? textarea.selectionStart : value.length;
    let start = cursor;
    while (start > 0 && !/[\s,\n]/.test(value.charAt(start - 1))) {
      start -= 1;
    }
    let end = cursor;
    while (end < value.length && !/[\s,\n]/.test(value.charAt(end))) {
      end += 1;
    }
    const token = value.slice(start, end);
    const separatorIndex = signedMode ? token.indexOf(":") : -1;
    const query = separatorIndex >= 0 ? token.slice(0, separatorIndex).trim() : token.trim();
    const suffix = separatorIndex >= 0 ? token.slice(separatorIndex) : "";
    return { start, end, query, suffix };
  }

  function storeTextareaTokenState(textarea, signedMode) {
    if (!textarea) {
      return;
    }
    const state = getTextareaTokenState(textarea, signedMode);
    textarea.dataset.tokenStart = String(state.start);
    textarea.dataset.tokenEnd = String(state.end);
    textarea.dataset.tokenSuffix = state.suffix;
    textarea.dataset.tokenQuery = state.query;
  }

  function applyTextareaSuggestion(textarea, suggestionValue, signedMode) {
    const value = textarea.value || "";
    const start = Number.parseInt(textarea.dataset.tokenStart || "", 10);
    const end = Number.parseInt(textarea.dataset.tokenEnd || "", 10);
    const safeStart = Number.isFinite(start) ? start : value.length;
    const safeEnd = Number.isFinite(end) ? end : value.length;
    const suffix = signedMode ? String(textarea.dataset.tokenSuffix || "") : "";
    const replacement = `${suggestionValue}${suffix}`;
    textarea.value = `${value.slice(0, safeStart)}${replacement}${value.slice(safeEnd)}`;
    const nextCursor = safeStart + replacement.length;
    if (typeof textarea.focus === "function") {
      textarea.focus();
    }
    if (typeof textarea.setSelectionRange === "function") {
      textarea.setSelectionRange(nextCursor, nextCursor);
    }
    storeTextareaTokenState(textarea, signedMode);
  }

  function initCustomPerturbationForms(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-custom-perturbation-form]").forEach(function (form) {
      const modeSelect = form.querySelector("[data-custom-perturbation-mode]");
      const targetInput = form.querySelector("[data-custom-perturbation-target-input]");
      const targetSuggestions = form.querySelector("[data-custom-perturbation-target-suggestions]");
      const geneTextarea = form.querySelector("[data-custom-perturbation-gene-text]");
      const geneSuggestions = form.querySelector("[data-custom-perturbation-gene-suggestions]");
      const signedGeneTextarea = form.querySelector("[data-custom-perturbation-signed-gene-text]");
      const signedGeneSuggestions = form.querySelector("[data-custom-perturbation-signed-gene-suggestions]");

      if (form.dataset.customPerturbationBound !== "true") {
        form.addEventListener("change", function (event) {
          const target = event.target;
          if (!target) {
            return;
          }
          if (
            target.matches("[data-custom-perturbation-mode]")
            || target.matches("[data-custom-perturbation-scope]")
          ) {
            syncCustomPerturbationForm(form);
            hideCustomPerturbationSuggestions(targetSuggestions);
            hideCustomPerturbationSuggestions(geneSuggestions);
            hideCustomPerturbationSuggestions(signedGeneSuggestions);
          }
        });
        form.dataset.customPerturbationBound = "true";
      }

      if (targetInput && !targetInput.dataset.customPerturbationAutocompleteBound) {
        const scheduleTargetSuggestions = debounce(async function () {
          const mode = modeSelect ? modeSelect.value : "gene";
          if (mode === "gene") {
            hideCustomPerturbationSuggestions(targetSuggestions);
            return;
          }
          const suggestions = await fetchCustomPerturbationSuggestions(form, mode, targetInput.value || "", 12);
          renderCustomPerturbationSuggestions(targetSuggestions, suggestions, function (suggestion) {
            targetInput.value = suggestion.value || "";
            if (suggestion.fill_gene_text && geneTextarea) {
              geneTextarea.value = suggestion.fill_gene_text;
            }
            if (suggestion.fill_signed_gene_text && signedGeneTextarea) {
              signedGeneTextarea.value = suggestion.fill_signed_gene_text;
            }
          });
        }, 120);
        targetInput.addEventListener("input", scheduleTargetSuggestions);
        targetInput.addEventListener("focus", scheduleTargetSuggestions);
        targetInput.addEventListener("blur", function () {
          window.setTimeout(function () {
            hideCustomPerturbationSuggestions(targetSuggestions);
          }, 120);
        });
        targetInput.dataset.customPerturbationAutocompleteBound = "true";
      }

      if (geneTextarea && !geneTextarea.dataset.customPerturbationAutocompleteBound) {
        const scheduleGeneSuggestions = debounce(async function () {
          storeTextareaTokenState(geneTextarea, false);
          const query = geneTextarea.dataset.tokenQuery || "";
          if (!query) {
            hideCustomPerturbationSuggestions(geneSuggestions);
            return;
          }
          const suggestions = await fetchCustomPerturbationSuggestions(form, "gene", query, 12);
          renderCustomPerturbationSuggestions(geneSuggestions, suggestions, function (suggestion) {
            applyTextareaSuggestion(geneTextarea, suggestion.value || "", false);
          });
        }, 120);
        ["input", "click", "keyup", "focus"].forEach(function (eventName) {
          geneTextarea.addEventListener(eventName, scheduleGeneSuggestions);
        });
        geneTextarea.addEventListener("blur", function () {
          window.setTimeout(function () {
            hideCustomPerturbationSuggestions(geneSuggestions);
          }, 120);
        });
        geneTextarea.dataset.customPerturbationAutocompleteBound = "true";
      }

      if (signedGeneTextarea && !signedGeneTextarea.dataset.customPerturbationAutocompleteBound) {
        const scheduleSignedGeneSuggestions = debounce(async function () {
          storeTextareaTokenState(signedGeneTextarea, true);
          const query = signedGeneTextarea.dataset.tokenQuery || "";
          if (!query) {
            hideCustomPerturbationSuggestions(signedGeneSuggestions);
            return;
          }
          const suggestions = await fetchCustomPerturbationSuggestions(form, "gene", query, 12);
          renderCustomPerturbationSuggestions(signedGeneSuggestions, suggestions, function (suggestion) {
            applyTextareaSuggestion(signedGeneTextarea, suggestion.value || "", true);
          });
        }, 120);
        ["input", "click", "keyup", "focus"].forEach(function (eventName) {
          signedGeneTextarea.addEventListener(eventName, scheduleSignedGeneSuggestions);
        });
        signedGeneTextarea.addEventListener("blur", function () {
          window.setTimeout(function () {
            hideCustomPerturbationSuggestions(signedGeneSuggestions);
          }, 120);
        });
        signedGeneTextarea.dataset.customPerturbationAutocompleteBound = "true";
      }

      syncCustomPerturbationForm(form);
    });
  }

  function readDashboardState() {
    try {
      const raw = window.sessionStorage.getItem(DASHBOARD_STATE_KEY);
      if (!raw) {
        return { hasState: false, selectedJobIds: [], openTaskIds: [] };
      }
      const parsed = JSON.parse(raw);
      return {
        hasState: true,
        selectedJobIds: Array.isArray(parsed.selectedJobIds) ? parsed.selectedJobIds : [],
        openTaskIds: Array.isArray(parsed.openTaskIds) ? parsed.openTaskIds : [],
      };
    } catch (_error) {
      return { hasState: false, selectedJobIds: [], openTaskIds: [] };
    }
  }

  function saveDashboardState() {
    const form = document.getElementById("dashboard-bulk-delete-form");
    if (!form) {
      return;
    }
    const selectedJobIds = Array.from(form.querySelectorAll('input[name="job_id"]:checked')).map(function (input) {
      return input.value;
    });
    const openTaskIds = Array.from(form.querySelectorAll("details.analysis-level-card[open]"))
      .map(function (details) {
        return details.getAttribute("data-task-id");
      })
      .filter(Boolean);
    window.sessionStorage.setItem(
      DASHBOARD_STATE_KEY,
      JSON.stringify({ selectedJobIds: selectedJobIds, openTaskIds: openTaskIds }),
    );
  }

  function restoreDashboardState(root) {
    const state = readDashboardState();
    if (!state.hasState) {
      return;
    }
    const scope = root || document;
    scope.querySelectorAll('input[name="job_id"]').forEach(function (input) {
      input.checked = state.selectedJobIds.indexOf(input.value) !== -1;
    });
    scope.querySelectorAll("details.analysis-level-card[data-task-id]").forEach(function (details) {
      const taskId = details.getAttribute("data-task-id");
      details.open = state.openTaskIds.indexOf(taskId) !== -1;
    });
  }

  function initDashboardStatePersistence() {
    restoreDashboardState(document);

    document.addEventListener("change", function (event) {
      if (event.target && event.target.matches('#dashboard-bulk-delete-form input[name="job_id"]')) {
        saveDashboardState();
      }
    });

    document.addEventListener(
      "toggle",
      function (event) {
        if (event.target && event.target.matches("#dashboard-bulk-delete-form details.analysis-level-card")) {
          saveDashboardState();
        }
      },
      true,
    );

    document.body.addEventListener("htmx:beforeSwap", function (event) {
      if (event.target && event.target.id === "dashboard-panel") {
        saveDashboardState();
      }
    });

    document.body.addEventListener("htmx:afterSwap", function (event) {
      if (event.target && event.target.id === "dashboard-panel") {
        restoreDashboardState(event.target);
      }
      initializeDynamicScope(event.target || document);
    });
  }

  function setVisualizationButtonsBusy(jobId, busy, busyLabel) {
    document.querySelectorAll(`[data-generate-visualizations-job="${jobId}"]`).forEach(function (button) {
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }
      if (!button.dataset.idleLabel) {
        button.dataset.idleLabel = button.textContent.trim() || "Generate Visualizations";
      }
      button.disabled = busy;
      button.classList.toggle("disabled-button", busy);
      button.textContent = busy ? (busyLabel || "Generating Visualizations...") : button.dataset.idleLabel;
    });
  }

  function setShareBundleButtonsBusy(jobId, busy, busyLabel) {
    document.querySelectorAll(`[data-export-share-bundle-job="${jobId}"]`).forEach(function (button) {
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }
      if (!button.dataset.idleLabel) {
        button.dataset.idleLabel = button.textContent.trim() || "Export Share Bundle";
      }
      button.disabled = busy;
      button.classList.toggle("disabled-button", busy);
      button.textContent = busy ? (busyLabel || "Exporting Share Bundle...") : button.dataset.idleLabel;
    });
  }

  function renderVisualizationTaskStatus(payload) {
    const updatedAt = payload && payload.updated_at
      ? new Date(payload.updated_at).toLocaleTimeString()
      : null;
    if (!payload || !payload.status) {
      return "Preparing visualization task...";
    }
    if (payload.status === "queued") {
      return "Visualization generation queued. This can take a few minutes on large analyses.";
    }
    if (payload.status === "running") {
      return updatedAt
        ? `Generating visualizations... Last update ${updatedAt}.`
        : "Generating visualizations...";
    }
    if (payload.status === "succeeded") {
      const counts = payload.result && payload.result.counts
        ? ` ${payload.result.counts.successes} outputs, ${payload.result.counts.failures} failures.`
        : "";
      if (payload.reused_existing) {
        return `Existing visualizations found.${counts} Reloading gallery...`;
      }
      return `Visualizations generated.${counts} Reloading gallery...`;
    }
    if (payload.status === "failed") {
      return payload.error ? `Visualization generation failed: ${payload.error}` : "Visualization generation failed.";
    }
    return JSON.stringify(payload, null, 2);
  }

  function renderShareBundleTaskStatus(payload) {
    const updatedAt = payload && payload.updated_at
      ? new Date(payload.updated_at).toLocaleTimeString()
      : null;
    if (!payload || !payload.status) {
      return "Preparing share bundle export...";
    }
    if (payload.status === "queued") {
      return "Share bundle export queued.";
    }
    if (payload.status === "running") {
      return updatedAt
        ? `Exporting share bundle... Last update ${updatedAt}.`
        : "Exporting share bundle...";
    }
    if (payload.status === "succeeded") {
      const counts = payload.result && payload.result.counts
        ? ` ${payload.result.counts.entries} entries, ${payload.result.counts.tracks} tracks.`
        : "";
      if (payload.reused_existing) {
        return `Existing share bundle found.${counts} Reloading page...`;
      }
      return `Share bundle exported.${counts} Reloading page...`;
    }
    if (payload.status === "failed") {
      return payload.error ? `Share bundle export failed: ${payload.error}` : "Share bundle export failed.";
    }
    return JSON.stringify(payload, null, 2);
  }

  async function pollTask(taskId, resultId, jobId) {
    const result = ensureResult(resultId);
    if (!result) {
      return;
    }
    let response;
    let payload;
    try {
      response = await fetch(`/api/direct-tasks/${taskId}`);
      payload = await parseResponse(response);
    } catch (error) {
      result.textContent = `Visualization status check failed: ${error instanceof Error ? error.message : String(error)}`;
      setVisualizationButtonsBusy(jobId, false);
      return;
    }
    result.textContent = response.ok ? renderVisualizationTaskStatus(payload) : JSON.stringify(payload, null, 2);
    if (!response.ok) {
      setVisualizationButtonsBusy(jobId, false);
      return;
    }
    if (payload.status === "queued" || payload.status === "running") {
      window.setTimeout(function () {
        pollTask(taskId, resultId, jobId);
      }, 1500);
      return;
    }
    if (payload.status === "succeeded") {
      window.setTimeout(function () {
        window.location.reload();
      }, 350);
      return;
    }
    setVisualizationButtonsBusy(jobId, false);
  }

  async function generateVisualizations(jobId, resultId) {
    const result = ensureResult(resultId);
    if (!result) {
      return;
    }
    setVisualizationButtonsBusy(jobId, true, "Starting Visualizations...");
    result.textContent = "Starting visualization generation...";
    let response;
    let payload;
    try {
      response = await fetch(`/api/analysis/${jobId}/visualizations/generate`, { method: "POST" });
      payload = await parseResponse(response);
    } catch (error) {
      result.textContent = `Visualization launch failed: ${error instanceof Error ? error.message : String(error)}`;
      setVisualizationButtonsBusy(jobId, false);
      return;
    }
    result.textContent = response.ok ? renderVisualizationTaskStatus(payload) : JSON.stringify(payload, null, 2);
    if (!response.ok) {
      setVisualizationButtonsBusy(jobId, false);
      return;
    }
    if (payload.status === "succeeded") {
      window.setTimeout(function () {
        window.location.reload();
      }, 350);
      return;
    }
    if (payload.task_id) {
      window.setTimeout(function () {
        pollTask(payload.task_id, resultId, jobId);
      }, 1000);
      return;
    }
    setVisualizationButtonsBusy(jobId, false);
  }

  async function pollShareBundleTask(taskId, resultId, jobId) {
    const result = ensureResult(resultId);
    if (!result) {
      return;
    }
    let response;
    let payload;
    try {
      response = await fetch(`/api/direct-tasks/${taskId}`);
      payload = await parseResponse(response);
    } catch (error) {
      result.textContent = `Share bundle status check failed: ${error instanceof Error ? error.message : String(error)}`;
      setShareBundleButtonsBusy(jobId, false);
      return;
    }
    result.textContent = response.ok ? renderShareBundleTaskStatus(payload) : JSON.stringify(payload, null, 2);
    if (!response.ok) {
      setShareBundleButtonsBusy(jobId, false);
      return;
    }
    if (payload.status === "queued" || payload.status === "running") {
      window.setTimeout(function () {
        pollShareBundleTask(taskId, resultId, jobId);
      }, 1500);
      return;
    }
    if (payload.status === "succeeded") {
      window.setTimeout(function () {
        window.location.reload();
      }, 350);
      return;
    }
    setShareBundleButtonsBusy(jobId, false);
  }

  async function exportShareBundle(jobId, resultId) {
    const result = ensureResult(resultId);
    if (!result) {
      return;
    }
    setShareBundleButtonsBusy(jobId, true, "Starting Offline Report...");
    result.textContent = "Starting offline report export...";
    let response;
    let payload;
    try {
      response = await fetch(`/api/analysis/${jobId}/share-bundle/export?force=true`, { method: "POST" });
      payload = await parseResponse(response);
    } catch (error) {
      result.textContent = `Offline report launch failed: ${error instanceof Error ? error.message : String(error)}`;
      setShareBundleButtonsBusy(jobId, false);
      return;
    }
    result.textContent = response.ok ? renderShareBundleTaskStatus(payload) : JSON.stringify(payload, null, 2);
    if (!response.ok) {
      setShareBundleButtonsBusy(jobId, false);
      return;
    }
    if (payload.status === "succeeded") {
      window.setTimeout(function () {
        window.location.reload();
      }, 350);
      return;
    }
    if (payload.task_id) {
      window.setTimeout(function () {
        pollShareBundleTask(payload.task_id, resultId, jobId);
      }, 1000);
      return;
    }
    setShareBundleButtonsBusy(jobId, false);
  }

  window.unagiConsole = {
    submitForm,
    postAction,
    setPath,
    generateVisualizations,
    exportShareBundle,
    selectTabGroup,
    selectCustomPerturbation,
    openCustomPerturbation,
  };

  initDashboardStatePersistence();
  initializeDynamicScope(document);
})();
