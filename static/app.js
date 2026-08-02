/* Chat client: SSE transport, incremental markdown, Plotly artifacts. */

(() => {
  "use strict";

  const transcript = document.getElementById("transcript");
  const intro = document.getElementById("intro");
  const form = document.getElementById("composer");
  const input = document.getElementById("question");
  const sendButton = document.getElementById("send");
  const stopButton = document.getElementById("stop");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  // Trimmed before sending: the server caps history too, but there is no point
  // shipping a transcript the model will not be given.
  const HISTORY_LIMIT = 10;

  const history = [];
  let controller = null;

  // ---------------------------------------------------------------- charts

  const PALETTE = [
    "#ffc53d", "#58a6ff", "#3fb950", "#f778ba",
    "#a371f7", "#f0883e", "#39c5cf", "#db6d28",
  ];

  const PLOT_FONT = {
    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif',
    size: 11.5,
    color: "#97a3b4",
  };

  const PLOT_CONFIG = {
    displayModeBar: false,
    responsive: true,
  };

  function baseLayout(spec) {
    return {
      // No in-plot title: the artifact header already carries it, and repeating
      // it just eats vertical space.
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: PLOT_FONT,
      margin: { l: 62, r: 62, t: 34, b: 62 },
      hovermode: "closest",
      hoverlabel: {
        bgcolor: "#1c222b",
        bordercolor: "#333c4a",
        font: { ...PLOT_FONT, color: "#e6edf3" },
      },
      // Above the plot rather than below it. Long rotated category labels grow
      // downwards without limit, and a legend underneath them ends up colliding
      // with both the ticks and the axis title.
      legend: {
        orientation: "h",
        y: 1.02,
        yanchor: "bottom",
        x: 0,
        font: { ...PLOT_FONT, size: 11 },
      },
      xaxis: axis(spec.x_title),
      yaxis: axis(spec.y_title),
    };
  }

  function axis(title) {
    return {
      title: { text: title || "", font: { ...PLOT_FONT, size: 11 } },
      gridcolor: "#262d38",
      zerolinecolor: "#333c4a",
      linecolor: "#333c4a",
      tickfont: { ...PLOT_FONT, size: 10.5 },
      automargin: true,
    };
  }

  /* Bar and line series sharing a category axis, with an optional second
     y-axis. Covers the funnel, per-zone supply/demand and conversion charts. */
  function renderSeries(node, spec) {
    const usesSecondAxis = (spec.series || []).some((s) => s.axis === "y2");

    const traces = (spec.series || []).map((series, index) => {
      const colour = PALETTE[index % PALETTE.length];
      const onSecondAxis = series.axis === "y2";
      const shared = {
        name: series.name,
        x: spec.x,
        y: series.values,
        yaxis: onSecondAxis ? "y2" : "y",
      };

      if (series.type === "line") {
        return {
          ...shared,
          type: "scatter",
          mode: "lines+markers",
          line: { color: colour, width: 2.4, shape: "spline", smoothing: 0.4 },
          marker: { color: colour, size: 6 },
        };
      }

      return {
        ...shared,
        type: "bar",
        marker: { color: colour, opacity: onSecondAxis ? 0.42 : 0.85 },
      };
    });

    const layout = baseLayout(spec);
    layout.barmode = "group";

    if (usesSecondAxis) {
      layout.yaxis2 = {
        ...axis(spec.y2_title),
        overlaying: "y",
        side: "right",
        gridcolor: "rgba(0,0,0,0)",
      };
    }

    Plotly.newPlot(node, traces, layout, PLOT_CONFIG);
  }

  function renderHeatmap(node, spec) {
    const trace = {
      type: "heatmap",
      x: spec.x,
      y: spec.y,
      z: spec.z,
      colorscale: [
        [0, "#1c3a5e"], [0.5, "#2d4f6b"], [0.75, "#c98f2a"], [1, "#ffc53d"],
      ],
      hoverongaps: false,
      colorbar: {
        title: { text: spec.color_title || "", font: { ...PLOT_FONT, size: 10.5 } },
        thickness: 11,
        len: 0.82,
        outlinewidth: 0,
        tickfont: { ...PLOT_FONT, size: 10 },
      },
    };

    const layout = baseLayout(spec);
    layout.margin.r = 84;
    layout.margin.t = 14;
    layout.margin.b = 52;
    Plotly.newPlot(node, [trace], layout, PLOT_CONFIG);
  }

  /* One trace per cluster so the legend carries the cluster descriptions,
     which is where the interpretation actually lives. */
  function renderClusterScatter(node, spec) {
    const groups = new Map();

    (spec.clusters || []).forEach((cluster, i) => {
      if (!groups.has(cluster)) groups.set(cluster, []);
      groups.get(cluster).push(i);
    });

    const many = (spec.clusters || []).length > 400;

    const traces = [...groups.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([cluster, indices], order) => {
        const description = (spec.cluster_descriptions || {})[String(cluster)];
        const labels = spec.labels
          ? indices.map((i) => spec.labels[i])
          : null;

        return {
          type: "scattergl",
          mode: many ? "markers" : "markers+text",
          name: description ? `${cluster}: ${description}` : `cluster ${cluster}`,
          x: indices.map((i) => spec.x[i]),
          y: indices.map((i) => spec.y[i]),
          text: labels,
          textposition: "top center",
          textfont: { ...PLOT_FONT, size: 9.5 },
          hovertemplate: labels
            ? "%{text}<extra></extra>"
            : `cluster ${cluster}<extra></extra>`,
          marker: {
            color: PALETTE[order % PALETTE.length],
            size: many ? 4.5 : 11,
            opacity: many ? 0.55 : 0.95,
            line: many ? undefined : { color: "#0e1116", width: 1 },
          },
        };
      });

    const layout = baseLayout(spec);
    // Cluster descriptions are sentences, so the legend needs room to wrap.
    layout.margin.t = groups.size > 4 ? 92 : 58;
    // Extra height spreads the points out; with named markers the labels
    // collide badly in dense regions at the default size.
    node.style.height = "440px";
    Plotly.newPlot(node, traces, layout, PLOT_CONFIG);
  }

  const RENDERERS = {
    funnel_by_dimension: renderSeries,
    supply_demand_by_zone: renderSeries,
    conversion_by_bucket: renderSeries,
    heatmap: renderHeatmap,
    cluster_scatter: renderClusterScatter,
  };

  // ------------------------------------------------------------- artifacts

  function artifactShell(artifact) {
    const box = document.createElement("figure");
    box.className = "artifact";

    const head = document.createElement("figcaption");
    head.className = "artifact-head";
    head.innerHTML =
      `<span class="artifact-title"></span><span class="artifact-source"></span>`;
    // The chart spec carries the more specific title, including any filter that
    // was applied, so prefer it over the generic one from the tool.
    head.querySelector(".artifact-title").textContent =
      (artifact.spec && artifact.spec.title) || artifact.title || "";
    head.querySelector(".artifact-source").textContent = artifact.source || "";
    box.appendChild(head);

    const body = document.createElement("div");
    body.className = "artifact-body";
    box.appendChild(body);

    if (artifact.note) {
      const note = document.createElement("p");
      note.className = "artifact-note";
      note.textContent = artifact.note;
      box.appendChild(note);
    }

    return { box, body };
  }

  function looksNumeric(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function formatCell(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (looksNumeric(value)) {
      if (Number.isInteger(value)) return value.toLocaleString();
      return Math.abs(value) < 0.001 && value !== 0
        ? value.toExponential(2)
        : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return String(value);
  }

  function renderTable(host, artifact) {
    const scroll = document.createElement("div");
    scroll.className = "table-scroll";

    const table = document.createElement("table");
    table.className = "data-table";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    (artifact.columns || []).forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column.replace(/_/g, " ");
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    (artifact.rows || []).forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((cell) => {
        const td = document.createElement("td");
        if (looksNumeric(cell)) td.className = "num";
        td.textContent = formatCell(cell);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    scroll.appendChild(table);
    host.appendChild(scroll);
  }

  function renderArtifact(container, artifact) {
    const { box, body } = artifactShell(artifact);

    if (artifact.kind === "table") {
      renderTable(body, artifact);
    } else if (artifact.kind === "chart") {
      const spec = artifact.spec || {};
      const draw = RENDERERS[spec.kind];
      if (!draw) return;

      const plot = document.createElement("div");
      plot.className = "plot";
      body.appendChild(plot);
      container.appendChild(box);

      // Plotly needs the node in the document to size itself.
      try {
        draw(plot, spec);
      } catch (error) {
        console.error("chart failed", spec.kind, error);
        box.remove();
      }
      return;
    } else {
      return;
    }

    container.appendChild(box);
  }

  // --------------------------------------------------------------- markdown

  marked.setOptions({ breaks: true, gfm: true });

  function toHtml(markdown) {
    return DOMPurify.sanitize(marked.parse(markdown || ""), {
      ALLOWED_TAGS: [
        "p", "br", "strong", "em", "del", "code", "pre", "blockquote",
        "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "thead", "tbody", "tr", "th", "td", "hr", "a",
      ],
      ALLOWED_ATTR: ["href", "title"],
    });
  }

  // ---------------------------------------------------------------- layout

  function atBottom() {
    const slack = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
    return slack < 130;
  }

  function scrollDown(force) {
    if (force || atBottom()) {
      transcript.scrollTop = transcript.scrollHeight;
    }
  }

  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    transcript.appendChild(wrap);
    scrollDown(true);
  }

  function addAgentMessage() {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-agent";

    const role = document.createElement("div");
    role.className = "role";
    role.textContent = "Analyst";
    wrap.appendChild(role);

    const trail = document.createElement("div");
    trail.className = "trail";
    wrap.appendChild(trail);

    const activity = document.createElement("div");
    activity.className = "thinking";
    activity.innerHTML =
      `<span class="pips"><i></i><i></i><i></i></span><span class="what">думає</span>`;
    wrap.appendChild(activity);

    const artifacts = document.createElement("div");
    wrap.appendChild(artifacts);

    const answer = document.createElement("div");
    answer.className = "answer";
    wrap.appendChild(answer);

    // Errors live outside .answer because the token handler rewrites that
    // element wholesale, which would otherwise erase the message.
    const errors = document.createElement("div");
    wrap.appendChild(errors);

    transcript.appendChild(wrap);
    scrollDown(true);

    return { wrap, trail, activity, artifacts, answer, errors };
  }

  function noteTool(view, name) {
    const item = document.createElement("span");
    item.className = "trail-item";
    item.innerHTML = `<span class="tick">✓</span><span></span>`;
    item.querySelector("span:last-child").textContent = name;
    view.trail.appendChild(item);
    scrollDown();
  }

  function setActivity(view, label) {
    const what = view.activity.querySelector(".what");
    if (what) what.textContent = label;
  }

  function showError(view, message) {
    const box = document.createElement("div");
    box.className = "error-box";
    box.textContent = message;
    view.errors.appendChild(box);
    scrollDown();
  }

  // ------------------------------------------------------------------ send

  function setBusy(busy) {
    input.disabled = busy;
    sendButton.disabled = busy;
    sendButton.hidden = busy;
    stopButton.hidden = !busy;
    if (!busy) input.focus();
  }

  async function ask(question) {
    if (intro) intro.remove();

    addUserMessage(question);
    const view = addAgentMessage();

    setBusy(true);
    controller = new AbortController();

    let answerText = "";
    let finished = false;

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: history.slice(-HISTORY_LIMIT),
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const detail = await response.text().catch(() => "");
        throw new Error(
          `Сервер відповів ${response.status}. ${detail.slice(0, 300)}`
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          const line = frame.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue; // keepalive comment

          let event;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }

          if (event.type === "token") {
            answerText += event.text;
            view.answer.innerHTML = toHtml(answerText) + '<span class="cursor"></span>';
            scrollDown();
          } else if (event.type === "tool") {
            noteTool(view, event.tool);
            setActivity(view, "running analysis");
          } else if (event.type === "status") {
            setActivity(view, event.message || "working");
          } else if (event.type === "artifact") {
            renderArtifact(view.artifacts, event.artifact);
            scrollDown();
          } else if (event.type === "error") {
            showError(view, event.message);
          } else if (event.type === "done") {
            finished = true;
            if (event.answer) answerText = event.answer;
          }
        }
      }
    } catch (error) {
      if (error.name === "AbortError") {
        setActivity(view, "stopped");
      } else {
        console.error(error);
        showError(view, error.message || "Запит не вдався.");
      }
    } finally {
      controller = null;
      view.activity.remove();
      // Re-render without the caret. Skipped when nothing was produced, so an
      // error-only turn does not end up as a blank assistant message.
      if (answerText) view.answer.innerHTML = toHtml(answerText);

      if (finished && answerText) {
        history.push({ role: "user", content: question });
        history.push({ role: "assistant", content: answerText });
      }

      setBusy(false);
      scrollDown();
    }
  }

  // ---------------------------------------------------------------- wiring

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || controller) return;
    input.value = "";
    resize();
    ask(question);
  });

  stopButton.addEventListener("click", () => {
    if (controller) controller.abort();
  });

  // Enter sends, Shift+Enter makes a new line.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  function resize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 190)}px`;
  }

  input.addEventListener("input", resize);

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (controller) return;
      ask(chip.dataset.q);
    });
  });

  // ---------------------------------------------------------------- health

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      const health = await response.json();

      if (!health.database) {
        statusDot.className = "status-dot err";
        statusText.textContent = "база недоступна";
        return;
      }

      const rides = (health.calculated_rides || 0).toLocaleString();

      if (!health.llm_configured) {
        statusDot.className = "status-dot warn";
        statusText.textContent = `${rides} поїздок · чат вимкнено`;
        input.placeholder = "Додайте OPENROUTER_API_KEY, щоб увімкнути чат";
        input.disabled = true;
        sendButton.disabled = true;
        return;
      }

      statusDot.className = "status-dot ok";
      statusText.textContent = `${rides} поїздок`;
    } catch {
      statusDot.className = "status-dot err";
      statusText.textContent = "офлайн";
    }
  }

  checkHealth();
  input.focus();
})();
