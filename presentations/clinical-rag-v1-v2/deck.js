import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const colors = {
  ink: 0x061723,
  teal: 0x16bfd0,
  coral: 0xff665c,
  amber: 0xffbc42,
  lime: 0xb9e84c,
  violet: 0x7057ff,
  blue: 0x2878f0,
  paper: 0xf7fbff
};

function makeRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  return renderer;
}

function connect(scene, a, b, color = colors.teal) {
  const geometry = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.62
  });
  const line = new THREE.Line(geometry, material);
  scene.add(line);
  return line;
}

function createNode(label, color, position, radius = 0.42) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 36, 36),
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.48,
      metalness: 0.18
    })
  );
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 1.22, 32, 32),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.11
    })
  );
  group.add(glow, body);
  group.position.copy(position);
  group.userData.label = label;
  return group;
}

function createParticle(color, size = 0.08) {
  return new THREE.Mesh(
    new THREE.SphereGeometry(size, 16, 16),
    new THREE.MeshBasicMaterial({ color })
  );
}

function initNetworkScene(canvasId, variant) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const renderer = makeRenderer(canvas);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.1, 100);
  camera.position.set(0, 0.6, 6.7);

  const root = new THREE.Group();
  if (variant === "ops") {
    root.position.set(1.65, -0.04, 0);
    root.scale.setScalar(0.96);
  } else {
    root.position.set(1.42, -0.06, 0);
    root.scale.setScalar(0.92);
  }
  scene.add(root);

  const lightA = new THREE.DirectionalLight(0xffffff, 2.2);
  lightA.position.set(1.5, 3, 4);
  scene.add(lightA);
  scene.add(new THREE.AmbientLight(0x8edce8, 1.1));

  const nodes = variant === "ops"
    ? [
        createNode("Client", colors.teal, new THREE.Vector3(-2.2, 1.0, 0), 0.35),
        createNode("API", colors.amber, new THREE.Vector3(-0.85, 0.35, 0.1), 0.42),
        createNode("RAG", colors.violet, new THREE.Vector3(0.45, 0.95, -0.15), 0.48),
        createNode("Blob", colors.blue, new THREE.Vector3(1.45, -0.05, 0.05), 0.38),
        createNode("Foundry", colors.coral, new THREE.Vector3(2.35, 0.86, 0.12), 0.42),
        createNode("Audit", colors.lime, new THREE.Vector3(0.65, -0.82, -0.05), 0.36)
      ]
    : [
        createNode("Question", colors.teal, new THREE.Vector3(-1.95, 0.82, 0), 0.42),
        createNode("Embedding", colors.amber, new THREE.Vector3(-0.55, 0.1, 0.2), 0.34),
        createNode("FAISS", colors.lime, new THREE.Vector3(0.72, 0.72, -0.1), 0.44),
        createNode("LLM", colors.violet, new THREE.Vector3(1.76, -0.12, 0.15), 0.42),
        createNode("Answer", colors.coral, new THREE.Vector3(0.02, -0.95, 0), 0.36)
      ];

  nodes.forEach((node) => root.add(node));
  for (let index = 0; index < nodes.length - 1; index += 1) {
    connect(root, nodes[index], nodes[index + 1], index % 2 ? colors.lime : colors.teal);
  }
  connect(root, nodes[nodes.length - 1], nodes[0], colors.coral);

  const particles = nodes.map((node, index) => {
    const particle = createParticle(index % 2 ? colors.coral : colors.amber, variant === "ops" ? 0.065 : 0.08);
    root.add(particle);
    return { particle, index, phase: index / nodes.length };
  });

  let pointerX = 0;
  let pointerY = 0;
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointerX = ((event.clientX - rect.left) / rect.width - 0.5) * 0.7;
    pointerY = ((event.clientY - rect.top) / rect.height - 0.5) * 0.4;
  });

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function animate(time = 0) {
    resize();
    const t = time * 0.001;
    root.rotation.y += (pointerX + Math.sin(t * 0.28) * 0.08 - root.rotation.y) * 0.04;
    root.rotation.x += (-pointerY + Math.sin(t * 0.22) * 0.04 - root.rotation.x) * 0.04;

    nodes.forEach((node, index) => {
      node.position.y += Math.sin(t * 1.8 + index) * 0.0009;
      node.children[0].scale.setScalar(1 + Math.sin(t * 2 + index) * 0.05);
    });

    particles.forEach(({ particle, index, phase }) => {
      const from = nodes[index];
      const to = nodes[(index + 1) % nodes.length];
      const localT = (t * 0.32 + phase) % 1;
      particle.position.lerpVectors(from.position, to.position, localT);
      particle.position.y += Math.sin(localT * Math.PI) * 0.22;
    });

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  animate();
  window.addEventListener("resize", resize);
  Reveal.on("slidechanged", resize);
}

function colorForScore(score) {
  const min = 0.47;
  const max = 0.758;
  const t = Math.max(0, Math.min(1, (score - min) / (max - min)));
  const hue = 6 + t * 142;
  const light = 57 + t * 4;
  return `hsl(${hue}, 78%, ${light}%)`;
}

function initHeatmap() {
  const mount = document.getElementById("model-heatmap");
  if (!mount) return;

  const embeddings = ["BioBERT", "BioLORD", "MedQuAD", "biomedbert", "e5-base", "mini-lm", "mpnet-v2", "ms-marco", "multi-qa"];
  const llms = ["deepseek", "gemma", "llama", "phi3", "qwen", "tinyllama"];
  const values = [
    [0.661, 0.606, 0.668, 0.699, 0.663, 0.606],
    [0.707, 0.658, 0.579, 0.631, 0.673, 0.647],
    [0.649, 0.646, 0.581, 0.527, 0.626, 0.595],
    [0.620, 0.669, 0.722, 0.679, 0.623, 0.721],
    [0.674, 0.690, 0.598, 0.682, 0.648, 0.617],
    [0.622, 0.613, 0.647, 0.608, 0.719, 0.612],
    [0.651, 0.624, 0.633, 0.595, 0.470, 0.620],
    [0.606, 0.654, 0.651, 0.567, 0.591, 0.560],
    [0.624, 0.648, 0.572, 0.758, 0.578, 0.508]
  ];

  const tip = document.getElementById("heatmap-tip");
  const combo = document.getElementById("selected-combo");
  const score = document.getElementById("selected-score");
  const note = document.getElementById("selected-note");

  mount.appendChild(label(""));
  llms.forEach((name) => mount.appendChild(label(name)));

  let selectedCell;
  embeddings.forEach((embedding, row) => {
    mount.appendChild(label(embedding, "y"));
    llms.forEach((llm, col) => {
      const cell = document.createElement("button");
      const value = values[row][col];
      cell.type = "button";
      cell.className = "heatmap-cell";
      cell.textContent = value.toFixed(3);
      cell.style.background = colorForScore(value);
      cell.style.animationDelay = `${(row * llms.length + col) * 14}ms`;
      cell.dataset.combo = `${embedding} + ${llm}`;
      cell.dataset.score = value.toFixed(3);
      cell.addEventListener("pointerenter", () => {
        tip.textContent = `${cell.dataset.combo}: F1 ${cell.dataset.score}`;
      });
      cell.addEventListener("click", () => selectCell(cell));
      if (embedding === "multi-qa" && llm === "phi3") {
        selectedCell = cell;
      }
      mount.appendChild(cell);
    });
  });

  selectCell(selectedCell);

  function label(text, extraClass = "") {
    const item = document.createElement("div");
    item.className = `heatmap-label ${extraClass}`;
    item.textContent = text;
    return item;
  }

  function selectCell(cell) {
    if (!cell) return;
    mount.querySelectorAll(".selected").forEach((item) => item.classList.remove("selected"));
    cell.classList.add("selected");
    combo.textContent = cell.dataset.combo;
    score.textContent = cell.dataset.score;
    note.textContent = Number(cell.dataset.score) >= 0.72
      ? "Top-tier result in the V1 evaluation grid."
      : "Useful comparison point for quality-speed trade-offs.";
    tip.textContent = `${cell.dataset.combo}: F1 ${cell.dataset.score}`;
  }
}

function initTypewriter() {
  const target = document.getElementById("stream-text");
  if (!target) return;

  const text = "Lab results for admission 25282710 include glucose 338.0, hematocrit 33.8, and hemoglobin 11.0. Each value is grounded in retrieved MIMIC-IV context.";
  let index = 0;

  function step() {
    target.textContent = text.slice(0, index);
    index = index >= text.length ? 0 : index + 1;
    window.setTimeout(step, index === 0 ? 900 : 28);
  }

  step();
}

initNetworkScene("rag-3d", "rag");
initNetworkScene("ops-3d", "ops");
initHeatmap();
initTypewriter();
