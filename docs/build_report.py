"""Build the project report PDF: docs/slingpuck_report.pdf.

    python docs/build_report.py

The function tables are read from the code with `ast` (names, signatures,
docstrings), so they always match the code. Hand-written text is in
docs/report_content.py. The HTML is printed to PDF with headless Chrome.
"""
import ast
import html
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))

from report_content import DESC, FILES  # noqa: E402

from src.slingpuck.config import load_config  # noqa: E402

SRC = ROOT / "src" / "slingpuck"
OUT_HTML = ROOT / "docs" / "report" / "slingpuck_report.html"
OUT_PDF = ROOT / "docs" / "slingpuck_report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
E = html.escape


# ------------------------------------------------------------------ code facts
def first_sentence(doc: str) -> str:
    text = " ".join(doc.strip().split("\n\n")[0].split())
    for end in (". ", ".\n"):
        if end in text:
            return text.split(end)[0] + "."
    return text


def signature(node) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class({bases})" if bases else "class"
    args = node.args
    parts = []
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for a, d in zip(args.args, defaults):
        if a.arg in ("self", "cls"):
            continue
        parts.append(a.arg + ("=" + ast.unparse(d) if d is not None else ""))
    if args.vararg:
        parts.append("*" + args.vararg.arg)
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(a.arg + ("=" + ast.unparse(d) if d is not None else ""))
    if args.kwarg:
        parts.append("**" + args.kwarg.arg)
    sig = "(" + ", ".join(parts) + ")"
    if len(sig) > 70:
        sig = sig[:67] + "...)"
    deco = [ast.unparse(d) for d in node.decorator_list]
    prefix = "property " if "property" in deco else ("classmethod " if "classmethod" in deco else "")
    return prefix + sig


def code_items(path: Path):
    """Top-level classes and functions, and class methods: (kind, qualname, signature, description)."""
    rel = path.relative_to(SRC).as_posix()
    tree = ast.parse(path.read_text())
    items = []

    def describe(qual, node):
        key = f"{rel}::{qual}"
        if key in DESC:
            return DESC[key]
        doc = ast.get_docstring(node)
        return first_sentence(doc) if doc else ""

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            items.append(("class", node.name, signature(node), describe(node.name, node)))
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    q = f"{node.name}.{sub.name}"
                    items.append(("method", q, signature(sub), describe(q, sub)))
        elif isinstance(node, ast.FunctionDef):
            items.append(("function", node.name, signature(node), describe(node.name, node)))
    return items


def module_doc(path: Path) -> str:
    doc = ast.get_docstring(ast.parse(path.read_text()))
    return doc or ""


def line_count(path: Path) -> int:
    return len(path.read_text().splitlines())


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def test_count() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


# ------------------------------------------------------------------- sections
ORDER = [
    ("Core", ["config.py", "kinematics.py"]),
    ("Physics", ["physics/backend.py", "physics/puck_dynamics.py", "physics/servo_model.py",
                 "physics/band_model.py", "physics/arm_deflection.py", "physics/mujoco_goalkeeper.py"]),
    ("Sensing", ["sensing/camera_model.py", "sensing/kalman_tracker.py"]),
    ("Environments", ["envs/goalkeeper_env.py", "envs/mujoco_goalkeeper_env.py", "envs/sling_env.py",
                      "envs/match_env.py", "envs/wrappers.py"]),
    ("Opponent and primitives", ["opponent/scripted_opponent.py", "primitives/block.py", "primitives/sling.py"]),
    ("Policies", ["policies/asymmetric.py", "policies/scripted.py", "policies/sling_scripted.py",
                  "policies/match_scripted.py"]),
    ("Training", ["train/common.py", "train/train_goalkeeper.py", "train/train_sling.py"]),
    ("Evaluation and viewers", ["eval/stats.py", "eval/save_rate_vs_speed.py", "eval/hole_rate.py",
                                "eval/match_eval.py", "eval/visualize_goalkeeper.py", "eval/view_mujoco.py"]),
    ("System identification", ["sysid/schema.py", "sysid/synthetic.py", "sysid/fit.py", "sysid/run_sysid.py"]),
    ("Deployment (old code, M5)", ["deploy/export_onnx.py", "deploy/latency_benchmark.py",
                                   "deploy/lerobot_interface.py"]),
]


def code_reference_html() -> str:
    covered = {f for _, files in ORDER for f in files}
    actual = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if p.name != "__init__.py"}
    missing = actual - covered
    if missing:
        raise SystemExit(f"report misses files: {sorted(missing)}")
    parts = []
    for group, files in ORDER:
        parts.append(f'<h2 class="group">{E(group)}</h2>')
        for rel in files:
            path = SRC / rel
            info = FILES[rel]
            parts.append(f'<div class="file"><h3><code>src/slingpuck/{E(rel)}</code></h3>')
            parts.append(f'<p class="ftitle">{E(info["title"])} &middot; {line_count(path)} lines</p>')
            for para in info["purpose"]:
                parts.append(f"<p>{E(para)}</p>")
            if info.get("how"):
                parts.append('<p class="label">How it works</p><ul>' +
                             "".join(f"<li>{E(x)}</li>" for x in info["how"]) + "</ul>")
            if info.get("notes"):
                parts.append('<p class="label">Important</p><ul class="notes">' +
                             "".join(f"<li>{E(x)}</li>" for x in info["notes"]) + "</ul>")
            items = code_items(path)
            if items:
                rows = "".join(
                    f'<tr class="{kind}"><td class="kind">{kind}</td><td><code>{E(q)}</code></td>'
                    f'<td><code class="sig">{E(sig)}</code></td><td>{E(desc)}</td></tr>'
                    for kind, q, sig, desc in items)
                parts.append('<table class="api"><thead><tr><th>Kind</th><th>Name</th><th>Arguments</th>'
                             f"<th>What it does</th></tr></thead><tbody>{rows}</tbody></table>")
            parts.append("</div>")
    return "\n".join(parts)


def placeholders_html(cfg) -> str:
    items = cfg["meta"]["placeholders"]
    cells = "".join(f"<li><code>{E(p)}</code></li>" for p in items)
    return f'<p>{len(items)} values are still placeholders in v0:</p><ul class="cols">{cells}</ul>'


def fig(name, caption, width="100%"):
    return (f'<figure><img src="../figures/{name}" style="width:{width}"/>'
            f"<figcaption>{E(caption)}</figcaption></figure>")


ARCH_SVG = """
<svg viewBox="-40 0 940 410" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial" font-size="12">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>
  <style>.b{fill:#f3f1ea;stroke:#b9a27c;stroke-width:1.2}.h{fill:#e6eefb;stroke:#2a78d6;stroke-width:1.2}
  .o{fill:#fdeee6;stroke:#eb6834;stroke-width:1.2}.g{fill:#e3f5ee;stroke:#1baf7a;stroke-width:1.2}
  .t{fill:#0b0b0b}.s{fill:#52514e;font-size:10.5px}.l{stroke:#52514e;stroke-width:1.2;fill:none;marker-end:url(#a)}</style>
  <rect class="b" x="10" y="20" width="180" height="80" rx="6"/>
  <text class="t" x="20" y="42" font-weight="bold">configs/</text>
  <text class="s" x="20" y="60">servo.yaml, env.yaml</text>
  <text class="s" x="20" y="74">physics_params_vN.yaml</text>
  <text class="s" x="20" y="88">train_*.yaml</text>
  <rect class="b" x="230" y="20" width="200" height="80" rx="6"/>
  <text class="t" x="240" y="42" font-weight="bold">config.py</text>
  <text class="s" x="240" y="60">merge, versions, placeholders,</text>
  <text class="s" x="240" y="74">strict mode, randomization,</text>
  <text class="s" x="240" y="88">run records</text>
  <rect class="h" x="10" y="140" width="200" height="110" rx="6"/>
  <text class="t" x="20" y="162" font-weight="bold">physics/ + sensing/</text>
  <text class="s" x="20" y="180">Fast2DPuckSim, MuJoCo backend</text>
  <text class="s" x="20" y="194">servo, band, arm deflection</text>
  <text class="s" x="20" y="208">camera, Kalman tracker</text>
  <text class="s" x="20" y="222">kinematics.py (action -> joint)</text>
  <rect class="h" x="250" y="140" width="200" height="110" rx="6"/>
  <text class="t" x="260" y="162" font-weight="bold">envs/ (Gymnasium)</text>
  <text class="s" x="260" y="180">Phase 1 GoalkeeperEnv (+MuJoCo)</text>
  <text class="s" x="260" y="194">Phase 2 SlingEnv</text>
  <text class="s" x="260" y="208">Phase 3 MatchEnv</text>
  <text class="s" x="260" y="222">PrivilegedObsWrapper</text>
  <rect class="o" x="490" y="140" width="190" height="110" rx="6"/>
  <text class="t" x="500" y="162" font-weight="bold">policies/ + train/</text>
  <text class="s" x="500" y="180">asymmetric actor-critic (PPO)</text>
  <text class="s" x="500" y="194">scripted baselines</text>
  <text class="s" x="500" y="208">train_goalkeeper, train_sling</text>
  <text class="s" x="500" y="222">-> logs/, tensorboard_logs/</text>
  <rect class="o" x="720" y="140" width="170" height="110" rx="6"/>
  <text class="t" x="730" y="162" font-weight="bold">eval/</text>
  <text class="s" x="730" y="180">save rate vs speed (P1)</text>
  <text class="s" x="730" y="194">hole rate + parity (P2)</text>
  <text class="s" x="730" y="208">match eval (P3), viewers</text>
  <text class="s" x="730" y="222">-> results/, docs/figures/</text>
  <rect class="g" x="250" y="300" width="200" height="90" rx="6"/>
  <text class="t" x="260" y="322" font-weight="bold">primitives/ + opponent/</text>
  <text class="s" x="260" y="340">block table (from Phase 1 env)</text>
  <text class="s" x="260" y="354">sling timing + P2 success rate</text>
  <text class="s" x="260" y="368">scripted human opponent</text>
  <rect class="g" x="10" y="300" width="200" height="90" rx="6"/>
  <text class="t" x="20" y="322" font-weight="bold">sysid/ (sim2real2sim)</text>
  <text class="s" x="20" y="340">real logs -> fits -></text>
  <text class="s" x="20" y="354">physics_params_v(N+1).yaml</text>
  <text class="s" x="20" y="368">parity check (eval/hole_rate)</text>
  <rect class="b" x="490" y="300" width="190" height="90" rx="6"/>
  <text class="t" x="500" y="322" font-weight="bold">deploy/ (M5)</text>
  <text class="s" x="500" y="340">ONNX export, latency test</text>
  <text class="s" x="500" y="354">LeRobot interface stub</text>
  <text class="s" x="500" y="368">(uses kinematics.py)</text>
  <path class="l" d="M190,60 L228,60"/>
  <path class="l" d="M330,100 L330,138"/>
  <path class="l" d="M240,100 L170,138"/>
  <path class="l" d="M210,195 L248,195"/>
  <path class="l" d="M450,195 L488,195"/>
  <path class="l" d="M680,195 L718,195"/>
  <path class="l" d="M350,250 L350,298"/>
  <path class="l" d="M370,298 L370,252"/>
  <path class="l" d="M110,298 L110,252"/>
  <path class="l" d="M585,250 L585,298"/>
  <path class="l" d="M10,345 L-22,345 L-22,60 L8,60"/>
  <text class="s" x="-36" y="404">sysid writes a new params version (loop back to configs/)</text>
</svg>
"""


def build_html() -> str:
    cfg = load_config("v0")
    commit = git("rev-parse", "--short", "HEAD")
    n_commits = git("rev-list", "--count", "HEAD")
    log_rows = "".join(
        f"<tr><td><code>{E(h[:7])}</code></td><td>{E(s)}</td></tr>"
        for h, s in (l.split(" ", 1) for l in git("log", "--reverse", "--format=%H %s").splitlines()))
    n_tests = test_count()
    n_src = sum(1 for p in SRC.rglob("*.py") if p.name != "__init__.py")
    n_lines = sum(line_count(p) for p in SRC.rglob("*.py"))
    tests_rows = "".join(f"<tr><td><code>{E(p.name)}</code></td><td>{line_count(p)}</td></tr>"
                         for p in sorted((ROOT / "tests").glob("test_*.py")))

    css = """
    @page { size: A4; margin: 16mm 15mm 18mm 15mm; }
    body { font-family: -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10.2pt;
           color: #1b1b1b; line-height: 1.42; }
    h1 { font-size: 22pt; margin: 0 0 6px; } h2 { font-size: 15pt; margin: 22px 0 8px; border-bottom: 2px solid #e1e0d9; padding-bottom: 3px; }
    h2.group { font-size: 13pt; color: #2a78d6; border-bottom: 1px solid #c9d9f2; page-break-after: avoid; }
    h3 { font-size: 11pt; margin: 14px 0 2px; page-break-after: avoid; }
    .section { page-break-before: always; }
    code { font-family: Menlo, 'SF Mono', Consolas, monospace; font-size: 8.6pt; background: #f4f3ef; padding: 0 2px; border-radius: 2px; }
    pre { font-family: Menlo, monospace; font-size: 8.4pt; background: #f4f3ef; padding: 8px 10px; border-radius: 4px; white-space: pre-wrap; page-break-inside: avoid; }
    table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9pt; page-break-inside: auto; }
    th, td { border-bottom: 1px solid #e1e0d9; padding: 3px 5px; text-align: left; vertical-align: top; }
    th { background: #f4f3ef; } tr { page-break-inside: avoid; }
    table.api td.kind { color: #898781; width: 48px; font-size: 8pt; }
    table.api tr.method td:nth-child(2) code { color: #3a3a3a; }
    table.api code.sig { background: none; color: #52514e; font-size: 7.8pt; }
    .file { margin-bottom: 8px; }
    .ftitle { color: #52514e; margin: 0 0 4px; font-style: italic; }
    .label { font-weight: bold; margin: 6px 0 2px; }
    ul { margin: 2px 0 6px 18px; padding: 0; } li { margin: 1px 0; }
    ul.notes li { color: #7a3a12; }
    ul.cols { columns: 2; font-size: 8.6pt; }
    figure { margin: 8px 0 12px; page-break-inside: avoid; text-align: center; }
    figure img { max-width: 100%; border: 1px solid #e1e0d9; }
    figcaption { font-size: 8.8pt; color: #52514e; margin-top: 3px; }
    .cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; }
    .cover .sub { font-size: 13pt; color: #52514e; } .cover .meta { margin-top: 26px; color: #52514e; }
    .box { background: #f6f8fc; border-left: 4px solid #2a78d6; padding: 6px 10px; margin: 8px 0; page-break-inside: avoid; }
    .warn { background: #fdf3ec; border-left: 4px solid #eb6834; padding: 6px 10px; margin: 8px 0; page-break-inside: avoid; }
    .toc li { margin: 2px 0; }
    """

    body = f"""
<div class="cover">
  <h1>slingpuck</h1>
  <div class="sub">Sim2real2sim reinforcement learning for an SO-101 arm playing Fast Sling Puck</div>
  <div class="sub">Project report: what is done, how the code works, what is still to do</div>
  <div class="meta">
    {date.today():%d %B %Y} &middot; git commit <code>{commit}</code> ({n_commits} commits)<br/>
    {n_src} source files, {n_lines} lines of Python &middot; {n_tests} tests, all passing<br/>
    Milestones M1&ndash;M4 done; M5 not started
  </div>
</div>

<div class="section">
<h2>Contents</h2>
<ol class="toc">
  <li>Summary</li><li>Status and history</li><li>Key decisions and findings</li>
  <li>How the system fits together</li><li>Configuration files</li><li>Results</li>
  <li>Code reference: every file, class and function</li><li>Tests</li><li>How to run</li>
  <li>What is still to do</li><li>Glossary</li>
</ol>

<h2>1. Summary</h2>
<p>The project builds a training and evaluation workspace for an SO-101 arm (6-DOF, Feetech STS3215 servos)
that plays tabletop sling puck on a moopok Fast Sling Puck board (size M), in three phases:
<b>Phase 1</b> goalkeeper (block pucks at the gate), <b>Phase 2</b> targeted shot (sling a puck through the
gate), and <b>Phase 3</b> a match against an opponent (choose when to block, sling or hold).</p>
<div class="box"><b>Done (M1&ndash;M4):</b>
<ul>
<li>A versioned configuration system: every unmeasured value is marked as a placeholder, and strict mode refuses
  to use it for real results.</li>
<li>Physics and sensing models: a fast 2D puck sim, a servo model, a camera model and a Kalman tracker, a band
  model with a closed hysteresis loop, arm deflection, and a MuJoCo backend with the real SO-101 model.</li>
<li>Three Gymnasium environments. Phases 1 and 2 are trained with an asymmetric actor-critic PPO (3 seeds each)
  and evaluated with confidence intervals. Phase 3 has a scripted opponent and rule-based baselines.</li>
<li>A system identification (sysid) pipeline, tested on synthetic logs: it fits the band, servo, compliance,
  friction, restitution and energy transfer, writes a new parameter version, and checks sim-vs-real parity.</li>
<li>Viewers (2D and 3D MuJoCo), a README, and {n_tests} tests.</li>
</ul></div>
<div class="warn"><b>Main open points:</b> real measurements (servo step response, band force test, shot logs,
human opponent timing), the real <b>scoring rules</b>, and milestone M5 (selector training, ablations,
vulnerability window, ONNX export and Jetson latency).</div>

<h2>2. Status and history</h2>
<table><tr><th>Milestone</th><th>Content</th><th>Status</th></tr>
<tr><td>M1</td><td>Physics, servo, camera, tracker, config versioning, unit tests</td><td>Done</td></tr>
<tr><td>M2</td><td>Phase 1 env, asymmetric PPO, save-rate eval, viewers</td><td>Done (3 seeds)</td></tr>
<tr><td>&mdash;</td><td>MuJoCo backend for Phase 1, 3D viewer, sim-to-sim eval (user request)</td><td>Done</td></tr>
<tr><td>M3</td><td>Phase 2 env, band and deflection models, sysid with synthetic logs, parity eval</td><td>Done (3 seeds)</td></tr>
<tr><td>M4</td><td>Opponent model, Phase 3 match env, frozen primitives, rule-based baselines</td><td>Done</td></tr>
<tr><td>M5</td><td>Selector training, ablations, vulnerability window, ONNX, latency</td><td>Not started</td></tr></table>
<p>All milestones wait for the user's review. Git history (oldest first):</p>
<table><tr><th>Commit</th><th>Message</th></tr>{log_rows}</table>
</div>

<div class="section">
<h2>3. Key decisions and findings</h2>
<ol>
<li><b>The original code had bugs that made its results invalid.</b> Opponent shots flew away from the gate (sign
error); goals during a sling recovery were not counted; the sim mapped action 1.0 to 0.19 rad and the robot stub to
0.5 rad; a stale camera frame was used again and again; the camera fell back to ground truth when no frame came;
random numbers were not seeded. All are fixed and covered by tests.</li>
<li><b>Board geometry from the listing (size M).</b> Outer 14.4 &times; 8.5 in; inner play area 0.335 &times; 0.185 m;
puck 1.19 &times; 0.4 in. The gate is only about 1.5 mm wider than the puck (user: "a little over 1.19 in"). The
user-measured puck mass of 27 g gives a density of 3.7 g/cm&sup3; (wood is 0.5&ndash;0.75), so it may be the mass of
6 pucks (6 &times; 4.5 g); this needs a check.</li>
<li><b>Phase 1 is a recovery task (user decision).</b> Because the gate is so narrow, a paddle settled at the gate
blocks everything. Phase 1 therefore measures the <i>time to cover</i> the gate when the arm starts away from it, which
is the input for the Phase 3 vulnerability window.</li>
<li><b>PPO learned a servo trick in 2D</b> (command the far side of the range, so the first-order lag keeps the
paddle at full speed). It beat "go to center" in 2D (e.g. 85% vs 78% at 1.9&ndash;2.3 m/s), but <b>not in MuJoCo</b>,
where the torque-limited arm starts more slowly (37 ms vs 81 ms to move halfway). The 2D servo model is the weakest
part of Phase 1; the real step response must be measured.</li>
<li><b>The band launches almost straight (Phase 2).</b> With a pre-stretched band and pulls of a few cm against an
18.5 cm span, the pull angle barely steers the puck; a shot scores only if the puck leaves the band within &plusmn;3 mm
of the gate center. The user chose <b>placement</b>: the arm slides the puck along the band, then pulls.</li>
<li><b>For a still puck, average the camera frames.</b> The constant-velocity Kalman filter gave ~2 mm error on a
still puck and made the correction step harmful; the frame mean gives ~0.7 mm.</li>
<li><b>Sysid works on synthetic data after four fixes.</b> Band stiffness, exponent and natural length trade off
(measure the natural length with a ruler); a deadband needs a grid search; frame selection by noisy positions biased
speeds 8% low; a bounce between two frames contaminated a segment. All shot parameters are now within about one
standard error over 6 synthetic data sets. Parity: the uncalibrated sim is 9.5 ms too fast in transit time; the
fitted one agrees (+0.7 ms).</li>
<li><b>Phase 3 is an event-level match over measured primitive tables</b> (70,000 decisions/s). With the
placeholder timings and the "first empty side wins" rule, the match is a <b>throughput race</b>: the plan's baseline
("sling only when the opponent reloads") loses, because a robot sling cycle (~3.1 s) is much longer than a human
reload (~0.6 s); the hand tell helps only a little. The answer to the research question depends on the real scoring
rules and the real robot and human timings.</li>
</ol>
</div>

<div class="section">
<h2>4. How the system fits together</h2>
{ARCH_SVG}
<p>The configuration flows into the physics and sensing models; the environments combine them into
Gymnasium tasks; the policies are trained on the environments and evaluated by eval/. Phase 3 uses the
Phase 1 and Phase 2 skills as <i>frozen primitives</i>: their measured performance tables. sysid closes
the sim2real2sim loop: real logs produce a new physics parameter version, which the sims load.</p>
<p class="label">Board frame</p>
<p>Origin at the board center; x across the width, y along the length. The agent half is y &lt; 0, the opponent
half y &gt; 0, and the center divider with the gate is at y = 0. Pucks launched by the agent move toward +y.</p>
<p class="label">Fidelity levels</p>
<table><tr><th>Sim</th><th>Used for</th><th>Speed</th></tr>
<tr><td>Fast2DPuckSim + ServoModel</td><td>Phase 1 and 2 training and eval</td><td>~2,300 control steps/s per process</td></tr>
<tr><td>MuJoCo (board, puck, SO-101 arm)</td><td>Phase 1 sim-to-sim check, 3D viewing</td><td>~180 episodes/s</td></tr>
<tr><td>MatchEnv (event level, primitive tables)</td><td>Phase 3 selector</td><td>~70,000 decisions/s</td></tr></table>
</div>

<div class="section">
<h2>5. Configuration files</h2>
<p><code>load_config(params_version)</code> merges, in this order: <code>servo.yaml</code> &rarr;
<code>physics_params_v&lt;N&gt;.yaml</code> &rarr; <code>env.yaml</code>.</p>
<table><tr><th>File</th><th>Content</th></tr>
<tr><td><code>configs/physics_params_v0.yaml</code></td><td>Measured or fitted physics, version 0 (uncalibrated): board, puck, band,
arm compliance, robot mounting and paddle, camera, servo overrides, and the placeholder list. Each value has a source tag:
[spec] listing text, [image] listing image, [user] user measurement, [derived] computed, TODO unknown.</td></tr>
<tr><td><code>configs/servo.yaml</code></td><td>Servo model: max rate (300 deg/s), lag time constant, deadband, command latency (all placeholders).</td></tr>
<tr><td><code>configs/env.yaml</code></td><td>Design choices: sim timing (physics 1 ms, control 30 Hz, MuJoCo 0.5 ms), tracker tuning,
<code>goalkeeper</code> task (pan range, speeds, pre-launch, recovery lock), <code>sling</code> task (ranges, noise, correction,
reward shaping), <code>opponent</code> (cycle timing, accuracy, tell), <code>match</code> (pucks, robot timings, block table grid),
<code>scoring</code> (placeholder rules) and <code>randomization</code> (domain randomization ranges).</td></tr>
<tr><td><code>configs/train_goalkeeper.yaml</code>, <code>train_sling.yaml</code></td><td>PPO hyperparameters, policy type (asymmetric or mlp),
number of envs, steps and seeds.</td></tr></table>
{placeholders_html(cfg)}
<p>Other data: <code>data/real_logs/schema.md</code> (how to record the sysid logs),
<code>data/real_logs/synthetic_example/</code> (small synthetic logs with <code>truth.yaml</code>),
<code>assets/mujoco/robotstudio_so101/</code> (SO-101 MuJoCo model and meshes), <code>assets/urdf/so101.urdf</code> (not used yet),
<code>cache/</code> (primitive tables), <code>logs/</code>, <code>tensorboard_logs/</code>, <code>results/</code> (outputs, not in git).</p>
</div>

<div class="section">
<h2>6. Results</h2>
<h3>Phase 1: goalkeeper, 2D physics (3 PPO seeds, 300 shots per speed bin)</h3>
{fig("save_rate_2d.png", "Save rate vs puck speed and vs paddle start offset. PPO beats 'go to center' in 2D through a servo-lag trick.")}
<h3>Phase 1: the same policies on MuJoCo physics (sim-to-sim)</h3>
{fig("mujoco_scene.png", "MuJoCo scene: SO-101 holding the paddle at the gate (side and gate cameras).", "80%")}
{fig("save_rate_mujoco.png", "On MuJoCo all policies lose 20-40 points above 1.2 m/s, and PPO is no better than 'go to center'.")}
<h3>Phase 2: targeted shot (1000 shots per policy and seed)</h3>
{fig("hole_rate.png", "PPO 94.6%, slide + camera correction 93.8%, slide to center 92.5%.")}
<h3>Sysid parity on synthetic logs</h3>
{fig("parity_synthetic.png", "Outcome parity cannot separate v0 from the fit; transit time can: v0 is 9.5 ms too fast, the fit agrees.")}
<h3>Phase 3: scripted selectors (400 matches each)</h3>
{fig("match_baselines.png", "Greedy slinging wins most; the plan's reload rule loses (placeholder timings and rules).")}
{fig("match_tell_sweep.png", "Tell sweep: the tell rule beats the reload rule, but its gain does not grow clearly with tell_strength.")}
</div>

<div class="section">
<h2>7. Code reference: every file, class and function</h2>
<p>For each file: what it is for, how it works, important notes, and a table of its classes, functions and methods
(read from the code). Small helper functions nested inside other functions are part of their parent's description.</p>
{code_reference_html()}
</div>

<div class="section">
<h2>8. Tests</h2>
<p>{n_tests} tests, all passing (<code>pytest -q</code>, about 40 s). They cover the physics sanity checks the plan asked
for (energy never increases through a bounce, the hysteresis loop closes, latency shifts signals correctly) and Gymnasium
API compliance for every env.</p>
<table><tr><th>Test file</th><th>Lines</th></tr>{tests_rows}</table>
<ul>
<li><b>test_config</b>: versions, latest, strict mode, servo overrides, randomization, run records.</li>
<li><b>test_servo / test_camera / test_tracker</b>: exact latency, rate limit, lag, deadband; frame rate, latency shift, dropouts, noise; convergence, latency compensation.</li>
<li><b>test_puck_dynamics</b>: friction stop, restitution, energy never increases, gate and divider, puck-puck, paddle contacts.</li>
<li><b>test_kinematics / test_goalkeeper_env / test_asymmetric_policy / test_stats</b>: mapping, check_env, outcomes, privileged isolation, CIs.</li>
<li><b>test_mujoco_goalkeeper</b>: paddle on the 2D arc, friction, calibrated restitution, gate events, check_env.</li>
<li><b>test_band / test_sling_env</b>: closed hysteresis loop, release energy, deflection, straight launch, Phase 2 env.</li>
<li><b>test_sysid</b>: schema checks, recovery of known values from synthetic logs, versioned output, synthetic guard.</li>
<li><b>test_match</b>: tell behavior, puck conservation, rewards, sling timing, hand ablation, reload detection.</li>
</ul>
</div>

<div class="section">
<h2>9. How to run</h2>
<p>Run from the project root with the miniconda Python (<code>python3</code>); the package is imported as
<code>src.slingpuck</code>. On macOS the live MuJoCo window needs <code>mjpython</code>.</p>
<pre>pytest -q                                                     # all tests

# Phase 1
python -m src.slingpuck.train.train_goalkeeper                # 3 seeds
python -m src.slingpuck.eval.save_rate_vs_speed --runs logs/goalkeeper/goalkeeper_v0_s*_*
python -m src.slingpuck.eval.save_rate_vs_speed --runs logs/goalkeeper/goalkeeper_v0_s*_* --backend mujoco
python -m src.slingpuck.eval.visualize_goalkeeper --run logs/goalkeeper/&lt;run&gt;
mjpython -m src.slingpuck.eval.view_mujoco --run logs/goalkeeper/&lt;run&gt;

# Phase 2
python -m src.slingpuck.train.train_sling
python -m src.slingpuck.eval.hole_rate --runs logs/sling/sling_v0_s*_*

# Sysid (try it on synthetic logs)
python -m src.slingpuck.sysid.synthetic --out data/real_logs/synthetic
python -m src.slingpuck.sysid.run_sysid --logs data/real_logs/synthetic --natural-length 0.152
python -m src.slingpuck.eval.hole_rate --parity data/real_logs/synthetic/shots.csv \\
       --versions v0 v1 --cfg-dir data/real_logs/synthetic/fit

# Phase 3
python -m src.slingpuck.eval.match_eval
python -m src.slingpuck.eval.match_eval --policies greedy_sling reload_rule tell_rule --tell 0 0.5 1

# This report
python docs/build_report.py</pre>
</div>

<div class="section">
<h2>10. What is still to do</h2>
<h3>Milestone M5</h3>
<ul>
<li>Train the Phase 3 selector (PPO on MatchEnv, asymmetric critic), several seeds; compare with the rule-based baselines.</li>
<li>Ablations: without hand kinematics (<code>--no-hand</code>), and a sweep over <code>tell_strength</code>; report how much
reaction time the anticipation recovers.</li>
<li>Vulnerability-window analysis (rebuild <code>vulnerability_window.py</code> on MatchEnv): risk of a goal vs the time of the
sling decision relative to the opponent cycle.</li>
<li>Deployment: update the ONNX export for the asymmetric actor, run the latency benchmark on the Jetson Orin Nano, and
finish the LeRobot interface (8-value observation, real tracker, hardware calls, rate and torque limits).</li>
<li>Run M5 either after the real values below are known, or as a sweep over them (user to decide).</li>
</ul>
<h3>Real measurements needed (all are placeholders now)</h3>
<table><tr><th>Measurement</th><th>Why it matters</th></tr>
<tr><td><b>Game scoring rules</b></td><td>Decide whether Phase 3 is a throughput race; the reward and win rule</td></tr>
<tr><td><b>Pan servo step response with the paddle mounted</b></td><td>Decides which servo model (2D or MuJoCo) is right; Phase 1 reaction time</td></tr>
<tr><td>Robot fetch, sling and move times</td><td>Length of the vulnerability window in Phase 3</td></tr>
<tr><td>Human opponent timing and accuracy (video)</td><td>Opponent cycle, threat rate, tell</td></tr>
<tr><td>Gate width, divider thickness, wall height</td><td>Gate geometry (Phase 1 threat zone, Phase 2 target)</td></tr>
<tr><td>One puck's mass; puck size on size M</td><td>Launch speed; all contacts</td></tr>
<tr><td>Band: peg spacing, distance to the end wall, max pull, natural length; force-gauge test</td><td>Band model fit</td></tr>
<tr><td>Shot log (incl. divider bounces) and shot repeatability</td><td>Compliance, friction, restitution, energy transfer; Phase 2 miss rate</td></tr>
<tr><td>Arm base position and height, joint zero offset, end-effector size and mass</td><td>Paddle arc, MuJoCo arm pose</td></tr>
<tr><td>Camera latency, noise, dropout rate</td><td>Tracker accuracy</td></tr></table>
<h3>Known limitations</h3>
<ul>
<li>The Kalman tracker does not model wall bounces or a launch from rest (lags ~10 cm at +80 ms).</li>
<li>Phase 1 training is slow per step because episodes are short; the pan encoder has no noise yet.</li>
<li>MuJoCo: the threat check uses the 2D sim; servo rate and lag are not randomized.</li>
<li>Phase 2: the band vertex is at the puck center; only deflection along the pull is modeled; camera latency and
paddle restitution cannot be fitted from the current logs; the arm's reach to the band must be checked.</li>
<li>Phase 3: outcomes come from tables, not physics; one opponent puck at a time; fixed fetch time; the opponent
never blocks.</li>
<li>Deployment code is from before the rework (M5).</li>
</ul>
</div>

<div class="section">
<h2>11. Glossary</h2>
<table>
<tr><td><b>sim2real2sim</b></td><td>Train in sim, test on the real robot, use real logs to correct the sim (sysid), and train again.</td></tr>
<tr><td><b>Placeholder</b></td><td>A config value that is not measured yet; listed under <code>placeholders:</code>; strict mode refuses it.</td></tr>
<tr><td><b>Domain randomization</b></td><td>Sampling physics values from ranges at every episode, so a policy does not depend on one exact value.</td></tr>
<tr><td><b>Asymmetric actor-critic</b></td><td>The actor (deployed) sees only real sensors; the critic (training only) also sees sim truth.</td></tr>
<tr><td><b>Privileged observation</b></td><td>Sim-only information (true positions, randomized values, opponent intent) given to the critic.</td></tr>
<tr><td><b>Restitution</b></td><td>Ratio of normal speed after and before a bounce (1 = no energy loss).</td></tr>
<tr><td><b>Hysteresis (band)</b></td><td>The band pulls back with less force than it took to stretch it; the loop area is the lost energy.</td></tr>
<tr><td><b>Compliance</b></td><td>How far the arm gives way per newton of band force (m/N).</td></tr>
<tr><td><b>Energy transfer</b></td><td>Fraction of the band's release work that the puck keeps as kinetic energy.</td></tr>
<tr><td><b>Recovery task / time to cover</b></td><td>Phase 1 setup where the paddle starts away from the gate; measures how fast the gate can be covered.</td></tr>
<tr><td><b>Frozen primitive</b></td><td>A trained or scripted low-level skill that Phase 3 uses as a fixed block (here: its measured outcome table).</td></tr>
<tr><td><b>Tell / tell_strength</b></td><td>How much the opponent's hand reveals where the shot will go (0 nothing, 1 exactly).</td></tr>
<tr><td><b>Vulnerability window</b></td><td>The time the gate is undefended because the arm is slinging or returning.</td></tr>
<tr><td><b>Wilson / t interval</b></td><td>95% confidence intervals for a proportion / for a mean across seeds.</td></tr>
<tr><td><b>Brier score</b></td><td>Mean squared error between predicted probabilities and 0/1 outcomes (lower is better).</td></tr>
<tr><td><b>Parity</b></td><td>Agreement between sim predictions and real logs for the same commands.</td></tr>
</table>
</div>
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>slingpuck report</title><style>{css}</style></head><body>{body}</body></html>"


def main():
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html())
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
