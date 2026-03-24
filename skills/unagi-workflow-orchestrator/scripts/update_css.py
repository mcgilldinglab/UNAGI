from pathlib import Path
import re

# Dev-only helper for regenerating the shipped console CSS from the skill checkout.
# Resolve paths relative to this script so the skill can be cloned elsewhere.
script_dir = Path(__file__).resolve().parent
css_path = script_dir.parent / "web" / "static" / "monitor.css"

with css_path.open("r", encoding="utf-8") as f:
    css = f.read()

# Replace root variables
new_root = """
:root {
  --bg: #0f172a;
  --panel: rgba(30, 41, 59, 0.7);
  --panel-strong: rgba(51, 65, 85, 0.95);
  --ink: #f8fafc;
  --ink-soft: #cbd5e1;
  --muted: #94a3b8;
  --line: rgba(255, 255, 255, 0.1);
  --accent: #3b82f6;
  --accent-2: #8b5cf6;
  --ok: #10b981;
  --warn: #f59e0b;
  --bad: #ef4444;
  --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  --glass-border: 1px solid rgba(255, 255, 255, 0.08);
}
"""
css = re.sub(r':root\s*\{[^}]+\}', new_root.strip(), css, count=1)

# Replace body styles
new_body = """
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  background-image: 
    radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15), transparent 50%),
    radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.15), transparent 50%);
  background-attachment: fixed;
  font-family: "Inter", "Segoe UI", sans-serif;
  letter-spacing: -0.01em;
}
"""
css = re.sub(r'body\s*\{[^}]+\}', new_body.strip(), css, count=1)

# Ensure transitions exist for interactive elements
css += """

/* Global Animations & Glassmorphism Additions */
a, button, input, select, textarea, .module-summary, .task-card, .artifact-card {
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.panel, .detail-card, .telemetry-card, .gpu-card, .stat-card, .subpanel, .task-card, .empty-state-panel, .artifact-card, .analysis-temporal-graph-widget, .analysis-stage-composition-widget, .module-section, .analysis-umap-widget {
  background: var(--panel) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: var(--glass-border) !important;
  box-shadow: var(--shadow) !important;
}

.task-card:hover, .artifact-card:hover, .button-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;
  border-color: rgba(255, 255, 255, 0.15) !important;
}

input, select, textarea {
  background: rgba(15, 23, 42, 0.6) !important;
  border: var(--glass-border) !important;
  color: var(--ink) !important;
}

input:focus, select:focus, textarea:focus {
  outline: none;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25) !important;
}

.button-link.primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
  color: white !important;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.button-link.primary:hover {
  filter: brightness(1.1);
  box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
}

.header-chip span, .status-pill, .segment-option, .chip-check, .idrem-path-chip {
  background: rgba(30, 41, 59, 0.8) !important;
  border: var(--glass-border) !important;
  backdrop-filter: blur(8px);
}

.filter-bar button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  border: none !important;
  transition: all 0.2s ease;
}
.filter-bar button:hover {
  filter: brightness(1.1);
  transform: translateY(-1px);
}

/* Remove hardcoded white / light backgrounds */
"""

# Hardcoded background replacements
css = css.replace('rgba(255, 255, 255, 0.7)', 'rgba(30, 41, 59, 0.6)')
css = css.replace('rgba(255, 255, 255, 0.66)', 'rgba(30, 41, 59, 0.6)')
css = css.replace('rgba(255, 255, 255, 0.72)', 'rgba(30, 41, 59, 0.6)')
css = css.replace('rgba(255, 255, 255, 0.6)', 'rgba(30, 41, 59, 0.5)')
css = css.replace('rgba(255, 255, 255, 0.82)', 'rgba(30, 41, 59, 0.8)')
css = css.replace('rgba(255, 255, 255, 0.5)', 'rgba(30, 41, 59, 0.4)')
css = css.replace('rgba(255, 255, 255, 0.92)', 'rgba(30, 41, 59, 0.9)')
css = css.replace('rgba(255, 255, 255, 0.78)', 'rgba(30, 41, 59, 0.7)')
css = css.replace('background: #fff;', 'background: var(--panel-strong);')

# Gradients
css = css.replace('linear-gradient(145deg, rgba(255, 248, 239, 0.96), rgba(244, 250, 247, 0.88))', 'linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9))')
css = css.replace('linear-gradient(135deg, rgba(193, 95, 49, 0.14), rgba(45, 125, 124, 0.12))', 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))')
css = css.replace('rgba(193, 95, 49, 0.28)', 'rgba(59, 130, 246, 0.4)')
css = css.replace('rgba(45, 125, 124, 0.12)', 'rgba(59, 130, 246, 0.15)')
css = css.replace('rgba(45, 125, 124, 0.22)', 'rgba(59, 130, 246, 0.25)')
css = css.replace('linear-gradient(135deg, var(--accent), #dd8f63)', 'linear-gradient(135deg, var(--accent), var(--accent-2))')

with css_path.open("w", encoding="utf-8") as f:
    f.write(css)
print("CSS updated successfully")
