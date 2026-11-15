#!/usr/bin/env python3
"""Simplified pipeline diagrams and result tables on a white 16:9 canvas (stdlib only).

Example images come from docs/diagrams/sample.json:
  1. npm --prefix e2e run diagram-examples   (source frames + avatar frames, app must run)
  2. python scripts/build-diagrams.py        (this file: SVGs + index.html)
  3. npm --prefix e2e run diagram-png        (optional 2x PNGs for slides)
Skeletons are drawn as vectors directly from the sample's captured motion file.
"""
from html import escape
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/diagrams'
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = json.loads((OUT / 'sample.json').read_text(encoding='utf-8'))
MOTION = json.loads((ROOT / SAMPLE['motion']).read_text(encoding='utf-8'))
W, H = 1600, 900
FONT = 'Inter,Arial,Helvetica,sans-serif'
INK, MUTED, LINE = '#1d3a33', '#6b7d77', '#dfe5e1'
C = {'green': '#23684f', 'blue': '#336a91', 'amber': '#a0661c', 'purple': '#74578d', 'gray': '#8a9893'}
TINT = {'green': '#eef5f0', 'blue': '#edf3f8', 'amber': '#fbf3e6', 'purple': '#f3eff8', 'gray': '#f2f4f3'}


def wrap(text, size, width):
    return textwrap.wrap(text, width=max(4, int(width / (size * 0.52)))) or ['']


class Page:
    def __init__(self, name, title, subtitle=''):
        self.name = name
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                      f'width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>'
                      + ''.join(f'<marker id="m-{k}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
                                f'<path d="M0,1 L8,5 L0,9" fill="none" stroke="{v}" stroke-width="1.6"/></marker>'
                                for k, v in C.items())
                      + f'</defs><rect width="{W}" height="{H}" fill="#ffffff"/>']
        self.clip = 0
        self.text(64, 84, title, 36, INK, 700)
        if subtitle:
            self.text(64, 122, subtitle, 19, MUTED)

    def text(self, x, y, t, size=18, color=INK, weight=400, anchor='start', italic=False):
        style = ' font-style="italic"' if italic else ''
        self.parts.append(f'<text x="{x}" y="{y}" fill="{C.get(color, color)}" font-size="{size}" font-weight="{weight}" '
                          f'font-family="{FONT}" text-anchor="{anchor}"{style}>{escape(str(t))}</text>')

    def lines(self, x, y, items, size=16, color=INK, width=300, step=None, anchor='start'):
        step = step or size + 7
        rows = [r for item in items for r in wrap(item, size, width)]
        for i, row in enumerate(rows):
            self.text(x, y + i * step, row, size, color, anchor=anchor)
        return y + len(rows) * step

    def rect(self, x, y, w, h, fill, stroke=None, rx=14, sw=1.3, dash=False):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
        d = ' stroke-dasharray="6 5"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{s}{d}/>')

    def box(self, x, y, w, h, title, body=(), color='green', tag=None, center=False):
        self.rect(x, y, w, h, TINT[color], C[color])
        ty = y + 34
        if tag:
            self.text(x + 20, y + 28, tag.upper(), 12, color, 700)
            ty = y + 56
        anchor, tx = ('middle', x + w / 2) if center else ('start', x + 20)
        title_rows = wrap(title, 21, w - 40)
        for i, row in enumerate(title_rows):
            self.text(tx, ty + i * 26, row, 21, color, 700, anchor)
        by = ty + (len(title_rows) - 1) * 26 + 28
        end = self.lines(tx, by, body, 16, INK, w - 40, anchor=anchor)
        if end - 14 > y + h:
            raise ValueError(f'{self.name}: box overflow "{title}"')

    def arrow(self, pts, color='green', dashed=False, label=None, lx=None, ly=None):
        d = 'M' + ' L'.join(f'{x},{y}' for x, y in pts)
        dash = ' stroke-dasharray="8 6"' if dashed else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="2.2" stroke-linejoin="round" '
                          f'marker-end="url(#m-{color})"{dash}/>')
        if label:
            self.text(lx, ly, label, 14, color, 700)

    def image(self, x, y, w, h, href, rx=12, caption=None):
        self.clip += 1
        cid = f'c{self.clip}'
        self.parts.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>'
                          f'<image href="{href}" xlink:href="{href}" x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>'
                          f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{LINE}"/>')
        if caption:
            self.text(x + w / 2, y + h + 24, caption, 14, MUTED, anchor='middle')

    def skeleton(self, x, y, w, h, time_ms, caption=None):
        """Vector drawing of the captured joints nearest to time_ms (front view, metres)."""
        self.rect(x, y, w, h, '#fafbfa', LINE, 12, 1)
        frame = min(MOTION['frames'], key=lambda f: abs(f['time_ms'] - time_ms))
        j = frame['joints']
        neck = j.get('neck', [0, 1.45, 0])
        top = max(f['joints']['head'][1] for f in MOTION['frames'] if 'head' in f['joints']) + 0.12
        bottom = min(0.93, min(v[1] for fr in MOTION['frames'] for v in fr['joints'].values()) - 0.03)
        scale = min((w - 30) / 0.84, (h - 24) / (top - bottom))
        oy = y + 12 + ((h - 24) - (top - bottom) * scale) / 2

        def p(v):
            return x + w / 2 + (v[0] - neck[0]) * scale, oy + (top - v[1]) * scale

        def seg(a, b, color, width):
            if a in j and b in j:
                (x1, y1), (x2, y2) = p(j[a]), p(j[b])
                self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                                  f'stroke-width="{width}" stroke-linecap="round"/>')
        body = C['green']
        hips = {'left_hip': [neck[0] + 0.12, 1.0, 0], 'right_hip': [neck[0] - 0.12, 1.0, 0]}
        j = {**j, **hips}
        if 'head' in j:
            hx, hy = p(j['head'])
            self.parts.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="{0.085 * scale:.1f}" fill="none" stroke="{body}" stroke-width="3"/>')
        for a, b in [('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                     ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                     ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')]:
            seg(a, b, body, 4)
        for side, color in [('left', C['amber']), ('right', C['blue'])]:
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                chain = [f'{side}_wrist'] + [f'{side}_{finger}_{i}' for i in range(1, 5)]
                for a, b in zip(chain, chain[1:]):
                    seg(a, b, color, 1.8)
        for name, v in j.items():
            if name.endswith(('_shoulder', '_elbow', '_wrist')):
                cx, cy = p(v)
                self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#ffffff" stroke="{body}" stroke-width="2"/>')
        if caption:
            self.text(x + w / 2, y + h + 24, caption, 14, MUTED, anchor='middle')

    def table(self, x, y, widths, header, rows, highlight=None, row_h=58, notes=None):
        total = sum(widths)
        self.rect(x, y, total, row_h, '#f4f7f5', None, 10)
        cx = x
        for wcol, head in zip(widths, header):
            self.text(cx + 18, y + row_h / 2 + 6, head, 15, MUTED, 700)
            cx += wcol
        for r, row in enumerate(rows):
            ry = y + row_h * (r + 1)
            if highlight is not None and r == highlight:
                self.rect(x, ry + 3, total, row_h - 6, TINT['green'], None, 10)
            self.parts.append(f'<line x1="{x}" y1="{ry + row_h}" x2="{x + total}" y2="{ry + row_h}" stroke="{LINE}"/>')
            cx = x
            for c, (wcol, cell) in enumerate(zip(widths, row)):
                bold = 700 if c == 0 else 400
                color = C['green'] if (highlight == r and c == 0) else INK
                rows_txt = wrap(cell, 16, wcol - 30)
                top = ry + row_h / 2 + 6 - (len(rows_txt) - 1) * 10
                for i, t in enumerate(rows_txt):
                    self.text(cx + 18, top + i * 20, t, 16, color, bold)
                cx += wcol
        return y + row_h * (len(rows) + 1)

    def footer(self, text):
        self.parts.append(f'<line x1="64" y1="852" x2="{W - 64}" y2="852" stroke="{LINE}"/>')
        self.text(64, 878, text, 13, MUTED)

    def save(self):
        self.parts.append('</svg>')
        (OUT / self.name).write_text('\n'.join(self.parts), encoding='utf-8')
        return self.name


T = SAMPLE['times_ms']
CREDIT = f"Example: {SAMPLE['label']} · source video: {SAMPLE['credit']}"
pages = []

# 1 ─ Overview ──────────────────────────────────────────────────────────────
p = Page('01-overview.svg', 'From a French sentence to a signing avatar',
         'The whole application in five steps. Every sign comes from a real recorded signer.')
stages = [('French text', ['Typed in the Studio'], 'green'),
          ('Concept lookup', ['Words → known signs', 'Unknown words shown'], 'green'),
          ('Sign sequence', ['Ordered list of', 'sign IDs (glosses)'], 'green'),
          ('Motion stitching', ['Recorded clips joined', 'with smooth bridges'], 'green'),
          ('3D avatar', ['Plays the motion,', 'exports a video'], 'green')]
bx, bw, gap, by = 64, 248, 54, 350
for i, (title, body, color) in enumerate(stages):
    x = bx + i * (bw + gap)
    p.box(x, by, bw, 124, title, body, color, center=True)
    if i:
        p.arrow([(x - gap + 4, by + 62), (x - 6, by + 62)])
look_c, stitch_c = bx + (bw + gap) + bw / 2, bx + 3 * (bw + gap) + bw / 2
p.box(look_c - 110, 176, stitch_c - look_c + 220, 96, 'Real sign library',
      ['1,446 signs captured from LSF videos (Dicta-Sign + Wikimedia Commons)'], 'blue', center=True)
p.arrow([(look_c, 272), (look_c, by - 6)], 'blue', label='known words', lx=look_c + 12, ly=318)
p.arrow([(stitch_c, 272), (stitch_c, by - 6)], 'blue', label='motion clips', lx=stitch_c + 12, ly=318)
# worked example row
ey = 530
p.text(64, ey + 4, 'EXAMPLE', 13, MUTED, 700)
ex = [f'« {SAMPLE["text"]} »', SAMPLE['label'].lower(), SAMPLE['label'],
      f'{MOTION["duration_ms"] / 1000:.2f} s · {len(MOTION["frames"])} frames']
for i, t in enumerate(ex):
    x = bx + i * (bw + gap) + bw / 2
    p.rect(x - 110, ey + 24, 220, 44, '#ffffff', LINE, 22)
    p.text(x, ey + 52, t, 16, INK, 600, 'middle')
p.image(bx + 4 * (bw + gap), ey + 20, bw, 260, 'examples/avatar_1.png')
p.footer(CREDIT)
pages.append(p.save())

# 2 ─ Capture ───────────────────────────────────────────────────────────────
p = Page('02-capture.svg', 'Turning a signer video into motion data',
         'Each library sign is captured once from its source video, then reused by the avatar.')
p.image(64, 170, 300, 300, 'examples/source_1.png', caption='Source video frame')
p.arrow([(374, 320), (430, 320)])
p.box(440, 245, 280, 150, 'Landmark detection', ['MediaPipe Holistic', 'Body 33 · face 468', 'Hands 21 points each'], center=True)
p.arrow([(730, 320), (786, 320)])
p.skeleton(796, 170, 300, 300, T[1], caption='Captured joints (same moment)')
p.arrow([(1106, 320), (1162, 320)])
p.box(1172, 245, 364, 150, 'Motion file', [f'{len(MOTION["tracked_joints"])} joints per frame', 'Metres · Y up · timestamps',
                                         'Face: mouth + eyebrows'], 'blue', center=True)
p.text(64, 560, 'When a hand is missing, three detectors are tried in turn', 20, INK, 700)
ladder = [('1 · Holistic', ['Full-body model', 'finds the hand'], 'green'),
          ('2 · Hands detector', ['Second model on', 'the whole frame'], 'green'),
          ('3 · Zoom on wrist', ['Crop at 2 sizes,', 'both must agree'], 'green'),
          ('Still missing', ['Left empty —', 'never invented'], 'amber')]
for i, (title, body, color) in enumerate(ladder):
    x = 64 + i * 372
    p.box(x, 590, 330, 116, title, body, color, center=True)
    if i:
        p.arrow([(x - 38, 648), (x - 6, 648)], 'gray', label=None)
p.lines(64, 760, ['Quality gate: clips with poor hand tracking stay out of the library '
                  '(Dicta-Sign: 1,007 accepted, 18 held for review).'], 16, MUTED, 1460)
p.footer(CREDIT)
pages.append(p.save())

# 3 ─ Retargeting ───────────────────────────────────────────────────────────
p = Page('03-retargeting.svg', 'Driving the 3D character from captured points',
         'The same five steps run for every frame. Captured data is never modified, only adapted.')
p.skeleton(64, 190, 330, 420, T[1], caption='Captured joints')
steps = [('1  Scale & anchor', 'Match shoulder width, pin at the neck'),
         ('2  Arm IK', 'Shoulder + elbow angles reach the wrist'),
         ('3  Palm orientation', 'From wrist, index, middle and pinky'),
         ('4  Fingers', 'Bend and spread per joint, within limits'),
         ('5  Clean-up', 'Remove spikes · fill short gaps · stop hand overlap')]
for i, (title, body) in enumerate(steps):
    y = 175 + i * 92
    p.rect(470, y, 640, 78, TINT['green'], C['green'])
    p.text(494, y + 33, title, 20, 'green', 700)
    p.text(494, y + 59, body, 16, INK)
p.arrow([(404, 400), (460, 400)])
p.arrow([(1120, 400), (1176, 400)])
p.image(1186, 190, 350, 420, 'examples/avatar_1.png', caption='Rendered avatar (same moment)')
p.lines(470, 680, ['Display-only layer on top: breathing, blinking and small eye/head movements, '
                   'reduced while signing because head movement can carry meaning in LSF.'], 16, MUTED, 640)
p.footer(CREDIT)
pages.append(p.save())

# 4 ─ Timeline ──────────────────────────────────────────────────────────────
p = Page('04-timeline.svg', 'Joining signs into one smooth sequence',
         'Each clip plays only its signing core; new frames are added only between cores.')
ty = 210
segs = [('Sign 1', 330, 'green'), ('bridge', 90, 'amber'), ('Sign 2', 420, 'green'), ('bridge', 90, 'amber'), ('Sign 3', 300, 'green')]
x = 64
for label, w, color in segs:
    p.rect(x, ty, w, 64, TINT[color], C[color], 10)
    p.text(x + w / 2, ty + 39, '333 ms' if label == 'bridge' else label, 17, color, 700, 'middle')
    x += w + 6
p.text(64, ty - 18, 'SEQUENCE', 13, MUTED, 700)
p.lines(64, ty + 104, ['The hands-at-rest start and end of each dictionary clip are skipped inside a sentence. Bridges keep '
                        'the arms moving at their measured speed, and the hand shape morphs around the wrist.'], 16, MUTED, 1220)
# bridge curve chart: current Hermite (keeps end speed) vs the earlier quintic (stops at both ends)
cx0, cy0, cw, ch = 1330, 170, 206, 150
p.rect(cx0, cy0, cw, ch, '#fafbfa', LINE, 10, 1)
for name, color, dash in (('quintic', C['gray'], ' stroke-dasharray="5 4"'), ('hermite', C['amber'], '')):
    pts = []
    for k in range(41):
        u = k / 40
        if name == 'quintic':
            b = u ** 3 * (10 + u * (-15 + 6 * u))
        else:
            m = 0.8  # end speeds as a fraction of the bridge's travel
            b = (u ** 3 - 2 * u ** 2 + u) * m + (-2 * u ** 3 + 3 * u ** 2) + (u ** 3 - u ** 2) * m
        pts.append(f'{cx0 + 14 + u * (cw - 28):.1f},{cy0 + ch - 16 - b * (ch - 32):.1f}')
    p.parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
p.text(cx0 + cw / 2, cy0 + ch + 24, 'Arm path over 333 ms', 14, MUTED, anchor='middle')
p.text(cx0 + cw / 2, cy0 + ch + 44, 'solid: now · dashed: before', 13, MUTED, anchor='middle')
# audio timing
ay = 450
p.text(64, ay - 18, 'OPTIONAL: FOLLOW A VOICE RECORDING', 13, MUTED, 700)
audio = [('voiced', 340), ('pause', 120), ('voiced', 500), ('pause', 150), ('voiced', 338)]
x = 64
for kind, w in audio:
    voiced = kind == 'voiced'
    p.rect(x, ay, w, 48, TINT['blue'] if voiced else '#ffffff', C['blue'], 10, 1.2, dash=not voiced)
    p.text(x + w / 2, ay + 30, 'speech' if voiced else 'pause', 15, 'blue', 600, 'middle')
    x += w + 6
p.lines(64, ay + 88, ['Signing time is stretched across the voiced parts and holds still during pauses.'], 16, MUTED, 1460)
# hand states
hy = 640
p.text(64, hy - 18, 'HAND TRACKING STATES DURING PLAYBACK', 13, MUTED, 700)
states = [('Observed', 'Tracked in the video', 'green', 400), ('Short gap', '≤ 180 ms: interpolated', 'blue', 300),
          ('Long gap', 'Held, then fades to rest', 'amber', 380), ('Untracked', 'Neutral rest pose', 'gray', 350)]
x = 64
for title, body, color, w in states:
    p.rect(x, hy, w, 92, TINT[color], C[color], 10)
    p.text(x + 18, hy + 38, title, 19, color, 700)
    p.text(x + 18, hy + 66, body, 16, INK)
    x += w + 8
p.footer('Bridge length 333 ms; hand gap threshold 180 ms. Implementation: frontend/src/utils/planStitcher.ts, avatarPlayback.ts')
pages.append(p.save())

# 5 ─ Frames: source → pose → avatar ─────────────────────────────────────────
p = Page('05-source-pose-avatar.svg', f'One sign, three views: {SAMPLE["label"]}',
         'The same moments in the source video, in the captured joints, and on the avatar.')
labels = SAMPLE.get('time_labels') or [''] * len(T)
col_w, col_gap, x0, y0, cell_h = 384, 30, 300, 160, 205
for c, t in enumerate(T):
    x = x0 + c * (col_w + col_gap)
    p.text(x + col_w / 2, y0, f'{t / 1000:.2f} s · {labels[c]}', 17, INK, 700, 'middle')
rows = [('Source video', 'image', 'source'), ('Captured joints', 'skeleton', None), ('Avatar', 'image', 'avatar')]
for r, (label, kind, prefix) in enumerate(rows):
    y = y0 + 22 + r * (cell_h + 18)
    p.text(64, y + cell_h / 2 + 6, label, 19, 'green', 700)
    for c, t in enumerate(T):
        x = x0 + c * (col_w + col_gap)
        if kind == 'image':
            p.image(x, y, col_w, cell_h, f'examples/{prefix}_{c}.png')
        else:
            p.skeleton(x, y, col_w, cell_h, t)
p.footer(CREDIT)
pages.append(p.save())

# 6 ─ Data sources table ────────────────────────────────────────────────────
p = Page('06-data-sources.svg', 'Data sources', 'What each dataset contributes and how much of it is usable.')
end = p.table(64, 170, [300, 420, 360, 200, 192], ['Source', 'Content', 'Volume', 'Used for', 'Status'], [
    ['Dicta-Sign (Hamburg)', 'Public LSF dictionary videos', '1,025 videos → 1,007 accepted, 18 in review', 'Sign library', 'Rights pending'],
    ['Wikimedia Commons', 'Single-sign LSF videos', '446 videos → 439 accepted, 7 rejected', 'Sign library', 'Per-file CC'],
    ['UD / SUD STKAutogramm', 'French ↔ LSF gloss annotations', '128 sentence pairs (61 / 28 / 39)', 'Text→gloss research', 'Research only'],
], row_h=78)
stats = [('1,446', 'real signs in the library'), ('907', 'distinct Dicta-Sign glosses'), ('36,414', 'captured motion frames'),
         ('42.5 min', 'of captured signing')]
for i, (big, small) in enumerate(stats):
    x = 64 + i * 372
    p.rect(x, end + 60, 340, 130, '#fafbfa', LINE, 14, 1)
    p.text(x + 26, end + 122, big, 38, 'green', 700)
    p.text(x + 26, end + 158, small, 16, MUTED)
p.footer('Sources: data/source_inventory/dictasign_ingestion_report.json, ingestion_report.json, research/DATASET_CARD.md')
pages.append(p.save())

# 7 ─ Language models table ─────────────────────────────────────────────────
p = Page('07-language-results.svg', 'Text → sign: what was tried',
         'Exact match and WER on 39 held-out test sentences (lower WER is better; above 100% means extra words).')
end = p.table(64, 160, [430, 190, 190, 150, 512], ['Approach', 'Exact match', 'WER', 'BLEU', 'Observed behaviour'], [
    ['Retrieval baseline', '0%', '120.9%', '0.301', 'Best of the four, still not usable'],
    ['T5-small + LoRA (CPU)', '0%', '207.3%', '0', 'Mostly copies the French input'],
    ['mT5-small + LoRA, 8 epochs', '0%', '100%', '0', 'Outputs an empty placeholder token'],
    ['mT5-small, best val., 21 epochs', '0%', '518.0%', '0.042', 'Repeats and invents gloss labels'],
], highlight=0, row_h=62)
p.text(64, end + 60, 'Concept planning (24 test sentences, picks signs from the library)', 20, INK, 700)
p.table(64, end + 84, [430, 300, 220, 522], ['Approach', 'Exact sequence', 'F1', 'Time per sentence'], [
    ['Qwen3.5-9B + guards', '13 / 24 (54.2%)', '0.760', '6.8 s (median, local GPU)'],
    ['Literal word lookup', '12 / 24 (50.0%)', '0.708', '0.003 s'],
], highlight=0, row_h=62)
p.footer('Conclusion: 128 sentence pairs are far too few to train a translator; the app uses lookup, with optional Qwen suggestions.')
pages.append(p.save())

# 8 ─ Motion quality ────────────────────────────────────────────────────────
p = Page('08-motion-quality.svg', 'Motion quality: measured improvements', 'Engineering measurements on real captures. Lower is better everywhere.')
p.text(64, 180, 'Filling missing frames inside a sign (93 test windows)', 19, INK, 700)
end = p.table(64, 196, [230, 170, 170, 170], ['Method', 'RMSE', 'Boundary', 'Per clip'], [
    ['CGAN (trained)', '0.315', '0.167', '0.252'],
    ['Linear', '0.320', '0.167', '0.273'],
    ['Quintic', '0.330', '0.173', '0.285'],
], highlight=0, row_h=56)
p.lines(64, end + 34, ['The CGAN wins by a small margin within a sign. Between signs the player uses '
                       'deterministic quintic bridges, which need no training.'], 15, MUTED, 720)
# finger jitter bars
jit = [('BONJOUR', 56.6, 12.7), ('BIBLIOTHÈQUE', 79.8, 18.7), ('MERCI', 58.3, 19.4), ('RENCONTRER', 35.1, 12.0),
       ('ENSEMBLE', 78.7, 26.5), ('LAVER', 49.8, 20.6)]
gx, gy = 880, 180
p.text(gx, gy, 'Finger jitter (95th percentile, ° per 40 ms)', 19, INK, 700)
for i, (name, before, after) in enumerate(jit):
    y = gy + 36 + i * 46
    p.text(gx, y + 17, name, 15, INK, 600)
    p.rect(gx + 150, y, before * 5.2, 12, '#e3e8e5', None, 6)
    p.rect(gx + 150, y + 15, after * 5.2, 12, C['green'], None, 6)
    p.text(gx + 158 + before * 5.2, y + 11, f'{before:.1f}', 13, MUTED)
    p.text(gx + 158 + after * 5.2, y + 26, f'{after:.1f}', 13, 'green', 700)
ly = gy + 36 + 6 * 46 + 8
p.rect(gx, ly, 14, 10, '#e3e8e5', None, 3); p.text(gx + 22, ly + 10, 'before', 13, MUTED)
p.rect(gx + 90, ly, 14, 10, C['green'], None, 3); p.text(gx + 112, ly + 10, 'after', 13, MUTED)
# collision + brow stats
cards = [('16.3 → 0.3 mm', 'Mean hand overlap, 14 phrases (−98%)'), ('44.0 → 18.2 mm', 'Worst hand overlap, 14 phrases'),
         ('178 → 0', 'Saturated eyebrow samples (of 410)')]
for i, (big, small) in enumerate(cards):
    x = 64 + i * 500
    p.rect(x, 650, 460, 120, '#fafbfa', LINE, 14, 1)
    p.text(x + 24, 705, big, 32, 'green', 700)
    p.text(x + 24, 742, small, 16, MUTED)
p.footer('Sources: research/runs/transition-intrasign-v1/metrics.json, docs/AVATAR_PLAYBACK.md, docs/HAND_COLLISION_FIX.md')
pages.append(p.save())

# Gallery page
items = ''.join(f'<figure><img src="{n}" alt="{n}"><figcaption>{n}</figcaption></figure>' for n in pages)
(OUT / 'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>SignaCulture diagrams</title>
<style>body{{margin:0;background:#fff;font-family:Inter,Arial,sans-serif;color:#1d3a33}}main{{max-width:1600px;margin:auto;padding:24px}}
figure{{margin:0 0 40px}}img{{width:100%;border:1px solid #e6ebe8;border-radius:8px}}figcaption{{font-size:13px;color:#6b7d77;margin-top:6px}}</style>
<main>{items}</main></html>''', encoding='utf-8')
print('Wrote', ', '.join(pages), '+ index.html')
