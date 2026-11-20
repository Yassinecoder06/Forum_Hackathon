"""Word-level prosody from measured acoustic frames and ASR word timings.

Prominence, pace and liveliness are acoustic estimates, not linguistic stress annotation:
they suggest which words a speaker emphasised so the avatar's signing can follow the voice.
"""
import difflib
import math
import re
import statistics

VOWEL_GROUP = re.compile(r"[aeiouyàâäéèêëîïôöùûüÿœæ]+", re.IGNORECASE)
WORD_CHARS = re.compile(r"[^\w]+", re.UNICODE)

# Weighted z-score combination; floors stop tiny natural variation in flat speech from
# being amplified into "emphasis" by the normalisation.
WEIGHTS = {'energy': 0.45, 'pitch': 0.35, 'length': 0.2}
STD_FLOOR = {'energy': 2.0, 'pitch': 1.5, 'length': 0.04}  # dB, semitones, s/syllable
PROMINENCE_MIN = 1.0
# At least one raw cue must also clearly exceed the speaker's typical word.
RAW_MIN = {'energy': 3.0, 'pitch': 2.0, 'length': 1.3}  # +dB, +semitones, x median


def syllables(text):
    """French syllable estimate: vowel groups, minus a silent final 'e'/'es'/'ent'."""
    word = text.casefold()
    groups = VOWEL_GROUP.findall(word)
    count = len(groups)
    if count > 1 and re.search(r"[^aeiouyàâäéèêëîïôöùûüÿœæ](e|es|ent)$", word):
        count -= 1
    return max(1, count)


def normalize(text):
    return WORD_CHARS.sub('', text.casefold())


def _field(item, name):
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _frames_in(frames, start, end):
    return [f for f in frames if _field(f, 'end_ms') > start and _field(f, 'start_ms') < end]


PHRASE_PAUSE_MS = 300
MIN_PHRASE_FOR_TREND = 4


def _expected(values, times, phrases):
    """Value each word would have without stress.

    Declination (loudness and pitch falling across a phrase) restarts after pauses, so the drift is
    fitted per phrase, leaving the judged word out so a stressed word cannot hide itself. Words in
    short phrases are compared with the speaker's typical (median) word instead."""
    typical = statistics.median(values)
    out = [typical] * len(values)
    for phrase in phrases:
        if len(phrase) < MIN_PHRASE_FOR_TREND:
            continue
        for i in phrase:
            others = [j for j in phrase if j != i]
            ts, vs = [times[j] for j in others], [values[j] for j in others]
            mt, mv = statistics.fmean(ts), statistics.fmean(vs)
            var = sum((t - mt) ** 2 for t in ts)
            slope = sum((t - mt) * (v - mv) for t, v in zip(ts, vs)) / var if var else 0.0
            out[i] = mv + slope * (times[i] - mt)
    return out


def _z(values, floor):
    mean = statistics.fmean(values)
    std = max(statistics.pstdev(values), floor)
    return [(v - mean) / std for v in values]


def analyse_prosody(frames, words):
    """Per-word prominence, pace and liveliness. Returns None without usable words."""
    words = [w for w in words if normalize(_field(w, 'text'))]
    if not words:
        return None
    pitches = [_field(f, 'pitch_hz') for f in frames if _field(f, 'pitch_hz')]
    reference = statistics.median(pitches) if pitches else None

    def semitones(hz):
        return 12 * math.log2(hz / reference)

    raw = []
    for w in words:
        start, end = _field(w, 'start_ms'), _field(w, 'end_ms')
        inside = _frames_in(frames, start, end)
        voiced = [f for f in inside if not _field(f, 'is_pause')] or inside
        energy = statistics.fmean(_field(f, 'energy') for f in voiced) if voiced else 0.0
        energy_db = 20 * math.log10(max(energy, 1e-4))
        word_pitch = [_field(f, 'pitch_hz') for f in voiced if _field(f, 'pitch_hz')]
        # Mean, not peak: single-frame pitch errors would otherwise dominate.
        pitch_st = statistics.fmean(semitones(p) for p in word_pitch) if word_pitch and reference else None
        count = syllables(_field(w, 'text'))
        duration = max(1, end - start) / 1000
        # Lengthening uses only the voiced part: ASR word spans can absorb neighbouring silence
        # (notably before the first word), which would otherwise read as a drawn-out word.
        voiced_s = sum(min(end, _field(f, 'end_ms')) - max(start, _field(f, 'start_ms'))
                       for f in inside if not _field(f, 'is_pause')) / 1000
        # Liveliness: pitch spread within +/-1 s of the word centre (semitones std).
        centre = (start + end) / 2
        around = [_field(f, 'pitch_hz') for f in _frames_in(frames, centre - 1000, centre + 1000)
                  if _field(f, 'pitch_hz')]
        spread = statistics.pstdev([semitones(p) for p in around]) if len(around) > 2 and reference else 0.0
        raw.append(dict(text=_field(w, 'text'), start_ms=start, end_ms=end, syllables=count,
                        energy=energy_db, pitch=pitch_st, length=max(voiced_s, 0.04) / count,
                        syl_per_s=count / duration, liveliness=min(1.0, max(0.0, (spread - 1.0) / 4.0))))

    # Words with no measured pitch take the utterance mean so they are neither favoured nor penalised.
    known = [r['pitch'] for r in raw if r['pitch'] is not None]
    fill = statistics.fmean(known) if known else 0.0
    for r in raw:
        if r['pitch'] is None:
            r['pitch'] = fill
    centres = [(r['start_ms'] + r['end_ms']) / 2000 for r in raw]
    phrases, current = [], [0]
    for i in range(1, len(raw)):
        if raw[i]['start_ms'] - raw[i - 1]['end_ms'] >= PHRASE_PAUSE_MS:
            phrases.append(current); current = []
        current.append(i)
    phrases.append(current)

    def relative(key):
        values = [r[key] for r in raw]
        return [v - e for v, e in zip(values, _expected(values, centres, phrases))]
    level = {'energy': relative('energy'), 'pitch': relative('pitch'),
             'length': [r['length'] for r in raw]}
    z = {k: _z(level[k], STD_FLOOR[k]) for k in WEIGHTS}
    medians = {k: statistics.median(level[k]) for k in WEIGHTS}
    for i, r in enumerate(raw):
        r['prominence'] = sum(WEIGHTS[k] * z[k][i] for k in WEIGHTS)
        r['raw_cue'] = (level['energy'][i] - medians['energy'] >= RAW_MIN['energy']
                        or level['pitch'][i] - medians['pitch'] >= RAW_MIN['pitch']
                        or level['length'][i] >= RAW_MIN['length'] * medians['length'])
    limit = max(1, math.ceil(len(raw) / 4))
    ranked = sorted(range(len(raw)), key=lambda i: -raw[i]['prominence'])[:limit]
    total_syllables = sum(r['syllables'] for r in raw)
    speaking_s = sum(max(1, r['end_ms'] - r['start_ms']) for r in raw) / 1000
    out = []
    for i, r in enumerate(raw):
        emphasized = len(raw) >= 2 and i in ranked and r['prominence'] >= PROMINENCE_MIN and r['raw_cue']
        out.append(dict(text=r['text'], start_ms=r['start_ms'], end_ms=r['end_ms'],
                        prominence=round(r['prominence'], 3), emphasized=emphasized,
                        syl_per_s=round(r['syl_per_s'], 2), liveliness=round(r['liveliness'], 3)))
    return dict(pace_syl_per_s=round(total_syllables / speaking_s, 2), words=out)


def align_tokens(tokens, words):
    """Index of the transcript word matching each text token (None when unmatched)."""
    a = [normalize(t) for t in tokens]
    b = [normalize(_field(w, 'text')) for w in words]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = [None] * len(tokens)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            out[block.a + k] = block.b + k
    return out


def segment_prosody(spans, tokens, prosody):
    """Per gloss: matched word span, emphasis (0..1) and liveliness, or None if unmatched."""
    if not prosody or not spans:
        return None
    words = prosody['words'] if isinstance(prosody, dict) else [w.model_dump() for w in prosody.words]
    index = align_tokens(tokens, words)
    out = []
    for start, end in spans:
        matched = [words[j] for j in index[start:end] if j is not None]
        if not matched:
            out.append(None)
            continue
        emphasized = [w for w in matched if w['emphasized']]
        emphasis = min(1.0, max(w['prominence'] for w in emphasized) / 2.0) if emphasized else 0.0
        out.append(dict(word_start_ms=min(w['start_ms'] for w in matched),
                        word_end_ms=max(w['end_ms'] for w in matched),
                        emphasis=round(emphasis, 3),
                        liveliness=round(statistics.fmean(w['liveliness'] for w in matched), 3)))
    return out
