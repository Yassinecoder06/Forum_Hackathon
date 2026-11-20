"""Proportional timing using measured voiced regions; not word-level alignment."""
def voice_regions(audio):
    return [(f.start_ms,f.end_ms) for f in audio.frames if not f.is_pause] if audio else []

def segment_timing(motions,audio,transition_ms=0):
    original=sum(m.duration_ms for m in motions)+max(0,len(motions)-1)*transition_ms
    regions=voice_regions(audio)
    voiced=sum(end-start for start,end in regions)
    if not audio or not voiced or not original:
        position=0;segments=[]
        for motion in motions:
            segments.append((position,motion.duration_ms));position+=motion.duration_ms+transition_ms
        return segments,original,'original'
    def at_progress(progress):
        target=progress*voiced
        elapsed=0
        for start,end in regions:
            if target<elapsed+end-start:
                return round(start+target-elapsed)
            elapsed+=end-start
        return audio.duration_ms
    starts=[];elapsed=0
    for motion in motions:
        starts.append(at_progress(elapsed/original));elapsed+=motion.duration_ms+transition_ms
    segments=[(start,max(1,(starts[i+1] if i+1<len(starts) else audio.duration_ms)-start)) for i,start in enumerate(starts)]
    return segments,audio.duration_ms,'audio_voiced'

# Word-aligned timing: each sign starts at its spoken word(s). Playback rate is clamped so signs
# stay readable; emphasised signs are slowed slightly and end with a short hold.
MIN_BRIDGE_MS = 200
RATE_MIN, RATE_MAX = 0.75, 1.35
EMPHASIS_SLOWDOWN = 0.12
EMPHASIS_HOLD_MS = 220


def word_timing(motions, prosody, audio):
    """(start_ms, duration_ms, hold_ms) per sign, total ms; None when under half the signs align."""
    if not motions or not prosody or len(prosody) != len(motions) or not audio:
        return None
    if sum(p is not None for p in prosody) * 2 < len(motions):
        return None
    out, previous_end = [], 0
    for i, (motion, p) in enumerate(zip(motions, prosody)):
        natural = max(1, motion.duration_ms)
        gap = MIN_BRIDGE_MS if i else 0
        emphasis = p['emphasis'] if p else 0.0
        if p:
            start = max(p['word_start_ms'], previous_end + gap)
            following = next((q['word_start_ms'] for q in prosody[i + 1:] if q), None)
            target = (following - MIN_BRIDGE_MS if following is not None else audio.duration_ms) - start
            rate = min(RATE_MAX, max(RATE_MIN, natural / max(1, target)))
        else:
            start, rate = previous_end + gap, 1.0
        hold = round(EMPHASIS_HOLD_MS * emphasis)
        duration = round(natural / rate * (1 + EMPHASIS_SLOWDOWN * emphasis)) + hold
        out.append((start, duration, hold))
        previous_end = start + duration
    return out, max(previous_end, audio.duration_ms)
