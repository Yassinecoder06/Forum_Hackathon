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
