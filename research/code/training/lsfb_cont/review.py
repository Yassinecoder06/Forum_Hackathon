"""Create a local video-backed review pack without unpacking the full video TAR."""
import argparse
import html
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


def recording_from_index(root, rid, output):
    member = rid+'.mp4'
    if Path(member).name != member:
        raise ValueError('Unsafe recording ID')
    index = json.loads((root/'videos.tar.index.json').read_text())
    offset, length = index[member]
    archive = root/'videos.tar'
    if offset < 512 or length <= 0 or offset+length > archive.stat().st_size:
        raise ValueError('Video index outside archive')
    with archive.open('rb') as f:
        f.seek(offset-512)
        header = tarfile.TarInfo.frombuf(f.read(512), 'utf-8', 'surrogateescape')
        if header.name != member or header.size != length or not header.isfile():
            raise ValueError('Index does not match TAR member: '+member)
        remaining = length
        with output.open('wb') as dest:
            while remaining:
                block = f.read(min(1024*1024, remaining))
                if not block:
                    raise ValueError('Truncated video member')
                dest.write(block)
                remaining -= len(block)


PAGE = r'''<!doctype html><meta charset="utf-8"><title>LSFB lexical plan review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 20px;background:#f5f7fa;color:#162030}header{position:sticky;top:0;background:#f5f7fa;padding:12px 0}article{background:white;padding:22px;border:1px solid #ccd5df;border-radius:12px;margin:16px 0}video{max-width:100%;max-height:420px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f6;padding:12px}button,input,select,textarea{font:inherit;padding:8px}textarea{width:95%;min-height:65px}.flag{color:#8b3618}.ok{color:#176b3a}label{display:block;margin:8px 0}small{display:block;color:#526171}</style>
<header><h1>LSFB lexical plan review</h1><p>Task: French + preceding dialogue → lexical operations. Nonlexical events remain attached. Every source row starts unreviewed.</p><label>Reviewer <input id="reviewer" placeholder="Name / reviewer ID"></label><button id="download">Download review decisions</button><span id="count"></span></header><main id="rows"></main>
<script type="application/json" id="payload">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent), decisions={};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 for(const [i,r] of data.entries()){
 const a=document.createElement('article');
 a.innerHTML=`<h2>${esc(r.french_text)}</h2><small>${esc(r.id)} · ${esc(r.split)}</small><p class="flag">Review flags: ${esc(r.review_flags.join(', '))}<br>Technical exclusions: ${esc(r.technical_exclusions.join(', '))}</p>${r.clip?`<video controls preload="none" src="${esc(r.clip)}"></video><small>Clip begins ${r.clip_start_ms/1000}s into the original recording. Target interval: ${r.interval_ms.map(x=>x/1000).join('–')}s.</small>`:'<p class="flag">No video excerpt available. Review cannot establish alignment.</p>'}<details open><summary>Completed preceding turns (French only)</summary><pre>${esc(JSON.stringify(r.context,null,2))}</pre></details><details open><summary>Exact model input (after context budgeting)</summary><pre>${esc(r.input_text)}</pre></details><h3>Proposed lexical operations</h3><textarea class="target">${esc(JSON.stringify(r.target_operations))}</textarea><details><summary>Original annotations, boundary events and hand tiers</summary><pre>${esc(JSON.stringify({aligned:r.aligned_events,boundary:r.boundary_excluded_events,hands:r.hand_tiers},null,2))}</pre></details><label><input type="checkbox" class="alignment"> Source video, signer and temporal alignment checked</label><label><input type="checkbox" class="meaning"> French meaning and lexical target checked by an LSFB reviewer</label><label><input type="checkbox" class="context"> Included preceding context is sufficient</label><label><input type="checkbox" class="representation"> Fingerspelling, repetition and hand representation checked</label><textarea class="notes" placeholder="Review notes, corrections, unresolved problems"></textarea><p><select class="decision"><option value="">Pending</option><option value="accept">Accept</option><option value="reject">Reject</option><option value="needs_context">Needs more context</option><option value="needs_alignment">Needs alignment repair</option></select> <button class="save">Record decision</button><span class="status"></span></p>`;
 if(r.experimental_lexical_target){
  const note=document.createElement('p');note.className='flag';
  note.textContent='Experimental training projection: '+JSON.stringify(r.experimental_lexical_target)+'. This omits nonlexical events; the source operations below remain separately reviewable.';
  a.querySelector('h3').before(note);
 }
 const video=a.querySelector('video');
 if(video){
  const panel=document.createElement('details');panel.open=true;
  panel.innerHTML='<summary>Synchronized source annotation timeline</summary><p class="clock"></p><p>Click an event to seek. Highlighting follows source timestamps; it does not certify semantic alignment.</p><div class="events"></div>';
  video.after(panel);
  const eventButtons=[];
  for(const [tier,events] of Object.entries(r.hand_tiers||{})){
   const lane=document.createElement('p'),label=document.createElement('strong');label.textContent=tier+' ';lane.append(label);
   for(const e of events){
    const button=document.createElement('button');button.type='button';
    button.textContent=`${e.gloss||e.lemma||e.sign_type||'event'} [${(e.start_ms/1000).toFixed(2)}–${(e.end_ms/1000).toFixed(2)}]`;
    button.style.margin='3px';button.title=e.sign_type||'unspecified';
    button.onclick=()=>{video.currentTime=Math.max(0,(e.start_ms-r.clip_start_ms)/1000);};
    lane.append(button);eventButtons.push([button,e]);
   }
   panel.querySelector('.events').append(lane);
  }
  const update=()=>{
   const ms=video.currentTime*1000+r.clip_start_ms,inside=ms>=r.interval_ms[0]&&ms<=r.interval_ms[1];
   panel.querySelector('.clock').textContent=`Source ${(ms/1000).toFixed(2)}s · ${inside?'inside':'outside'} target translation interval`;
   for(const [button,e] of eventButtons){const active=ms>=e.start_ms&&ms<e.end_ms;button.style.background=active?'#b5efd2':'';button.setAttribute('aria-pressed',String(active));}
  };
  video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);update();
 }
 a.querySelector('.save').onclick=()=>{
  try{
   const reviewer=document.getElementById('reviewer').value.trim(), decision=a.querySelector('.decision').value;
   if(!reviewer||!decision)throw Error('Enter reviewer and decision.');
   const checks=Object.fromEntries(['alignment','meaning','context','representation'].map(k=>[k,a.querySelector('.'+k).checked]));
   if(decision==='accept'&&!r.clip)throw Error('Acceptance requires an available source video excerpt.');
   if(decision==='accept'&&!Object.values(checks).every(Boolean))throw Error('Acceptance requires all four checks.');
   const corrected_target=JSON.parse(a.querySelector('.target').value);
   if(!Array.isArray(corrected_target))throw Error('Target must be a JSON array.');
   decisions[r.id]={id:r.id,content_sha256:r.content_sha256,task:r.task,decision,reviewer,checks,corrected_target,notes:a.querySelector('.notes').value.trim(),reviewed_at:new Date().toISOString()};
   a.querySelector('.status').textContent=' Recorded'; document.getElementById('count').textContent=` ${Object.keys(decisions).length} decisions`;
  }catch(e){a.querySelector('.status').textContent=e.message;}
 };
 document.getElementById('rows').append(a);
}
document.getElementById('download').onclick=()=>{const blob=new Blob([Object.values(decisions).map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='lsfb-review-decisions.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script>'''


def build_review(root, candidates, output, limit=50):
    output.mkdir(parents=True, exist_ok=True)
    (output/'clips').mkdir(exist_ok=True)
    rows = [json.loads(l) for l in candidates.read_text().splitlines()][:limit]
    result = []
    failures = []
    with tempfile.TemporaryDirectory(prefix='lsfb-review-') as tmp:
        cached_rid = None
        for i, row in enumerate(rows):
            rid = row['recording_id']
            try:
                source = Path(tmp)/'recording.mp4'
                if rid != cached_rid:
                    recording_from_index(root, rid, source)
                    cached_rid = rid
                duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(source)], text=True))
                start = max(0, row['interval_ms'][0]/1000-8)
                end = min(duration, row['interval_ms'][1]/1000+3)
                if row['interval_ms'][1]/1000 > duration+.1 or end <= start:
                    raise ValueError('Source annotation outside video duration')
                clip = output/'clips'/f'{i:03d}.mp4'
                subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(start), '-i', str(source),
                                '-t', str(end-start), '-an', '-vf', 'scale=640:-2', '-r', '25', '-c:v', 'libx264',
                                '-preset', 'veryfast', '-crf', '26', '-movflags', '+faststart', str(clip)], check=True)
                result.append({**row, 'clip': f'clips/{clip.name}', 'clip_start_ms': round(start*1000)})
            except (ValueError, KeyError, subprocess.CalledProcessError) as error:
                failures.append({'id': row['id'], 'error': str(error)})
                result.append({**row, 'clip': None})
            print(f'Review excerpt {i+1}/{len(rows)}', flush=True)
    payload = json.dumps(result, ensure_ascii=False).replace('<', '\\u003c')
    (output/'index.html').write_text(PAGE.replace('__DATA__', payload))
    (output/'video_manifest.json').write_text(json.dumps({'rows': len(rows), 'clips': len(rows)-len(failures), 'failures': failures}, ensure_ascii=False, indent=2))
    print(json.dumps({'review_html': str(output/'index.html'), 'clips': len(rows)-len(failures), 'failures': failures}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=50)
    a = p.parse_args()
    build_review(a.root, a.candidates, a.output, a.limit)
