"""Bounded local audio decoding, acoustic features and optional real inference."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import wave
import numpy as np
from .capture_library import ROOT,runtime_python,storage

def analyse(raw, filename, include_emotion=False, include_transcription=False):
    with tempfile.TemporaryDirectory(prefix='signaculture-audio-') as temp:
        temp=Path(temp);source=temp/('input'+Path(filename).suffix.lower());source.write_bytes(raw)
        wav=temp/'voice.wav'
        decoded=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(source),'-vn','-t','61','-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav)],capture_output=True,timeout=30)
        if decoded.returncode: raise ValueError('Fichier audio illisible ou vidéo sans piste audio.')
        with wave.open(str(wav)) as stream:
            rate=stream.getframerate();n=stream.getnframes()
            if not 0<n/rate<=60: raise ValueError('Audio limité à 60 secondes.')
            x=np.frombuffer(stream.readframes(n),dtype='<i2').astype(float)/32768
        frames=[];step=640
        for start in range(0,len(x),step):
            chunk=x[start:start+step];energy=float(np.sqrt(np.mean(chunk**2)));pitch=None
            if energy>=.01 and len(chunk)>=320:
                y=chunk-chunk.mean();size=1 << (2*len(y)-1).bit_length()
                spectrum=np.fft.rfft(y,size);corr=np.fft.irfft(spectrum*spectrum.conj(),size)[:len(y)]
                lo,hi=32,min(267,len(corr)-2)
                lag=lo+int(np.argmax(corr[lo:hi+1]))
                if corr[0]>0 and corr[lag]/corr[0]>.4:
                    denom=corr[lag-1]-2*corr[lag]+corr[lag+1]
                    offset=.5*(corr[lag-1]-corr[lag+1])/denom if abs(denom)>1e-12 else 0
                    pitch=float(rate/(lag+max(-.5,min(.5,offset))))
            frames.append({'start_ms':round(start/rate*1000),'end_ms':max(1,round((start+len(chunk))/rate*1000)),
                'pitch_hz':pitch,'energy':energy,'is_pause':energy<.01})
        emotion=None;transcription=None;warnings=['La hauteur est une estimation acoustique ; une voix bruitée peut être mal suivie.']
        version='fft-autocorrelation-rms-v2'
        if include_emotion:
            model=ROOT/'.models/emotion2vec-seed'
            if not (model/'model.pt').is_file() or not runtime_python('audio').is_file():
                warnings.append('Le modèle émotionnel local est absent ; les mesures acoustiques restent disponibles.')
            elif all(frame['is_pause'] for frame in frames):
                warnings.append('Signal trop faible pour estimer une émotion.')
            else:
                output=temp/'emotion.json'
                try:
                    result=subprocess.run([str(runtime_python('audio')),str(ROOT/'backend/app/workers/emotion.py'),str(wav),str(model),str(output)],capture_output=True,text=True,timeout=120,cwd=ROOT)
                except subprocess.TimeoutExpired:
                    warnings.append('Le modèle émotionnel a dépassé le délai ; les mesures acoustiques restent disponibles.')
                else:
                    if result.returncode:
                        warnings.append('Le modèle émotionnel a échoué ; les mesures acoustiques restent disponibles.')
                        log=storage();(log/'emotion-error.log').write_text(result.stderr[-10000:])
                    else:
                        emotion=json.loads(output.read_text())['scores'];version+=' + emotion2vec_plus_seed'
                        warnings.append('Les catégories émotionnelles sont des hypothèses du modèle, pas une mesure de l’état intérieur.')
        else: warnings.append('Analyse émotionnelle non demandée.')
        duration_ms=max(1,round(n/rate*1000))
        if include_transcription:
            model=ROOT/'.models/faster-whisper-small'
            if not (model/'model.bin').is_file() or not runtime_python('audio').is_file():
                warnings.append('Le modèle de transcription locale est absent ; le texte français doit être saisi manuellement.')
            else:
                output=temp/'transcription.json'
                result=subprocess.run([str(runtime_python('audio')),str(ROOT/'backend/app/workers/transcribe.py'),str(wav),str(model),str(output)],capture_output=True,text=True,timeout=240,cwd=ROOT)
                if result.returncode:
                    warnings.append('La transcription locale a échoué ; vérifiez le texte avant de composer.')
                    log=storage();(log/'asr-error.log').write_text(result.stderr[-10000:])
                else:
                    transcription=json.loads(output.read_text())
                    for segment in transcription['segments']:
                        segment['start_ms']=min(segment['start_ms'],max(0,duration_ms-1))
                        segment['end_ms']=min(max(segment['start_ms']+1,segment['end_ms']),duration_ms)
                    warnings.append('La transcription est forcée en français et doit être relue avant de composer.')
                    version+=' + faster-whisper-small'
        else:
            warnings.append('Transcription non demandée.')
        return {'duration_ms':duration_ms,'frames':frames,'emotion_hypothesis':emotion,'model_version':version,'warnings':warnings,'transcription':transcription}
