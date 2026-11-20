const byId = id => document.getElementById(id)
const video = byId('video')
const seek = byId('seek')
const play = byId('play')
const layers = [...document.querySelectorAll('[data-layer]')]
const faceLayer = document.querySelector('[data-layer=face]')
const hamerPreview = document.body.dataset.preview === 'hamer'
const hamerBodyModel = document.body.dataset.source === 'sapiens' ? 'Sapiens 0.3B' : 'MediaPipe'
const matignonPreview = document.body.dataset.source === 'matignon'
const modelIds = hamerPreview ? ['hamer'] : ['mediapipe', 'mmpose', 'sapiens', 'sdpose', 'hamer']
if (hamerPreview) {
  document.querySelector('h1').textContent = matignonPreview ? 'Matignon notebook 07 · Full frame' : `${hamerBodyModel} + HaMeR`
  if (matignonPreview) {
    byId('hamer').closest('section').querySelector('h3').textContent = 'MediaPipe Holistic overlay'
    byId('hamer-mesh').closest('label').hidden = true
  }
}
const panels = modelIds.map(id => {
  const canvas = byId(id)
  const note = document.createElement('p')
  note.className = 'avatar-label'
  note.textContent = matignonPreview ? 'Noa · MediaPipe landmark playback' : id === 'hamer' ? `3D hands · ${hamerBodyModel} body` : '3D avatar · driven by 2D keypoints'
  const avatar = document.createElement('div')
  avatar.className = 'avatar-preview'
  avatar.dataset.model = id
  avatar.textContent = 'Waiting for keypoints…'
  canvas.closest('section').append(note, avatar)
  return {id, canvas, avatar, context: canvas.getContext('2d'), status: byId(`${id}-status`), frames: []}
})
const avatarModule = import('/avatar.js?v=head-pose').catch(() => null)
const upperBody = ['left_shoulder','right_shoulder','right_elbow','right_wrist','left_elbow','left_wrist']
const fingers = ['thumb','index','middle','ring','pinky']

// Bridge short arm-tracking gaps without producing new hand observations.
function estimateArmPoint(frames, index, pointAt) {
  const time = frames[index].time_ms
  let previous, next
  for (let i = index - 1; i >= 0 && time - frames[i].time_ms <= 320; i--) {
    const point = pointAt(frames[i])
    if (point) {
      previous = {point, time: frames[i].time_ms}
      break
    }
  }
  for (let i = index + 1; i < frames.length && frames[i].time_ms - time <= 320; i++) {
    const point = pointAt(frames[i])
    if (point) {
      next = {point, time: frames[i].time_ms}
      break
    }
  }
  if (previous && next && next.time - previous.time <= 320) {
    const alpha = (time - previous.time) / (next.time - previous.time)
    return [0, 1].map(axis => previous.point[axis] + (next.point[axis] - previous.point[axis]) * alpha)
  }
  if (previous && time - previous.time <= 140) return previous.point
  if (next && next.time - time <= 140) return next.point
}

function projectedMotion(frames, aspect, fps) {
  const mapped = frames.flatMap((frame, index) => {
    const [left, right] = frame.body ?? []
    if (!left || !right) return []
    const distance = Math.hypot((left[0] - right[0]) * aspect, left[1] - right[1])
    if (distance < .035) return []
    const scale = .34 / distance
    const centerX = (left[0] + right[0]) / 2
    const centerY = (left[1] + right[1]) / 2
    const project = (p, depth) => [
      (p[0] - centerX) * aspect * scale,
      1.45 + (centerY - p[1]) * scale,
      depth,
    ]
    const joints = {neck: [0,1.45,.07], hips: [0,1,0], head: [0,1.65,.08]}
    upperBody.forEach((joint, i) => {
      if (frame.body[i]) joints[joint] = project(frame.body[i], joint.endsWith('wrist') ? .26 : .08)
    })
    const wristSources = {}
    for (const side of ['left', 'right']) {
      const hand = frame[side]
      const wrist = `${side}_wrist`
      const elbow = `${side}_elbow`
      if (hamerPreview) {
        if (!joints[elbow]) {
          const estimate = estimateArmPoint(frames, index, sample => sample.body?.[upperBody.indexOf(elbow)])
          if (estimate) joints[elbow] = project(estimate, .08)
        }
        if (!hand?.[0] && !joints[wrist]) {
          const estimate = estimateArmPoint(frames, index, sample => sample[side]?.[0] ?? sample.body?.[upperBody.indexOf(wrist)])
          if (estimate) joints[wrist] = project(estimate, .26)
          wristSources[side] = estimate && joints[elbow] ? 'estimated' : 'unavailable'
        } else {
          wristSources[side] = !joints[elbow] ? 'unavailable' : hand?.[0] ? 'hand' : 'body'
        }
      }
      if (!hand?.[0]) continue
      joints[`${side}_wrist`] = project(hand[0], .26)
      fingers.forEach((finger, i) => {
        for (let j = 1; j <= 4; j++) {
          const p = hand[i * 4 + j]
          if (p) {
            const depth = frame[`${side}_depth`]?.[i * 4 + j] ?? 0
            joints[`${side}_${finger}_${j}`] = project(p, .26 + depth)
          }
        }
      })
    }
    return [{time_ms: frame.time_ms, joints, wristSources}]
  })
  if (!mapped.length) return null
  return {rig_version: 'signaculture-v1', fps, duration_ms: mapped.at(-1).time_ms,
    joints: [...new Set(mapped.flatMap(frame => Object.keys(frame.joints)))], frames: mapped}
}

function drawAvatar(panel, frame, aspect) {
  const bodyFrames = panel.id === 'hamer' && !hamerPreview ? panels[0].frames : null
  if (panel.avatarFrames !== panel.frames || panel.avatarAspect !== aspect || panel.avatarBody !== bodyFrames) {
    panel.avatarFrames = panel.frames
    panel.avatarAspect = aspect
    panel.avatarBody = bodyFrames
    const frames = bodyFrames ? panel.frames.map(frame => ({
      ...frame, body: nearestFrame(bodyFrames, frame.time_ms, 50)?.body,
    })) : panel.frames
    panel.motion = projectedMotion(frames, aspect, sourceFps)
    panel.motionSamples = new Map(panel.motion?.frames.map(frame => [frame.time_ms, frame]))
  }
  if (!panel.mounting && panel.motion) {
    panel.mounting = true
    avatarModule.then(module => {
      if (!module) {
        panel.avatar.textContent = 'Avatar viewer unavailable.'
        return
      }
      panel.renderAvatar = module.mountAvatar(panel.avatar)
      draw()
    })
  }
  const sample = frame && panel.motionSamples.get(frame.time_ms)
  const time = sample?.time_ms ?? null
  panel.wristSources = sample?.wristSources
  const motion = time === null ? null : panel.motion
  panel.avatar.dataset.sampleTimeMs = time ?? ''
  const face = motion && faceLayer.checked ? frame.face_blendshapes : undefined
  if (panel.renderAvatar && (panel.renderedTime !== time || panel.renderedMotion !== motion || panel.renderedFace !== face)) {
    panel.renderAvatar(motion, time ?? 0, face, motion ? frame.head_rotation : undefined)
    panel.renderedFace = face
    panel.renderedTime = time
    panel.renderedMotion = motion
  }
}
const bodyEdges = [[0,1],[1,2],[2,3],[0,4],[4,5],[0,6],[1,7],[6,7],[6,8],[8,9],[7,10],[10,11]]
const handEdges = Array.from({length: 5}, (_, i) => {
  const root = 1 + i * 4
  return [[0,root],[root,root+1],[root+1,root+2],[root+2,root+3]]
}).flat()
const styles = {
  body: ['#6ce9bf', bodyEdges, 2.4],
  left: ['#ffe284', handEdges, 2],
  right: ['#83bbff', handEdges, 2],
  face: ['#ff9aca', [], 0.85],
}
const mediapipeBody = [11,12,14,16,13,15,23,24,25,27,26,28]
let clips = []
let selected = -1
let clipRequest
let videoSources = []
let videoLoadTimer
let pollTimer
let presentationCallback
let presentedTime
let sourceFps = 25
let playbackRequested = false

function nearestFrame(frames, time, tolerance) {
  let lo = 0
  let hi = frames.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].time_ms < time) lo = mid + 1
    else hi = mid
  }
  const frame = lo && Math.abs(frames[lo - 1].time_ms - time) < Math.abs(frames[lo].time_ms - time)
    ? frames[lo - 1] : frames[lo]
  return frame && Math.abs(frame.time_ms - time) <= tolerance ? frame : undefined
}

function drawHandMeshes(context, meshes, faces, width, height) {
  // HaMeR's released config projects a 256px crop with a focal length of 5000px.
  const focal = 5000 / 256 * Math.max(width, height)
  const triangles = meshes.flatMap(vertices => faces.map(([a, b, c]) => {
    const points = [vertices[a], vertices[b], vertices[c]]
    return {points, depth: points[0][2] + points[1][2] + points[2][2]}
  })).sort((a, b) => b.depth - a.depth)
  for (const {points: [a, b, c]} of triangles) {
    const ux = ((b[0] - .5) * b[2] - (a[0] - .5) * a[2]) * width / focal
    const uy = ((b[1] - .5) * b[2] - (a[1] - .5) * a[2]) * height / focal
    const vx = ((c[0] - .5) * c[2] - (a[0] - .5) * a[2]) * width / focal
    const vy = ((c[1] - .5) * c[2] - (a[1] - .5) * a[2]) * height / focal
    const uz = b[2] - a[2]
    const vz = c[2] - a[2]
    const nx = uy * vz - uz * vy
    const ny = uz * vx - ux * vz
    const nz = ux * vy - uy * vx
    const length = Math.hypot(nx, ny, nz)
    if (!length) continue
    const shade = 0.62 + 0.38 * Math.abs((.25 * nx - .35 * ny + .9 * nz) / length)
    context.fillStyle = `rgb(${Math.round(190 * shade)}, ${Math.round(222 * shade)}, ${Math.round(250 * shade)})`
    context.strokeStyle = context.fillStyle
    context.lineWidth = .75
    context.beginPath()
    context.moveTo(a[0] * width, a[1] * height)
    context.lineTo(b[0] * width, b[1] * height)
    context.lineTo(c[0] * width, c[1] * height)
    context.closePath()
    context.fill()
    context.stroke()
  }
}

function draw(mediaTime = presentedTime ?? video.currentTime) {
  const ready = video.readyState >= 2 && video.videoWidth && video.videoHeight
  const showMesh = byId('hamer-mesh').checked
  for (const panel of panels) {
    const {canvas, context} = panel
    if (!ready) {
      context.clearRect(0, 0, canvas.width, canvas.height)
      continue
    }
    const {videoWidth: width, videoHeight: height} = video
    const frame = nearestFrame(panel.frames, mediaTime * 1000, panel.tolerance ?? 125)
    canvas.dataset.sampleTimeMs = frame?.time_ms ?? ''
    drawAvatar(panel, frame, width / height)
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
      canvas.parentElement.style.aspectRatio = `${width} / ${height}`
    }
    context.drawImage(video, 0, 0, width, height)
    if (hamerPreview) {
      const tracked = ['left', 'right'].filter(side => frame?.[side])
      const restored = ['left', 'right'].filter(side => frame?.fallback_hands?.includes(side))
      const missing = ['left', 'right'].filter(side => !tracked.includes(side)).map(side => {
        const source = panel.wristSources?.[side]
        const label = source === 'body' ? 'body wrist only' : source === 'estimated' ? 'estimated wrist' : 'placement uncertain'
        return `${side}: ${label}`
      })
      byId('tracking-status').textContent = frame
        ? `Hand tracking: ${tracked.length}/2${restored.length ? ` · fallback: ${restored.join(', ')}` : ''}${missing.length ? ` · ${missing.join(' · ')}` : ''} · Face: ${Object.keys(frame.face_blendshapes ?? {}).length ? 'tracked' : frame.face?.some(Boolean) ? 'landmarks only' : 'unavailable'} · Head pose: ${frame.head_rotation ? 'estimated' : 'unavailable'}`
        : 'No extraction sample at this position.'
    }
    if (!frame) continue
    const meshes = []
    for (const {checked, dataset: {layer}} of layers) {
      if (!checked || !frame[layer]) continue
      const mesh = frame[`${layer}_mesh`]
      if (showMesh && panel.meshFaces && mesh) {
        meshes.push(mesh)
        continue
      }
      const [color, edges, radius] = styles[layer]
      const points = frame[layer].map(point => point && [point[0] * width, point[1] * height])
      context.strokeStyle = context.fillStyle = color
      context.lineWidth = Math.max(1, width / 320)
      context.beginPath()
      for (const [from, to] of edges) {
        if (!points[from] || !points[to]) continue
        context.moveTo(...points[from])
        context.lineTo(...points[to])
      }
      context.stroke()
      context.beginPath()
      const dot = Math.max(.8, radius * width / 480)
      for (const point of points) {
        if (!point) continue
        context.moveTo(point[0] + dot, point[1])
        context.arc(...point, dot, 0, 2 * Math.PI)
      }
      context.fill()
    }
    if (meshes.length) drawHandMeshes(context, meshes, panel.meshFaces, width, height)
  }
}

function updateTime() {
  const duration = Number.isFinite(video.duration) ? video.duration : 0
  seek.max = duration
  seek.value = video.currentTime
  for (const control of [seek, play, byId('step-back'), byId('step-forward')]) control.disabled = duration <= 0
  play.textContent = video.paused ? 'Play' : 'Pause'
  byId('time').textContent = `${video.currentTime.toFixed(2)} / ${duration.toFixed(2)} s`
}

function loadNextVideo() {
  clearTimeout(videoLoadTimer)
  const url = videoSources.shift()
  if (!url) {
    video.removeAttribute('src')
    video.load()
    byId('video-status').textContent = 'Source video unavailable. Open the original Dicta-Sign entry to check the recording.'
    return
  }
  byId('video-status').textContent = 'Loading source video…'
  video.src = url
  video.load()
  videoLoadTimer = setTimeout(loadNextVideo, 7000)
}
video.addEventListener('loadeddata', () => {
  clearTimeout(videoLoadTimer)
  byId('video-status').textContent = ''
  if (playbackRequested) void playVideo()
})
video.addEventListener('error', () => {
  if (video.error && video.hasAttribute('src')) loadNextVideo()
})

const hasVideoFrameCallback = typeof video.requestVideoFrameCallback === 'function'
function schedulePresentation() {
  if (presentationCallback !== undefined || video.paused || video.ended) return
  const present = (_now, metadata) => {
    presentationCallback = undefined
    presentedTime = metadata?.mediaTime ?? video.currentTime
    draw(presentedTime)
    schedulePresentation()
  }
  presentationCallback = hasVideoFrameCallback
    ? video.requestVideoFrameCallback(present)
    : requestAnimationFrame(present)
}
video.addEventListener('play', schedulePresentation)
video.addEventListener('ended', () => { playbackRequested = false })
for (const event of ['loadeddata', 'loadedmetadata', 'seeked', 'timeupdate', 'play', 'pause', 'ended', 'emptied']) {
  video.addEventListener(event, () => {
    updateTime()
    if ((video.paused || video.ended || event === 'emptied') && presentationCallback !== undefined) {
      if (hasVideoFrameCallback) video.cancelVideoFrameCallback(presentationCallback)
      else cancelAnimationFrame(presentationCallback)
      presentationCallback = undefined
    }
    if (event === 'seeked' || event === 'loadeddata' || event === 'emptied') presentedTime = undefined
    if (video.paused || event === 'seeked' || event === 'loadeddata') draw()
  })
}
for (const layer of [...layers, byId('hamer-mesh')]) layer.addEventListener('change', () => draw())
seek.addEventListener('input', () => { video.currentTime = Number(seek.value) })
function step(direction) {
  playbackRequested = false
  video.pause()
  video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + direction / sourceFps))
}
byId('step-back').onclick = () => step(-1)
byId('step-forward').onclick = () => step(1)
async function playVideo() {
  const source = video.src
  if (video.ended) video.currentTime = 0
  try { await video.play() }
  catch (error) {
    if (error.name !== 'AbortError' && video.src === source && !video.error) {
      playbackRequested = false
      byId('video-status').textContent = 'Playback could not start. Press Play to retry.'
    }
  }
}
play.onclick = () => {
  playbackRequested = video.paused
  if (playbackRequested) void playVideo()
  else video.pause()
}

function renderList() {
  const query = byId('search').value.trim().toLocaleLowerCase('fr')
  const buttons = []
  clips.forEach((clip, index) => {
    if (!`${clip.concept} ${clip.gloss}`.toLocaleLowerCase('fr').includes(query)) return
    const button = document.createElement('button')
    button.type = 'button'
    button.dataset.index = index
    button.setAttribute('aria-current', String(selected === index))
    const number = document.createElement('small')
    number.textContent = String(index + 1).padStart(2, '0')
    button.append(number, document.createTextNode(clip.concept || clip.gloss || 'Untitled clip'))
    button.onclick = () => select(index)
    buttons.push(button)
  })
  byId('list').replaceChildren(...buttons)
  if (!buttons.length) byId('list').textContent = 'No matching clips.'
}

function fromMediaPipe(frame) {
  const convert = (points, visibility = false) => points?.map(point =>
    point && Number.isFinite(point.x) && Number.isFinite(point.y) && (!visibility || (point.visibility ?? 1) >= .5)
      ? [point.x, point.y] : null)
  return {
    time_ms: frame.time_ms,
    body: convert(mediapipeBody.map(index => frame.PLM?.[index]), true),
    left: convert(frame.LLM), right: convert(frame.RLM), face: convert(frame.FLM),
  }
}

async function loadPanel(panel, clip, signal) {
  const baseline = panel.id === 'mediapipe'
  try {
    const url = `/api/clips/${clip.id}/${baseline ? 'landmarks' : `pose/${panel.id}`}`
    const response = await fetch(url, {signal, cache: 'no-store'})
    if (!response.ok) throw Error(response.status)
    const data = await response.json()
    if (signal.aborted) return false
    if (data.status === 'pending') {
      panel.status.textContent = 'Extraction not yet available for this clip.'
      return true
    }
    if (data.status === 'blocked') {
      panel.status.textContent = data.reason
      panel.avatar.textContent = 'Avatar unavailable until extraction runs.'
      return false
    }
    panel.frames = baseline ? data.map(fromMediaPipe) : data.frames
    panel.meshFaces = data.mesh_faces
    if (!baseline) {
      sourceFps = data.source_fps || sourceFps
      panel.tolerance = (data.sample_interval_ms || 180) / 2 + 40
    }
    panel.status.textContent = baseline
      ? `${panel.frames.length} frames · cached Holistic + hand fallback`
      : `${panel.frames.length} frames · ${data.checkpoint} · ${data.device}${data.precision === 'bf16' ? ' BF16' : ''}`
    if (data.error) panel.status.textContent = data.error
    draw()
  } catch (error) {
    if (!signal.aborted) panel.status.textContent = 'Could not load the extracted keypoints.'
  }
  return false
}

async function select(index) {
  const clip = clips[index]
  if (!clip) return
  selected = index
  clipRequest?.abort()
  clearTimeout(pollTimer)
  clipRequest = new AbortController()
  const {signal} = clipRequest
  sourceFps = 25
  for (const panel of panels) {
    panel.frames = []
    panel.motion = null
    panel.renderedMotion = undefined
    panel.renderedTime = undefined
    panel.renderAvatar?.(null, 0)
    panel.status.textContent = 'Loading keypoints…'
  }
  byId('concept').textContent = clip.concept || clip.gloss || 'Untitled clip'
  byId('gloss').textContent = clip.gloss ? `Portal gloss · ${clip.gloss}` : ''
  const source = byId('source')
  source.href = clip.source
  source.hidden = false
  byId('prev').disabled = index === 0
  byId('next').disabled = index === clips.length - 1
  const list = byId('list')
  list.querySelector('[aria-current=true]')?.setAttribute('aria-current', 'false')
  const active = list.querySelector(`[data-index="${index}"]`)
  active?.setAttribute('aria-current', 'true')
  active?.scrollIntoView({block: 'nearest', inline: 'nearest'})
  const mediaUrl = `/api/clips/${clip.id}`
  videoSources = clip.sourceVideo
    ? [clip.sourceVideo]
    : [clip.hasVideo && `${mediaUrl}/video`, clip.hasWebm && `${mediaUrl}/video.webm`].filter(Boolean)
  loadNextVideo()
  draw()

  async function poll(activePanels) {
    const pending = await Promise.all(activePanels.map(panel => loadPanel(panel, clip, signal)))
    const remaining = activePanels.filter((_, i) => pending[i])
    if (!signal.aborted && remaining.length) pollTimer = setTimeout(() => poll(remaining), 6000)
  }
  void poll(panels)
}

byId('status').textContent = matignonPreview
  ? 'Notebook 07 uses MediaPipe Holistic for body, hands and face. Full-frame input. Face landmarks appear on the video overlay.'
  : hamerPreview
  ? `${hamerBodyModel} tracks the body and locates the hands. HaMeR reconstructs the hands. The overlay and avatar share the same extraction timestamp.`
  : 'All overlays use the same video time. The models use different landmark layouts and confidence scales.'
byId('search').oninput = renderList
byId('prev').onclick = () => select(selected - 1)
byId('next').onclick = () => select(selected + 1)
try {
  const response = await fetch('/api/clips')
  if (!response.ok) throw Error(response.status)
  clips = await response.json()
  byId('summary').textContent = `France-LSF · ${clips.length} clips`
  renderList()
  if (clips.length) select(0)
} catch {
  byId('summary').textContent = 'Could not load the local inventory.'
}
