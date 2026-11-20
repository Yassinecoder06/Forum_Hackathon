const $ = id => document.getElementById(id)
const video = $('video')
const canvas = $('overlay')
const ctx = canvas.getContext('2d')
const bodyEdges = [[0, 7], [0, 8], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
const handEdges = Array.from({length: 5}, (_, i) => {
  const root = i * 4 + 1
  return [[0, root], [root, root + 1], [root + 1, root + 2], [root + 2, root + 3]]
}).flat()

let selected, allWindows = [], visible = [], activeIndex = -1, activeWindow, pose, source
let request = 0, segmentRequest = 0, animation
const stamp = ms => {
  const s = Math.max(0, Math.floor(ms / 1000))
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map(n => String(n).padStart(2, '0')).join(':')
}
const number = value => new Intl.NumberFormat('fr-FR').format(value)
const ratio = value => `${Math.round(value * 100)} %`
const fetchJSON = async path => {
  const response = await fetch(path)
  if (!response.ok) throw new Error(`Données indisponibles (${response.status})`)
  return response.json()
}
const showError = error => { $('status').textContent = error.message }

function drawDots(frame, offset, count, edges, color, project, radius) {
  ctx.fillStyle = ctx.strokeStyle = color
  ctx.beginPath()
  for (const [a,b] of edges) {
    if (frame[offset+a] && frame[offset+b]) {
      ctx.moveTo(...project(frame[offset+a]))
      ctx.lineTo(...project(frame[offset+b]))
    }
  }
  ctx.stroke()
  ctx.beginPath()
  for (let i = offset; i < offset + count; i++) {
    if (!frame[i]) continue
    const [x,y] = project(frame[i])
    ctx.moveTo(x+radius, y)
    ctx.arc(x, y, radius, 0, Math.PI * 2)
  }
  ctx.fill()
}

function nearest(timestamp) {
  const times = pose?.times ?? []
  let low = 0, high = times.length
  while (low < high) {
    const mid = (low + high) >> 1
    if (times[mid] < timestamp) low = mid + 1
    else high = mid
  }
  const sample = [low - 1, low].filter(i => i >= 0 && i < times.length)
    .reduce((best, i) => best < 0 || Math.abs(times[i] - timestamp) < Math.abs(times[best] - timestamp) ? i : best, -1)
  return sample >= 0 && Math.abs(times[sample] - timestamp) <= 100 ? sample : -1
}

function draw(time = video.currentTime) {
  $('seek').value = time
  $('clock').value = stamp(time * 1000)
  if (!source || video.seeking || video.readyState < 2) return
  const [x0,y0,x1,y1] = source.crop_xyxy_pixels
  const crop = $('crop').checked
  const w = x1 - x0, h = y1 - y0
  const width = crop ? w : video.videoWidth, height = crop ? h : video.videoHeight
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
    canvas.style.aspectRatio = `${width}/${height}`
  }
  if (crop) ctx.drawImage(video, x0,y0,w,h, 0,0,width,height)
  else ctx.drawImage(video,0,0,width,height)
  canvas.parentElement.classList.add('canvas-ready')
  const sample = nearest(time * 1000)
  canvas.dataset.sampleTimeMs = sample < 0 ? '' : pose.times[sample]
  const flags = sample >= 0 ? pose.has[sample] : []
  $('coverage').textContent = sample < 0 ? 'Pose indisponible à cet instant' : `${Number(flags[1]) + Number(flags[2])}/2 mains détectées`
  if (!$('landmarks').checked || sample < 0) return
  const project = ([x,y]) => [x * w + (crop ? 0 : x0), y * h + (crop ? 0 : y0)]
  ctx.lineWidth = Math.max(1, w / 210)
  const frame = pose.points[sample]
  drawDots(frame, 0, 23, bodyEdges, '#4ee5b2', project, 2.5)
  drawDots(frame, 23, 21, handEdges, '#ffca61', project, 2)
  drawDots(frame, 44, 21, handEdges, '#87c9ff', project, 2)
  drawDots(frame, 65, 92, [], '#ffb0d5', project, 1.2)
}

function renderList() {
  const search = $('query').value.trim().toLocaleLowerCase('fr')
  visible = allWindows.flatMap((item,index) =>
    (!$('candidates').checked || item.candidate) && item.french_text.toLocaleLowerCase('fr').includes(search) ? [index] : [])
  $('list').replaceChildren(...visible.map(index => {
    const row = allWindows[index]
    const li = document.createElement('li')
    const button = document.createElement('button')
    button.type = 'button'
    button.dataset.index = index
    button.setAttribute('aria-current', String(index === activeIndex))
    const info = document.createElement('span')
    info.textContent = `${stamp(row.start_ms)} · suivi des mains ${ratio(row.any_hand_coverage)}`
    button.append(info, document.createTextNode(row.french_text))
    button.onclick = () => { void selectWindow(index).catch(showError) }
    li.append(button)
    return li
  }))
  $('next').disabled = !visible.length || !activeWindow
}

async function selectWindow(index) {
  const version = ++segmentRequest
  video.pause()
  canvas.parentElement.classList.remove('canvas-ready')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  activeIndex = index
  activeWindow = allWindows[index]
  pose = null
  delete canvas.dataset.sampleTimeMs
  $('subtitle').textContent = activeWindow.french_text
  $('play').disabled = $('seek').disabled = true
  $('status').textContent = ''
  $('seek').min = activeWindow.start_ms / 1000
  $('seek').max = activeWindow.end_ms / 1000
  for (const button of $('list').querySelectorAll('button')) {
    button.setAttribute('aria-current', String(Number(button.dataset.index) === index))
  }
  const track = await fetchJSON(`/api/broadcasts/${selected}/windows/${index}`)
  if (version !== segmentRequest) return
  pose = track
  const start = activeWindow.start_ms / 1000
  if (Math.abs(video.currentTime - start) > .001) video.currentTime = start
  draw(start)
  $('play').disabled = $('seek').disabled = false
  $('next').disabled = !visible.length
}

async function openBroadcast(id) {
  const version = ++request
  segmentRequest++
  video.pause()
  selected = id
  source = activeWindow = pose = null
  delete canvas.dataset.sampleTimeMs
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  activeIndex = -1
  allWindows = []
  $('play').disabled = $('seek').disabled = $('next').disabled = true
  $('list').replaceChildren()
  $('subtitle').textContent = ''
  $('status').textContent = 'Chargement de la diffusion…'
  const result = await fetchJSON(`/api/broadcasts/${id}`)
  if (version !== request) return
  source = result.meta
  allWindows = result.windows
  $('summary').textContent = `${number(source.candidate_windows)} segments avec suivi exploitable sur ${number(allWindows.length)}`
  video.src = `/api/broadcasts/${id}/video`
  canvas.parentElement.classList.remove('canvas-ready')
  renderList()
  if (!visible.length && allWindows.length) {
    $('candidates').checked = false
    renderList()
  }
  if (visible.length) await selectWindow(visible[0])
}

function tick() {
  if (video.paused || !activeWindow) return
  if (video.currentTime >= activeWindow.end_ms / 1000) {
    if ($('loop').checked) video.currentTime = activeWindow.start_ms / 1000
    else {
      video.pause()
      draw(activeWindow.end_ms / 1000)
      return
    }
  }
  draw()
  animation = requestAnimationFrame(tick)
}
video.addEventListener('play', () => {
  cancelAnimationFrame(animation)
  $('play').textContent = 'Pause'
  animation = requestAnimationFrame(tick)
})
video.addEventListener('pause', () => {
  cancelAnimationFrame(animation)
  $('play').textContent = 'Lire le segment'
})
for (const name of ['loadeddata', 'seeked']) video.addEventListener(name, () => draw())
video.addEventListener('error', () => showError(Error('La vidéo ne peut pas être lue.')))
$('play').onclick = () => {
  if (!video.paused) return video.pause()
  if (video.currentTime < activeWindow.start_ms / 1000 || video.currentTime >= activeWindow.end_ms / 1000)
    video.currentTime = activeWindow.start_ms / 1000
  void video.play().catch(showError)
}
$('seek').oninput = e => { video.currentTime = Number(e.target.value) }
$('speed').onchange = e => { video.playbackRate = Number(e.target.value) }
$('next').onclick = () => {
  const next = visible.find(index => index > activeIndex) ?? visible[0]
  if (next !== undefined) void selectWindow(next).catch(showError)
}
$('query').oninput = renderList
$('candidates').onchange = renderList
$('crop').onchange = $('landmarks').onchange = () => draw()
$('broadcast').onchange = e => { void openBroadcast(e.target.value).catch(showError) }

try {
  const broadcasts = await fetchJSON('/api/broadcasts')
  $('broadcast').replaceChildren(...broadcasts.map(row => new Option(row.name, row.id)))
  if (broadcasts.length) await openBroadcast(broadcasts[0].id)
  else $('status').textContent = 'L’extraction est en cours.'
} catch (error) {
  showError(error)
}
