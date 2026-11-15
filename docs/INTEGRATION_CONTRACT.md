# Contrat de développement v1
Local React/TypeScript/Vite + Three.js frontend; Python FastAPI API. All times ms.
API localhost:8000, UI localhost:5173; Vite /api proxy strips /api. JSON keys snake_case.
GET /health -> {status:"ok"}; GET /glossary -> {entries: GlossEntry[]}; GET /motions/{motion_id} -> MotionAsset; GET /assets/{filename} serves only registered approved-for-local-preview assets.
POST /analyze-text and /gloss-lookup body {text:string}; response TranslationDraft.
POST /compose-animation body {text:string, audio_features?:AudioFeatures}; response AnimationPlan.
POST /analyze-audio multipart file WAV -> AudioFeatures. Deterministic real signal analysis required, emotion unavailable explicit unless real local model configured.
POST /review body {plan_id,reviewer,status:"approved"|"rejected",notes}; response ReviewRecord. Persist to SQLite, approval cannot bypass rights/animation review.
POST /render body {plan_id,purpose:"preview"|"publication"}; preview returns verified plan and assets, publication blocked until ALL approvals and rights checked. No claim video exported unless actual video file generated.
GlossEntry: {gloss_id,concept_fr,gloss,language:"LSF",variant,source_id,motion_id,aliases:string[],review_status:"pending"|"approved"|"rejected"}.
MotionAsset: {motion_id,gloss_id,path,rig_version:"signaculture-v1",fps,duration_ms,license,quality_status,rights_status:"cleared"|"pending",animation_review:"pending"|"approved"|"rejected",synthetic:boolean}.
TranslationDraft: {source_text,tokens:string[],lemmas:string[],gloss_sequence:GlossEntry[],unknowns:string[],linguistic_review:"pending",translation_validated:false}.
AnimationPlan: {plan_id,source_text,target_language:"LSF",segments:[{gloss_id,motion_id,start_ms,duration_ms,asset_url}],unknowns:string[],duration_ms,linguistic_review,animation_review,rights_status,render_allowed_for_publication:false,audio_features?:AudioFeatures}.
AudioFeatures: {duration_ms,frames:[{start_ms,end_ms,pitch_hz:number|null,energy:number,is_pause:boolean}],emotion_hypothesis:null,model_version:string,warnings:string[]}.
Motion JSON file: {rig_version:"signaculture-v1",fps,duration_ms,joints:string[],frames:[{time_ms,joints:Record<string,[number,number,number]>}]}. Coordinates meters, Y up, X right, Z forward; joints head,neck,hips,left_shoulder,left_elbow,left_wrist,right_shoulder,right_elbow,right_wrist plus fingers where available. Interpolate positions. Synthetic demo must be prominently labeled nonvalidated technical fixture. Never invent approved LSF.
Data paths relative repo root; data/glossary.json is array GlossEntry; data/motions_manifest.json array MotionAsset; data/approved_motion_assets/*.json assets. 5-10 controlled concepts allowed, fixtures synthetic and pending. Backend tests must use same schema. No automatic downloads of restricted videos or large model weights.
File ownership: architect backend/app/schemas/**, data/**, experiments/**, docs/architecture.md. Backend all backend except schemas. Frontend frontend/**. QA root config, .github/**, scripts/**, docs except architecture and this contract, README.md. Lead owns contract + orchestration. Commit own work only. Do not merge/rebase others or spawn agents. Read plan as requirements; report actual limitations.
