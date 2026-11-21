"""Serve the built frontend, showcase site, and API as a public or authenticated research preview."""
import base64
import binascii
import hmac
import os
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('SIGNACULTURE_STORAGE', str(ROOT / '.runtime'))
os.environ.setdefault('SIGNACULTURE_DB_PATH', str(ROOT / '.runtime/plans.sqlite3'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if not os.getenv('SIGNACULTURE_FFMPEG') and shutil.which('ffmpeg'):
    os.environ['SIGNACULTURE_FFMPEG'] = shutil.which('ffmpeg')

from backend.app.main import app as api


class RangeSafeGZip:
    """Keep media byte ranges in the original representation."""
    def __init__(self, app, **kwargs):
        self.app = app
        self.compressed = GZipMiddleware(app, **kwargs)

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and any(k.lower() == b'range' for k, _ in scope.get('headers', [])):
            await self.app(scope, receive, send)
        else:
            await self.compressed(scope, receive, send)


class SPAFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code == 404 and not Path(path).suffix:
                return await super().get_response('index.html', scope)
            raise


def create_app(frontend=None, username=None, password=None):
    # Public access is open by default: anyone with the link can access it without credentials.
    # Auth is only enabled if explicitly configured with SIGNACULTURE_REQUIRE_AUTH=1,
    # or if both username and password are provided as explicit arguments.
    require_auth = False
    if username is not None and password is not None:
        require_auth = True
    elif os.environ.get('SIGNACULTURE_REQUIRE_AUTH', '').lower() in ('1', 'true', 'yes'):
        username = os.environ.get('SIGNACULTURE_PREVIEW_USER')
        password = os.environ.get('SIGNACULTURE_PREVIEW_PASSWORD')
        if not username or not password:
            raise RuntimeError('Preview username/password must be configured when SIGNACULTURE_REQUIRE_AUTH is set')
        require_auth = True

    frontend_dir = Path(frontend) if frontend else ROOT / 'frontend/dist'
    if not (frontend_dir / 'index.html').is_file():
        raise RuntimeError(f'Build the frontend before starting the preview server: {frontend_dir}')

    app = FastAPI(title='SignaCulture AI', docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RangeSafeGZip, minimum_size=1000, compresslevel=5)

    @app.middleware('http')
    async def authenticate(request, call_next):
        if require_auth and request.url.path != '/healthz':
            valid = False
            authorization = request.headers.get('authorization', '')
            if authorization.lower().startswith('basic '):
                try:
                    user, secret = base64.b64decode(authorization.split(' ', 1)[1], validate=True).decode('utf-8').split(':', 1)
                    user_ok = hmac.compare_digest(user.encode(), (username or '').encode())
                    secret_ok = hmac.compare_digest(secret.encode(), (password or '').encode())
                    valid = user_ok and secret_ok
                except (ValueError, UnicodeDecodeError, binascii.Error):
                    pass
            if not valid:
                return JSONResponse(
                    {'detail': 'Research preview sign-in required'},
                    status_code=401,
                    headers={'WWW-Authenticate': 'Basic realm="SignaCulture preview", charset="UTF-8"', 'Cache-Control': 'no-store'}
                )
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        path = request.url.path
        if path.startswith('/studio/assets/'):
            # Vite build output: file names carry a content hash, so they never change in place.
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        elif path.startswith('/assets/') or 'cache-control' in response.headers:
            # Landing media, and API files that set their own policy (the avatar: no-cache + ETag, so a
            # repeat visit revalidates with a 304 instead of downloading 12 MB again).
            response.headers.setdefault('Cache-Control', 'public, max-age=3600')
        else:
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/healthz')
    def health():
        return {'status': 'ok', 'service': 'signaculture-preview'}

    @app.api_route('/studio', methods=['GET', 'HEAD'])
    def studio():
        return RedirectResponse('/studio/', status_code=307)

    app.mount('/api', api)
    app.mount('/studio', SPAFiles(directory=frontend_dir, html=True), name='studio')

    webcam_dir = ROOT / 'webcam/dist'
    if (webcam_dir / 'index.html').is_file():
        app.mount('/webcam', StaticFiles(directory=webcam_dir, html=True), name='webcam')

    diagrams_dir = ROOT / 'docs/diagrams'
    if diagrams_dir.is_dir():
        app.mount('/diagrams', StaticFiles(directory=diagrams_dir), name='diagrams')

    site_dir = ROOT / 'site'
    if (site_dir / 'index.html').is_file():
        app.mount('/', StaticFiles(directory=site_dir, html=True), name='site')
    else:
        app.mount('/', SPAFiles(directory=frontend_dir, html=True), name='frontend')

    return app
