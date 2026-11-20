"""Serve the prepared local corpus viewer with HTTP byte-range support."""
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


app = Starlette(routes=[Mount("/", app=StaticFiles(directory=".runtime/lsfb-browser", html=True))])
