import os
import sys
import html
import traceback
from urllib.parse import parse_qsl, urlencode

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.app import app

    _orig_wsgi = app.wsgi_app

    class VercelPathFix:
        """Normalize PATH_INFO and SCRIPT_NAME for Vercel Serverless environment."""

        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app

        def __call__(self, environ, start_response):
            path = environ.get("PATH_INFO", "/")
            environ["SCRIPT_NAME"] = ""

            # Check if an endpoint was passed via query parameter
            qs = environ.get("QUERY_STRING", "")
            if "__endpoint__=" in qs:
                params = parse_qsl(qs, keep_blank_values=True)
                endpoint = None
                remaining = []
                for k, v in params:
                    if k == "__endpoint__":
                        endpoint = v
                    else:
                        remaining.append((k, v))
                if endpoint:
                    environ["PATH_INFO"] = "/" + endpoint.lstrip("/")
                    environ["QUERY_STRING"] = urlencode(remaining)
                    path = environ["PATH_INFO"]

            # Strip serverless function prefixes if present
            for prefix in ("/api/index.py", "/api/index", "/api"):
                if path == prefix:
                    path = "/"
                    break
                elif path.startswith(prefix + "/"):
                    path = path[len(prefix):]
                    break

            environ["PATH_INFO"] = path or "/"
            return self.wsgi_app(environ, start_response)

    app.wsgi_app = VercelPathFix(_orig_wsgi)

except Exception:
    err = traceback.format_exc()
    from flask import Flask

    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def show_error(path):
        return "<pre>" + html.escape(err) + "</pre>", 500
