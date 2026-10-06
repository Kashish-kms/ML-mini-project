import os
import sys
import html
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.app import app

    _orig_wsgi = app.wsgi_app

    class VercelPathFix:
        """Strip /api/index prefix that Vercel prepends to PATH_INFO."""

        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app

        def __call__(self, environ, start_response):
            path = environ.get("PATH_INFO", "/")
            for prefix in ("/api/index.py", "/api/index"):
                if path == prefix or path.startswith(prefix + "/"):
                    environ["PATH_INFO"] = path[len(prefix):] or "/"
                    break
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
