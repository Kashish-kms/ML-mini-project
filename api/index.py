import os
import sys
import json
import html
import traceback

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

            # Debug inspection endpoint
            if path.rstrip("/").endswith("/debug-wsgi") or environ.get("QUERY_STRING", "").find("__debug__=1") != -1:
                debug_info = {
                    "PATH_INFO": environ.get("PATH_INFO"),
                    "SCRIPT_NAME": environ.get("SCRIPT_NAME"),
                    "QUERY_STRING": environ.get("QUERY_STRING"),
                    "REQUEST_URI": environ.get("REQUEST_URI"),
                    "RAW_URI": environ.get("RAW_URI"),
                    "HTTP_X_MATCHED_PATH": environ.get("HTTP_X_MATCHED_PATH"),
                    "HTTP_X_FORWARDED_HOST": environ.get("HTTP_X_FORWARDED_HOST"),
                    "HTTP_HOST": environ.get("HTTP_HOST"),
                }
                body = json.dumps(debug_info, indent=2).encode("utf-8")
                start_response("200 OK", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(body))),
                ])
                return [body]

            # Clear SCRIPT_NAME to prevent Werkzeug routing confusion
            environ["SCRIPT_NAME"] = ""

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
