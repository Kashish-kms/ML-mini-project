import os
import sys
import html
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.app import app

    class VercelPathMiddleware:
        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app

        def __call__(self, environ, start_response):
            matched_path = environ.get("HTTP_X_MATCHED_PATH")
            if matched_path:
                environ["PATH_INFO"] = matched_path
            else:
                path = environ.get("PATH_INFO", "")
                for prefix in ("/api/index.py", "/api/index", "/api"):
                    if path.startswith(prefix):
                        new_path = path[len(prefix):]
                        environ["PATH_INFO"] = new_path if new_path.startswith("/") else ("/" + new_path if new_path else "/")
                        break
            return self.wsgi_app(environ, start_response)

    app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
except Exception:
    err = traceback.format_exc()
    from flask import Flask

    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def show_error(path):
        return "<pre>" + html.escape(err) + "</pre>", 500
