# app/__init__.py
import os
from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS
from .config import Config
from .extensions import db, migrate, jwt
from .tasks import celery as celery_app

def create_app(config_object=None):
    load_dotenv()
    app = Flask(__name__)

    if config_object:
        app.config.from_object(config_object)
    else:
        app.config.from_object(Config)

    # allow override via env
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", app.config.get("SECRET_KEY"))

    # CORS: allow origin from env or default to localhost:3000 (dev)
    allowed_origin = os.getenv("CORS_ORIGIN", "http://localhost:3000")
    CORS(
        app,
        resources={r"/api/*": {"origins": allowed_origin}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )

    # init extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # import models so migrations detect them
    from .models import User, ScrapeJob, EmailRecord  # noqa: F401

    # register blueprints
    from .routes.auth import bp as auth_bp
    from .routes.scrape import bp as scrape_bp
    from .routes.emails import bp as emails_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(scrape_bp, url_prefix="/api/scrape")
    app.register_blueprint(emails_bp, url_prefix="/api/emails")

    # attach celery config
    celery_app.conf.update(app.config)

    return app

def make_celery(app):
    return celery_app
