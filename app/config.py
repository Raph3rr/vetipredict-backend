# app/config.py
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(basedir, ".env"))

def build_mysql_uri():
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASS", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "email_scraper")
    engine = os.getenv("DB_ENGINE", "mysql+pymysql")
    if user and name:
        # example: mysql+pymysql://user:pass@host:3306/dbname?charset=utf8mb4
        auth = f"{user}:{password}@" if password or user else ""
        return f"{engine}://{auth}{host}:{port}/{name}?charset=utf8mb4"
    return None

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret")
    # allow explicit full URI first, otherwise build from parts
    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI") or build_mysql_uri() or "sqlite:///data.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-secret")
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))
