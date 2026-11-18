#models.py
from datetime import datetime
from .extensions import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ScrapeJob(db.Model):
    __tablename__ = "scrape_jobs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    source = db.Column(db.String(1024))  # url or filename
    status = db.Column(db.String(500), default="pending")  # pending, running, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime, nullable=True)
    meta = db.Column(db.JSON, nullable=True)

    user = db.relationship('User', backref='jobs')

class EmailRecord(db.Model):
    __tablename__ = "emails"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(320), index=True, nullable=False)
    source = db.Column(db.String(1024), nullable=True)  # url or filename
    job_id = db.Column(db.Integer, db.ForeignKey('scrape_jobs.id'), nullable=True)
    name = db.Column(db.String(255), nullable=True)
    context_snippet = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    job = db.relationship('ScrapeJob', backref='emails')
