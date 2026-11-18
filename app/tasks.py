from celery import Celery
from flask import current_app
from .models import ScrapeJob, EmailRecord
from .extensions import db
from .scraper import scrape_url_text, extract_emails_from_text, parse_uploaded_file
from datetime import datetime

celery = Celery(__name__)

def make_celery(app):
    celery.conf.broker_url = app.config['CELERY_BROKER_URL']
    celery.conf.result_backend = app.config['CELERY_RESULT_BACKEND']
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


@celery.task(bind=True)
def scrape_url_task(self, job_id, url, subject=None):
    # import create_app here to avoid circular import
    from . import create_app

    app = create_app()
    with app.app_context():
        current_app.logger.info("Scraping subject: %s", subject)
        job = ScrapeJob.query.get(job_id)
        if not job:
            return {"error": "job not found"}

        try:
            job.status = "running"
            db.session.commit()

            final_url, text = scrape_url_text(url)
            emails = extract_emails_from_text(text)

            for e in emails:
                rec = EmailRecord(email=e, source=final_url, job=job)
                db.session.add(rec)

            job.status = "completed"
            job.finished_at = datetime.utcnow()
            job.meta = {"found": len(emails)}
            db.session.commit()
            return {"found": len(emails)}

        except Exception as exc:
            job.status = "failed"
            job.meta = {"error": str(exc)}
            db.session.commit()
            raise


@celery.task(bind=True)
def scrape_upload_task(self, job_id, file_bytes, filename):
    # import create_app here to avoid circular import
    from . import create_app

    app = create_app()
    with app.app_context():
        job = ScrapeJob.query.get(job_id)
        if not job:
            return {"error": "job not found"}

        try:
            job.status = "running"
            db.session.commit()

            class FS:
                def __init__(self, b, filename):
                    self._b = b
                    self.filename = filename
                def read(self):
                    return self._b
                def seek(self, pos):
                    pass

            fs = FS(file_bytes, filename)
            fname, text = parse_uploaded_file(fs)
            emails = extract_emails_from_text(text)

            for e in emails:
                rec = EmailRecord(email=e, source=fname, job=job)
                db.session.add(rec)

            job.status = "completed"
            job.finished_at = datetime.utcnow()
            job.meta = {"found": len(emails)}
            db.session.commit()
            return {"found": len(emails)}

        except Exception as exc:
            job.status = "failed"
            job.meta = {"error": str(exc)}
            db.session.commit()
            raise
