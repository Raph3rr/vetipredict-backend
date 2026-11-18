# app/routes/scrape.py
import csv
import io
import traceback
from flask import Blueprint, request, jsonify, current_app, Response
from ..extensions import db
from ..models import ScrapeJob, EmailRecord, User
from ..tasks import scrape_url_task, scrape_upload_task
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request

bp = Blueprint("scrape", __name__)


def _bad_request(message, code=422):
    return jsonify({"error": message}), code


@bp.route("/debug", methods=["POST"])
@jwt_required()
def debug_payload():
    payload = request.get_json(silent=True) or {}
    current_app.logger.debug("DEBUG PAYLOAD RECEIVED: %r", payload)
    return jsonify({"payload": payload}), 200


# POST /url (handles OPTIONS preflight)
@bp.route("/url", methods=["POST", "OPTIONS"])
def scrape_url():
    if request.method == "OPTIONS":
        return "", 200  # preflight

    # Require JWT only for POST
    verify_jwt_in_request()

    try:
        payload = request.get_json(silent=True) or {}
        current_app.logger.debug("scrape_url called; headers=%s payload=%r",
                                 dict(request.headers), payload)

        url = payload.get("url")
        if not url or not isinstance(url, str) or url.strip() == "":
            return jsonify({"error": "`url` is required and must be a non-empty string."}), 400

        # parse async flag robustly (accept string or boolean)
        raw_async = payload.get("async", True)
        async_call = not (isinstance(raw_async, str) and raw_async.lower() == "false") and bool(raw_async)

        raw_subject = payload.get("subject", None)
        subject = str(raw_subject) if raw_subject is not None else f"Email Scrape Job: {url}"

        user_id = get_jwt_identity()
        job = ScrapeJob(user_id=user_id, source=url, status="pending")
        db.session.add(job)
        db.session.commit()

        current_app.logger.debug("Created job id=%s subject=%r async=%r", job.id, subject, async_call)

        if async_call:
            task = scrape_url_task.apply_async(args=[job.id, url, subject])
            return jsonify({"job_id": job.id, "task_id": task.id}), 202
        else:
            # run synchronously for debugging / tests
            result = scrape_url_task.run(job.id, url, subject)
            return jsonify({"job_id": job.id, "result": result}), 200


    except Exception as e:
        current_app.logger.error("scrape_url error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "internal_server_error", "detail": str(e)}), 500


@bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_file():
    try:
        file = request.files.get("file")
        if not file:
            current_app.logger.debug("upload_file: no file in request.files; keys=%s headers=%s",
                                     list(request.files.keys()), dict(request.headers))
            return _bad_request("file required in form-data as `file`", 400)

        async_val = request.form.get("async", "true")
        async_call = str(async_val).lower() != "false"

        user_id = get_jwt_identity()
        job = ScrapeJob(user_id=user_id, source=file.filename, status="pending")
        db.session.add(job)
        db.session.commit()

        content = file.read()
        try:
            file.seek(0)
        except Exception:
            pass

        if async_call:
            task = scrape_upload_task.apply_async(args=[job.id, content, file.filename])
            return jsonify({"job_id": job.id, "task_id": task.id}), 202
        else:
            result = scrape_upload_task.run(None, job.id, content, file.filename)
            return jsonify({"job_id": job.id, "result": result}), 200

    except Exception as e:
        current_app.logger.error("upload_file error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "internal_server_error", "detail": str(e)}), 500


@bp.route("/jobs/<int:job_id>", methods=["GET"])
@jwt_required()
def job_status(job_id):
    try:
        job = ScrapeJob.query.get_or_404(job_id)
        return jsonify({
            "job_id": job.id,
            "status": job.status,
            "meta": job.meta or {},
            "created_at": job.created_at.isoformat(),
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "source": job.source,
        })
    except Exception as e:
        current_app.logger.error("job_status error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "internal_server_error", "detail": str(e)}), 500


# List jobs (for current user) — used by frontend fetchJobs()
@bp.route("/jobs", methods=["GET"])
@jwt_required()
def list_jobs():
    try:
        user_id = get_jwt_identity()
        # optionally allow pagination
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        q = ScrapeJob.query.filter_by(user_id=user_id).order_by(ScrapeJob.created_at.desc())
        pagination = q.paginate(page=page, per_page=per_page, error_out=False)
        items = [{
            "job_id": j.id,
            "source": j.source,
            "status": j.status,
            "meta": j.meta or {},
            "created_at": j.created_at.isoformat(),
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        } for j in pagination.items]
        return jsonify({"items": items, "page": page, "pages": pagination.pages or 1})
    except Exception as e:
        current_app.logger.error("list_jobs error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "internal_server_error", "detail": str(e)}), 500