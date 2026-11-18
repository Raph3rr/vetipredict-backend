# routes/emails.py
from flask import Blueprint, request, jsonify, send_file, current_app
from ..extensions import db
from ..models import EmailRecord, ScrapeJob
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
import io
import csv
from sqlalchemy import desc

bp = Blueprint("emails", __name__)

@bp.route("", methods=["GET", "OPTIONS"], strict_slashes=False)
def list_emails():
    if request.method == "OPTIONS":
        return "", 200

    verify_jwt_in_request()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))
    q = request.args.get("q")
    source = request.args.get("source")
    query = EmailRecord.query
    if q:
        query = query.filter(EmailRecord.email.ilike(f"%{q}%"))
    if source:
        query = query.filter(EmailRecord.source.ilike(f"%{source}%"))
    query = query.order_by(desc(EmailRecord.created_at))
    pag = query.paginate(page=page, per_page=per_page, error_out=False)
    items = [{
        "id": e.id,
        "email": e.email,
        "source": e.source,
        "context_snippet": e.context_snippet,
        "created_at": e.created_at.isoformat()
    } for e in pag.items]
    return jsonify({
        "items": items,
        "total": pag.total,
        "page": pag.page,
        "per_page": pag.per_page,
        "pages": pag.pages
    })


@bp.route("/export", methods=["GET", "OPTIONS"], strict_slashes=False)
def export_csv():
    if request.method == "OPTIONS":
        return "", 200

    verify_jwt_in_request()
    q = request.args.get("q")
    source = request.args.get("source")
    query = EmailRecord.query
    if q:
        query = query.filter(EmailRecord.email.ilike(f"%{q}%"))
    if source:
        query = query.filter(EmailRecord.source.ilike(f"%{source}%"))
    query = query.order_by(EmailRecord.created_at)
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["id", "email", "source", "created_at"])
    for e in query.all():
        writer.writerow([e.id, e.email, e.source or "", e.created_at.isoformat()])
    mem = io.BytesIO()
    mem.write(si.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", download_name="emails_export.csv", as_attachment=True)
