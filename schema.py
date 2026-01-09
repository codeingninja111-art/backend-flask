from __future__ import annotations

import glob
import json
import logging
import os
import re

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from models import (
    AssignedTraining,
    Candidate,
    CandidateTest,
    Employee,
    JobPosting,
    JobTemplate,
    Permission,
    Requirement,
    Role,
    SLAConfig,
    Setting,
    Session as DbSession,
    TrainingMaster,
    User,
)
from pii import (
    decrypt_pii,
    encrypt_pii,
    get_pii_enc_key,
    hash_email,
    looks_like_sha256_hex,
    normalize_email,
    normalize_name,
    normalize_phone,
)


_log = logging.getLogger("schema")
_SAFE_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def _safe_ext(filename: str) -> str:
    _base, ext = os.path.splitext(str(filename or "").strip())
    if ext and _SAFE_EXT_RE.fullmatch(ext):
        return ext.lower()
    return ""


def _parse_name_mobile_from_cv_filename(filename: str) -> tuple[str, str]:
    """
    Best-effort parse of (candidateName, mobile) from a filename like:
      Name_Mobile_Source.pdf
      First_Last_Mobile_Source.pdf
    """

    raw = os.path.basename(str(filename or "").strip())
    if not raw:
        return "", ""
    base, _ext = os.path.splitext(raw)
    parts = [p.strip() for p in base.split("_") if p.strip()]
    if len(parts) < 2:
        return "", ""

    mobile_idx = None
    for i in range(1, len(parts)):
        digits = re.sub(r"\D+", "", parts[i])
        if len(digits) >= 8:
            mobile_idx = i
            break
    if mobile_idx is None:
        return "", ""

    name = " ".join(parts[:mobile_idx]).strip()
    mobile = parts[mobile_idx].strip()
    return name, mobile


def _quoted(name: str) -> str:
    return f"\"{str(name).replace('\"', '\"\"')}\""


def _ensure_column(engine, *, table: str, column: str, ddl_type: str, default_sql: str = "''") -> None:
    insp = inspect(engine)
    cols = {c.get("name") for c in insp.get_columns(table)}
    if column in cols:
        return
    ddl = f"ALTER TABLE {_quoted(table)} ADD COLUMN {_quoted(column)} {ddl_type} DEFAULT {default_sql}"
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _ensure_index(engine, *, name: str, table: str, column: str) -> None:
    ddl = f"CREATE INDEX IF NOT EXISTS {_quoted(name)} ON {_quoted(table)}({_quoted(column)})"
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _drop_index(engine, *, name: str) -> None:
    ddl = f"DROP INDEX IF EXISTS {_quoted(name)}"
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception:
        return


def _sqlite_version_tuple(engine) -> tuple[int, int, int] | None:
    try:
        if getattr(engine, "dialect", None) is None or engine.dialect.name != "sqlite":
            return None
        with engine.begin() as conn:
            v = str(conn.execute(text("select sqlite_version()")) .scalar() or "")
        parts = [int(p) for p in v.split(".")[:3] if p.isdigit()]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
        if len(parts) == 2:
            return (parts[0], parts[1], 0)
        return None
    except Exception:
        return None


def _drop_column_if_exists(engine, *, table: str, column: str) -> None:
    try:
        insp = inspect(engine)
        cols = {c.get("name") for c in insp.get_columns(table)}
        if column not in cols:
            return

        dialect = getattr(engine, "dialect", None)
        dname = dialect.name if dialect else ""

        if dname == "sqlite":
            v = _sqlite_version_tuple(engine)
            # SQLite supports DROP COLUMN from 3.35.0
            if not v or v < (3, 35, 0):
                return
            ddl = f"ALTER TABLE {_quoted(table)} DROP COLUMN {_quoted(column)}"
        else:
            ddl = f"ALTER TABLE {_quoted(table)} DROP COLUMN IF EXISTS {_quoted(column)}"

        with engine.begin() as conn:
            conn.execute(text(ddl))
    except Exception:
        return


def ensure_schema(engine) -> None:
    """
    Lightweight, idempotent schema evolution (no Alembic).

    Adds new columns and backfills deterministic hashes/masked values for PII fields.
    """
    pepper = str(os.getenv("PEPPER", "") or "").strip()

    # Users: keep encrypted-at-rest columns (optional).
    _ensure_column(engine, table="users", column="email_enc", ddl_type="TEXT")
    _ensure_column(engine, table="users", column="name_enc", ddl_type="TEXT")

    # Candidates: add normalized fields for dedupe/lookups + encrypted-at-rest columns.
    _ensure_column(engine, table="candidates", column="candidateNameNorm", ddl_type="TEXT")
    _ensure_column(engine, table="candidates", column="mobileNorm", ddl_type="TEXT")
    _ensure_column(engine, table="candidates", column="name_enc", ddl_type="TEXT")
    _ensure_column(engine, table="candidates", column="mobile_enc", ddl_type="TEXT")

    _ensure_index(engine, name="ix_candidates_candidateNameNorm", table="candidates", column="candidateNameNorm")
    _ensure_index(engine, name="ix_candidates_mobileNorm", table="candidates", column="mobileNorm")

    # CandidateTests (tenant isolation)
    _ensure_column(engine, table="candidate_tests", column="fillOwnerUserId", ddl_type="TEXT")
    _ensure_index(engine, name="ix_candidate_tests_fillOwnerUserId", table="candidate_tests", column="fillOwnerUserId")

    # Trainings: multi-video support
    _ensure_column(engine, table="trainings_master", column="videoLinksJson", ddl_type="TEXT")
    _ensure_column(engine, table="assigned_trainings", column="videoLinksJson", ddl_type="TEXT")

    _backfill_pii(engine)
    _backfill_actor_refs(engine, pepper=pepper)
    _backfill_test_owners(engine)
    _backfill_training_videos(engine)

    # Best-effort removal of legacy deterministic hash columns + indexes.
    _drop_index(engine, name="ix_users_email_hash")
    _drop_index(engine, name="ix_users_name_hash")
    _drop_index(engine, name="ix_candidates_name_hash")
    _drop_index(engine, name="ix_candidates_mobile_hash")

    _drop_column_if_exists(engine, table="users", column="email_hash")
    _drop_column_if_exists(engine, table="users", column="name_hash")
    _drop_column_if_exists(engine, table="users", column="email_masked")
    _drop_column_if_exists(engine, table="users", column="name_masked")

    _drop_column_if_exists(engine, table="candidates", column="name_hash")
    _drop_column_if_exists(engine, table="candidates", column="mobile_hash")
    _drop_column_if_exists(engine, table="candidates", column="name_masked")
    _drop_column_if_exists(engine, table="candidates", column="mobile_masked")


def _backfill_pii(engine) -> None:
    enc_key = get_pii_enc_key()
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads").strip() or "./uploads"
    file_storage_mode = os.getenv("FILE_STORAGE_MODE", "local").strip().lower()
    with Session(engine) as db:
        users = db.query(User).all()  # type: ignore[attr-defined]
        user_updates = 0
        for u in users:
            stored_email = str(getattr(u, "email", "") or "").strip()
            stored_name = str(getattr(u, "fullName", "") or "").strip()
            email_enc_existing = str(getattr(u, "email_enc", "") or "").strip()
            name_enc_existing = str(getattr(u, "name_enc", "") or "").strip()

            changed = False
            # If the DB currently stores masked/hash-like values, recover plaintext from encrypted columns.
            if enc_key and email_enc_existing and ("@" not in stored_email or looks_like_sha256_hex(stored_email) or "*" in stored_email):
                dec = decrypt_pii(email_enc_existing, key=enc_key, aad=f"user:{getattr(u, 'userId', '')}:email")
                dec_lc = normalize_email(dec)
                if dec_lc and "@" in dec_lc and dec_lc != stored_email:
                    u.email = dec_lc
                    changed = True
            if enc_key and name_enc_existing and (not stored_name or "*" in stored_name or looks_like_sha256_hex(stored_name)):
                dec = decrypt_pii(name_enc_existing, key=enc_key, aad=f"user:{getattr(u, 'userId', '')}:name")
                if dec and dec != stored_name:
                    u.fullName = dec
                    changed = True

            # Best-effort: populate encrypted columns when we have plaintext.
            if enc_key and not email_enc_existing and stored_email and "@" in stored_email and not looks_like_sha256_hex(stored_email) and "*" not in stored_email:
                enc = encrypt_pii(stored_email, key=enc_key, aad=f"user:{getattr(u, 'userId', '')}:email")
                if enc:
                    u.email_enc = enc
                    changed = True
            if enc_key and not name_enc_existing and stored_name and "*" not in stored_name and not looks_like_sha256_hex(stored_name):
                enc = encrypt_pii(stored_name, key=enc_key, aad=f"user:{getattr(u, 'userId', '')}:name")
                if enc:
                    u.name_enc = enc
                    changed = True

            if changed:
                user_updates += 1

        candidates = db.query(Candidate).all()  # type: ignore[attr-defined]
        cand_updates = 0
        for c in candidates:
            stored_name = str(getattr(c, "candidateName", "") or "").strip()
            name_enc_existing = str(getattr(c, "name_enc", "") or "").strip()
            cv_filename_existing = str(getattr(c, "cvFileName", "") or "").strip()
            stored_mobile = str(getattr(c, "mobile", "") or "").strip()
            mobile_enc_existing = str(getattr(c, "mobile_enc", "") or "").strip()
            changed = False

            # Recover plaintext from encrypted columns when candidateName/mobile are masked.
            if enc_key and name_enc_existing and (not stored_name or "*" in stored_name or looks_like_sha256_hex(stored_name)):
                dec = decrypt_pii(name_enc_existing, key=enc_key, aad=f"candidate:{getattr(c, 'candidateId', '')}:name")
                if dec and dec != stored_name:
                    c.candidateName = dec
                    stored_name = dec
                    changed = True
            if enc_key and mobile_enc_existing and (not stored_mobile or "x" in stored_mobile.lower() or "*" in stored_mobile or looks_like_sha256_hex(stored_mobile)):
                dec = decrypt_pii(mobile_enc_existing, key=enc_key, aad=f"candidate:{getattr(c, 'candidateId', '')}:mobile")
                dec_norm = normalize_phone(dec)
                if dec_norm and dec_norm != stored_mobile:
                    c.mobile = dec_norm
                    stored_mobile = dec_norm
                    changed = True

            # Recovery + scrubbing: older rows stored plaintext PII in CV filenames
            # (e.g., Name_Mobile_Source.pdf). Extract full PII into encrypted columns
            # and scrub DB+filesystem filenames to remove plaintext.
            if cv_filename_existing and enc_key:
                cv_name, cv_mobile = _parse_name_mobile_from_cv_filename(cv_filename_existing)

                cv_mobile_norm = normalize_phone(cv_mobile) if cv_mobile else ""
                allow_mobile = bool(cv_mobile_norm)
                # If mobile matches (or no mobile hash exists yet), trust CV filename name for display.
                allow_name = bool(cv_name) and allow_mobile

                if allow_mobile and not mobile_enc_existing:
                    enc = encrypt_pii(cv_mobile_norm, key=enc_key, aad=f"candidate:{getattr(c, 'candidateId', '')}:mobile")
                    if enc and str(getattr(c, "mobile_enc", "") or "") != enc:
                        c.mobile_enc = enc
                        changed = True

                if allow_name and not name_enc_existing:
                    enc = encrypt_pii(cv_name, key=enc_key, aad=f"candidate:{getattr(c, 'candidateId', '')}:name")
                    if enc and str(getattr(c, "name_enc", "") or "") != enc:
                        c.name_enc = enc
                        changed = True

                # Scrub CV filename and local upload file name (idempotent).
                if "_" in cv_filename_existing and any(ch.isdigit() for ch in cv_filename_existing):
                    cid = str(getattr(c, "candidateId", "") or "").strip()
                    if cid:
                        ext = _safe_ext(cv_filename_existing)
                        new_cv_name = f"CV_{cid}{ext}"
                        if str(getattr(c, "cvFileName", "") or "") != new_cv_name:
                            c.cvFileName = new_cv_name
                            changed = True

                        if file_storage_mode != "gas":
                            fid = str(getattr(c, "cvFileId", "") or "").strip()
                            if fid:
                                try:
                                    pattern = os.path.join(upload_dir, f"{fid}_*")
                                    matches = sorted(glob.glob(pattern))
                                    if matches:
                                        src = matches[0]
                                        dst = os.path.join(upload_dir, f"{fid}_{new_cv_name}")
                                        if src != dst and not os.path.exists(dst):
                                            os.rename(src, dst)
                                except Exception:
                                    pass

            if enc_key and not name_enc_existing and stored_name and "*" not in stored_name and not looks_like_sha256_hex(stored_name):
                enc = encrypt_pii(stored_name, key=enc_key, aad=f"candidate:{getattr(c, 'candidateId', '')}:name")
                if enc and str(getattr(c, "name_enc", "") or "") != enc:
                    c.name_enc = enc
                    changed = True
            if (
                enc_key
                and not mobile_enc_existing
                and stored_mobile
                and "*" not in stored_mobile
                and "x" not in stored_mobile.lower()
                and not looks_like_sha256_hex(stored_mobile)
            ):
                enc = encrypt_pii(stored_mobile, key=enc_key, aad=f"candidate:{getattr(c, 'candidateId', '')}:mobile")
                if enc and str(getattr(c, "mobile_enc", "") or "") != enc:
                    c.mobile_enc = enc
                    changed = True

            # Maintain normalized columns.
            name_norm = normalize_name(stored_name)
            mob_norm = normalize_phone(stored_mobile)
            if str(getattr(c, "candidateNameNorm", "") or "") != name_norm:
                c.candidateNameNorm = name_norm
                changed = True
            if str(getattr(c, "mobileNorm", "") or "") != mob_norm:
                c.mobileNorm = mob_norm
                changed = True

            if changed:
                cand_updates += 1

        # Sessions: store plaintext email when we can link to a userId.
        ses_updates = 0
        sessions = db.query(DbSession).all()  # type: ignore[attr-defined]
        users_by_id = {str(getattr(u, "userId", "") or ""): u for u in users}
        for s in sessions:
            uid = str(getattr(s, "userId", "") or "").strip()
            u = users_by_id.get(uid)
            if not u:
                continue
            u_email = str(getattr(u, "email", "") or "").strip()
            if u_email and str(getattr(s, "email", "") or "") != u_email:
                s.email = u_email
                ses_updates += 1

        if user_updates or cand_updates or ses_updates:
            db.commit()
            _log.info("PII backfill complete users=%s candidates=%s sessions=%s", user_updates, cand_updates, ses_updates)


def _backfill_actor_refs(engine, *, pepper: str) -> None:
    """
    Convert legacy audit-ish actor columns (e.g. createdBy/updatedBy) that may contain
    plaintext emails or email-hashes into stable, UI-friendly userIds.

    This avoids showing long SHA256 hex digests on the frontend while still keeping
    deterministic references for older rows.
    """

    def _convert_actor_ref(value: str, email_to_user_id: dict[str, str], email_hash_to_user_id: dict[str, str]) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""

        # Already a human-friendly identifier (USR-xxxx / EMP-xxxx / SYSTEM / etc.)
        if "@" not in raw and not looks_like_sha256_hex(raw):
            return raw

        if looks_like_sha256_hex(raw):
            h = raw.lower()
            return email_hash_to_user_id.get(h, h)

        # Plaintext email → userId
        if "@" in raw:
            e = normalize_email(raw)
            if e:
                return email_to_user_id.get(e, raw)
        return raw

    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())

    with Session(engine) as db:
        email_to_user_id: dict[str, str] = {}
        email_hash_to_user_id: dict[str, str] = {}
        if "users" in existing_tables:
            for u in db.query(User).all():  # type: ignore[attr-defined]
                user_id = str(getattr(u, "userId", "") or "").strip()
                if not user_id:
                    continue

                stored_email = str(getattr(u, "email", "") or "").strip()
                if "@" in stored_email:
                    email_to_user_id[normalize_email(stored_email)] = user_id

                    # Optional backward compat: map deterministic legacy hash(email) → userId
                    if pepper:
                        h = hash_email(stored_email, pepper)
                        if h:
                            email_hash_to_user_id[h.lower()] = user_id
                elif looks_like_sha256_hex(stored_email):
                    # If the email column itself contains a legacy hash.
                    email_hash_to_user_id[stored_email.lower()] = user_id

        targets: list[tuple[str, object, tuple[str, ...]]] = [
            ("users", User, ("createdBy", "updatedBy")),
            ("roles", Role, ("createdBy", "updatedBy")),
            ("permissions", Permission, ("updatedBy",)),
            ("settings", Setting, ("updatedBy",)),
            ("job_templates", JobTemplate, ("createdBy", "updatedBy")),
            ("requirements", Requirement, ("createdBy", "updatedBy")),
            ("job_posting", JobPosting, ("completedBy", "updatedBy")),
            ("candidates", Candidate, ("createdBy", "updatedBy")),
            ("employees", Employee, ("createdBy",)),
            ("sla_config", SLAConfig, ("updatedBy",)),
            ("trainings_master", TrainingMaster, ("created_by",)),
            ("assigned_trainings", AssignedTraining, ("assigned_by",)),
        ]

        updated = 0
        for table_name, model, fields in targets:
            if table_name not in existing_tables:
                continue
            for row in db.query(model).all():  # type: ignore[attr-defined]
                for field in fields:
                    old = str(getattr(row, field, "") or "").strip()
                    new = _convert_actor_ref(old, email_to_user_id, email_hash_to_user_id)
                    if new == old:
                        continue
                    setattr(row, field, new)
                    updated += 1

        if updated:
            db.commit()
            _log.info("Actor ref backfill complete updated=%s", updated)


def _backfill_test_owners(engine) -> None:
    with Session(engine) as db:
        rows = (
            db.query(CandidateTest)  # type: ignore[attr-defined]
            .filter(CandidateTest.fillOwnerUserId == "")
            .filter(CandidateTest.filledBy != "")
            .all()
        )
        if not rows:
            return
        for r in rows:
            r.fillOwnerUserId = str(getattr(r, "filledBy", "") or "").strip()
        db.commit()


def _backfill_training_videos(engine) -> None:
    with Session(engine) as db:
        updated = 0

        masters = (
            db.query(TrainingMaster)  # type: ignore[attr-defined]
            .filter(TrainingMaster.videoLinksJson == "")
            .filter(TrainingMaster.video_link != "")
            .all()
        )
        for t in masters:
            t.videoLinksJson = f"[{json.dumps(str(getattr(t, 'video_link', '') or '').strip())}]"
            updated += 1

        assigned = (
            db.query(AssignedTraining)  # type: ignore[attr-defined]
            .filter(AssignedTraining.videoLinksJson == "")
            .filter(AssignedTraining.video_link != "")
            .all()
        )
        for t in assigned:
            t.videoLinksJson = f"[{json.dumps(str(getattr(t, 'video_link', '') or '').strip())}]"
            updated += 1

        if updated:
            db.commit()
