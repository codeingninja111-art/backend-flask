from __future__ import annotations

from sqlalchemy import func, select

from actions.helpers import append_audit
from auth import issue_session_token, verify_google_id_token
from models import Candidate, Employee, User
from utils import ApiError, AuthContext, iso_utc_now, normalize_role
from pii import decrypt_pii, encrypt_pii, hash_email, looks_like_sha256_hex


def _find_user_by_email(db, cfg, email_or_hash: str):
    s = str(email_or_hash or "").strip()
    if not s:
        return None

    # Preferred: plaintext email lookup (case-insensitive).
    email_lc = str(s).lower().strip()
    if email_lc and "@" in email_lc:
        found = db.execute(select(User).where(func.lower(User.email) == email_lc)).scalars().first()
        if found:
            return found

    # Backward compatibility: older DBs stored deterministic email hashes in `users.email`.
    # If PEPPER is available we can still find those users.
    if str(getattr(cfg, "PEPPER", "") or "").strip() and (looks_like_sha256_hex(s) or ("@" in s)):
        email_h = s.lower() if looks_like_sha256_hex(s) else hash_email(s, cfg.PEPPER)
        if email_h:
            found = db.execute(select(User).where(User.email == email_h)).scalars().first()
            if found:
                return found
    return None


def _update_user_last_login(db, cfg, email_or_hash: str):
    u = _find_user_by_email(db, cfg, email_or_hash)
    if not u:
        return
    u.lastLoginAt = iso_utc_now()


def _find_employee_by_employee_id(db, employee_id: str):
    emp_id = str(employee_id or "").strip()
    if not emp_id:
        return None
    return db.execute(select(Employee).where(Employee.employeeId == emp_id)).scalar_one_or_none()


def login_exchange(data, auth: AuthContext | None, db, cfg):
    id_token = (data or {}).get("idToken")
    google_user = verify_google_id_token(
        id_token,
        google_client_id=cfg.GOOGLE_CLIENT_ID,
        allow_test_tokens=bool(cfg.AUTH_ALLOW_TEST_TOKENS),
    )

    plain_email = str(google_user.get("email") or "").strip().lower()
    plain_name = str(google_user.get("fullName") or "").strip()

    user = _find_user_by_email(db, cfg, plain_email)
    if not user:
        raise ApiError("AUTH_INVALID", "User not found in Users")

    if str(user.status or "").upper() != "ACTIVE":
        raise ApiError("AUTH_INVALID", "User is disabled")

    # Best-effort: store encrypted-at-rest full values on login so UI can show normal values
    # without storing plaintext in DB.
    if str(getattr(cfg, "PII_ENC_KEY", "") or "").strip():
        if plain_email and "@" in plain_email:
            enc = encrypt_pii(plain_email, key=cfg.PII_ENC_KEY, aad=f"user:{user.userId}:email")
            if enc:
                user.email_enc = enc
        if plain_name and "*" not in plain_name:
            enc = encrypt_pii(plain_name, key=cfg.PII_ENC_KEY, aad=f"user:{user.userId}:name")
            if enc:
                user.name_enc = enc

    _update_user_last_login(db, cfg, user.email)
    ses = issue_session_token(
        db,
        user_id=user.userId,
        email=user.email,
        role=user.role,
        session_ttl_minutes=cfg.SESSION_TTL_MINUTES,
    )

    append_audit(
        db,
        entityType="AUTH",
        entityId=str(user.userId),
        action="LOGIN_EXCHANGE",
        stageTag="AUTH_LOGIN",
        remark="",
        actor=AuthContext(valid=True, userId=user.userId, email=user.email, role=normalize_role(user.role) or "", expiresAt=ses["expiresAt"]),
        meta={"email": user.email},
    )

    return {
        "sessionToken": ses["sessionToken"],
        "expiresAt": ses["expiresAt"],
        "me": {
            "userId": user.userId,
            "email": plain_email or "",
            "fullName": plain_name or "",
            "role": normalize_role(user.role),
        },
    }


def employee_login(data, auth: AuthContext | None, db, cfg):
    employee_id = str((data or {}).get("employeeId") or "").strip()
    if not employee_id:
        raise ApiError("BAD_REQUEST", "Missing employeeId")

    emp = _find_employee_by_employee_id(db, employee_id)
    if not emp:
        raise ApiError("AUTH_INVALID", "Invalid employeeId")
    if not str(emp.candidateId or "").strip():
        raise ApiError("AUTH_INVALID", "Employee not linked to candidate")

    full_name = ""
    if str(getattr(cfg, "PII_ENC_KEY", "") or "").strip():
        cand = db.execute(select(Candidate).where(Candidate.candidateId == str(emp.candidateId or ""))).scalar_one_or_none()
        if cand:
            full_name = decrypt_pii(getattr(cand, "name_enc", "") or "", key=cfg.PII_ENC_KEY, aad=f"candidate:{cand.candidateId}:name")

    user_id = emp.employeeId
    ses = issue_session_token(
        db,
        user_id=user_id,
        email="",
        role="EMPLOYEE",
        session_ttl_minutes=cfg.SESSION_TTL_MINUTES,
    )

    try:
        append_audit(
            db,
            entityType="AUTH",
            entityId=str(user_id),
            action="EMPLOYEE_LOGIN",
            stageTag="AUTH_LOGIN",
            remark="",
            actor=AuthContext(valid=True, userId=user_id, email="", role="EMPLOYEE", expiresAt=ses["expiresAt"]),
            meta={"employeeId": emp.employeeId, "candidateId": emp.candidateId},
        )
    except Exception:
        pass

    return {
        "sessionToken": ses["sessionToken"],
        "expiresAt": ses["expiresAt"],
        "me": {
            "userId": emp.employeeId,
            "email": "",
            "fullName": full_name or emp.employeeId,
            "role": "EMPLOYEE",
            "employeeId": emp.employeeId,
            "candidateId": emp.candidateId,
            "jobRole": emp.jobRole or "",
            "jobTitle": emp.jobTitle or "",
        },
    }


def session_validate(data, auth: AuthContext | None, db, cfg):
    if not auth or not auth.valid:
        raise ApiError("AUTH_INVALID", "Invalid or expired session")
    return {
        "valid": True,
        "expiresAt": auth.expiresAt,
        # Don't echo internal identifiers (hashed email) back to the frontend; GET_ME returns masked profile fields.
        "me": {"userId": auth.userId, "email": "", "role": normalize_role(auth.role)},
    }


def get_me(data, auth: AuthContext | None, db, cfg):
    if not auth or not auth.valid:
        raise ApiError("AUTH_INVALID", "Invalid or expired session")

    if normalize_role(auth.role) == "EMPLOYEE":
        emp_id = str(auth.userId or "").strip()
        emp = _find_employee_by_employee_id(db, emp_id)
        if not emp:
            raise ApiError("AUTH_INVALID", "Employee missing")
        full_name = ""
        if str(getattr(cfg, "PII_ENC_KEY", "") or "").strip():
            cand = db.execute(select(Candidate).where(Candidate.candidateId == str(emp.candidateId or ""))).scalar_one_or_none()
            if cand:
                full_name = decrypt_pii(getattr(cand, "name_enc", "") or "", key=cfg.PII_ENC_KEY, aad=f"candidate:{cand.candidateId}:name")
        return {
            "me": {
                "userId": emp.employeeId,
                "email": "",
                "fullName": full_name or emp.employeeId,
                "role": "EMPLOYEE",
                "employeeId": emp.employeeId,
                "candidateId": emp.candidateId or "",
                "jobRole": emp.jobRole or "",
                "jobTitle": emp.jobTitle or "",
            }
        }

    user = _find_user_by_email(db, cfg, auth.email)
    if not user:
        raise ApiError("AUTH_INVALID", "User missing")
    if str(user.status or "").upper() != "ACTIVE":
        raise ApiError("AUTH_INVALID", "User is disabled")

    email_full = ""
    name_full = ""
    if str(getattr(cfg, "PII_ENC_KEY", "") or "").strip():
        email_full = decrypt_pii(getattr(user, "email_enc", "") or "", key=cfg.PII_ENC_KEY, aad=f"user:{user.userId}:email")
        name_full = decrypt_pii(getattr(user, "name_enc", "") or "", key=cfg.PII_ENC_KEY, aad=f"user:{user.userId}:name")

    return {
        "me": {
            "userId": user.userId,
            "email": email_full or "",
            "fullName": name_full or user.userId,
            "role": normalize_role(user.role),
        }
    }
