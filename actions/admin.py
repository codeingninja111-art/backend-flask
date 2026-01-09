from __future__ import annotations

import re

from sqlalchemy import func, select

from actions.helpers import append_audit, next_prefixed_id
from auth import permissions_for_role
from models import AuditLog, JobTemplate, Permission, Role, Setting, User
from utils import ApiError, AuthContext, iso_utc_now, normalize_role
from pii import decrypt_pii, encrypt_pii, hash_email


def users_list(data, auth: AuthContext | None, db, cfg):
    role = data.get("role") if isinstance(data, dict) else None
    status = data.get("status") if isinstance(data, dict) else None
    q = data.get("q") if isinstance(data, dict) else None
    page = int(data.get("page") or 1) if isinstance(data, dict) else 1
    page_size = int(data.get("pageSize") or 100) if isinstance(data, dict) else 100
    page = max(1, page)
    page_size = max(1, min(500, page_size))

    role_uc = str(role).upper().strip() if role else None
    status_uc = str(status).upper().strip() if status else None
    q_str = str(q).strip() if q else ""
    q_lc = q_str.lower() if q_str else None

    can_pii = bool(getattr(cfg, "PII_ENC_KEY", "").strip()) and bool(auth and auth.valid) and str(getattr(auth, "role", "") or "").upper() in set(
        getattr(cfg, "PII_VIEW_ROLES", []) or []
    )

    rows = db.execute(select(User)).scalars().all()
    items = []
    for u in rows:
        email_full = ""
        name_full = ""
        if can_pii:
            email_full = decrypt_pii(getattr(u, "email_enc", "") or "", key=cfg.PII_ENC_KEY, aad=f"user:{u.userId}:email")
            name_full = decrypt_pii(getattr(u, "name_enc", "") or "", key=cfg.PII_ENC_KEY, aad=f"user:{u.userId}:name")

        item = {
            "userId": u.userId or "",
            "email": email_full or (u.email or "") or (u.userId or ""),
            "fullName": name_full or (u.fullName or "") or (u.userId or ""),
            "role": u.role or "",
            "status": u.status or "",
            "lastLoginAt": u.lastLoginAt or "",
            "createdAt": u.createdAt or "",
            "updatedAt": u.updatedAt or "",
        }
        if role_uc and str(item["role"]).upper() != role_uc:
            continue
        if status_uc and str(item["status"]).upper() != status_uc:
            continue
        if q_lc:
            hay = f"{item['userId']} {item['email']} {item['fullName']} {item['role']}".lower()
            if q_lc not in hay:
                continue
        items.append(item)

    items.sort(key=lambda x: (str(x.get("role", "")), str(x.get("email", ""))))
    total = len(items)
    start = (page - 1) * page_size
    paged = items[start : start + page_size]
    return {"items": paged, "total": total}


def users_upsert(data, auth: AuthContext | None, db, cfg):
    email_raw = str(data.get("email") or "").strip()
    email = str(email_raw or "").lower().strip()
    full_name = str(data.get("fullName") or "").strip()
    role = str(data.get("role") or "").upper().strip()
    active = data.get("active") if isinstance(data.get("active"), bool) else None
    status = str(data.get("status") or "").upper().strip()

    if not email:
        raise ApiError("BAD_REQUEST", "Missing email")
    if not role:
        raise ApiError("BAD_REQUEST", "Missing role")

    role_row = db.execute(select(Role).where(Role.roleCode == role)).scalar_one_or_none()
    if role_row and str(role_row.status or "").upper() != "ACTIVE":
        raise ApiError("BAD_REQUEST", "Invalid or inactive role")
    if not role_row and role not in {"ADMIN", "EA", "HR", "OWNER", "EMPLOYEE"}:
        raise ApiError("BAD_REQUEST", "Invalid or inactive role")

    if active is True:
        final_status = "ACTIVE"
    elif active is False:
        final_status = "DISABLED"
    elif status:
        final_status = status
    else:
        final_status = "ACTIVE"

    now = iso_utc_now()
    updated_by = str(auth.userId or "") if auth else ""

    # Prefer plaintext email match
    existing = db.execute(select(User).where(func.lower(User.email) == email)).scalar_one_or_none()

    # Backward compatibility: older DBs stored deterministic email hashes in `users.email`.
    if not existing and str(getattr(cfg, "PEPPER", "") or "").strip():
        legacy_h = hash_email(email, cfg.PEPPER)
        if legacy_h:
            existing = db.execute(select(User).where(User.email == legacy_h)).scalar_one_or_none()
    if not existing:
        existing_ids = [x for x in db.execute(select(User.userId)).scalars().all()]
        user_id = next_prefixed_id(
            db,
            counter_key="USR",
            prefix="USR-",
            pad=4,
            existing_ids=existing_ids,
        )

        email_enc = ""
        name_enc = ""
        if str(getattr(cfg, "PII_ENC_KEY", "") or "").strip():
            email_enc = encrypt_pii(email, key=cfg.PII_ENC_KEY, aad=f"user:{user_id}:email")
            name_enc = encrypt_pii(full_name, key=cfg.PII_ENC_KEY, aad=f"user:{user_id}:name")

        db.add(
            User(
                userId=user_id,
                email=email,
                fullName=full_name,
                email_enc=email_enc,
                name_enc=name_enc,
                role=role,
                status=final_status,
                lastLoginAt="",
                createdAt=now,
                createdBy=updated_by,
                updatedAt=now,
                updatedBy=updated_by,
            )
        )

        append_audit(
            db,
            entityType="USER",
            entityId=user_id,
            action="USERS_UPSERT",
            stageTag="ADMIN_USERS_UPSERT",
            actor=auth,
            meta={"mode": "create", "email": email, "role": role, "status": final_status},
            at=now,
        )
        return {"userId": user_id, "mode": "create"}

    existing.email = email
    existing.fullName = full_name
    if str(getattr(cfg, "PII_ENC_KEY", "") or "").strip():
        email_enc = encrypt_pii(email, key=cfg.PII_ENC_KEY, aad=f"user:{existing.userId}:email")
        name_enc = encrypt_pii(full_name, key=cfg.PII_ENC_KEY, aad=f"user:{existing.userId}:name")
        if email_enc:
            existing.email_enc = email_enc
        if name_enc:
            existing.name_enc = name_enc
    existing.role = role
    existing.status = final_status
    existing.updatedAt = now
    existing.updatedBy = updated_by

    append_audit(
        db,
        entityType="USER",
        entityId=existing.userId,
        action="USERS_UPSERT",
        stageTag="ADMIN_USERS_UPSERT",
        actor=auth,
        meta={"mode": "update", "email": email, "role": role, "status": final_status},
        at=now,
    )
    return {"userId": existing.userId, "mode": "update"}


def roles_list(data, auth: AuthContext | None, db, cfg):
    include_inactive = data.get("includeInactive") if isinstance(data.get("includeInactive"), bool) else True
    rows = db.execute(select(Role)).scalars().all()
    items = []
    for r in rows:
        role_code = normalize_role(r.roleCode)
        if not role_code:
            continue
        status = str(r.status or "ACTIVE").upper()
        if not include_inactive and status != "ACTIVE":
            continue
        items.append(
            {
                "roleCode": role_code,
                "roleName": str(r.roleName or role_code),
                "status": status,
                "createdAt": r.createdAt or "",
                "createdBy": r.createdBy or "",
                "updatedAt": r.updatedAt or "",
                "updatedBy": r.updatedBy or "",
            }
        )
    items.sort(key=lambda x: str(x.get("roleCode", "")))
    return {"items": items}


def roles_upsert(data, auth: AuthContext | None, db, cfg):
    role_code = normalize_role(data.get("roleCode"))
    role_name = str(data.get("roleName") or "").strip()
    status = str(data.get("status") or "ACTIVE").upper().strip()

    if not role_code:
        raise ApiError("BAD_REQUEST", "Missing roleCode")
    if not re.match(r"^[A-Z0-9_]{2,30}$", role_code):
        raise ApiError("BAD_REQUEST", "Invalid roleCode")
    if not role_name:
        role_name = role_code
    if status not in {"ACTIVE", "INACTIVE"}:
        raise ApiError("BAD_REQUEST", "Invalid status")

    now = iso_utc_now()
    by = str(auth.userId or "") if auth else ""

    existing = db.execute(select(Role).where(Role.roleCode == role_code)).scalar_one_or_none()
    if not existing:
        db.add(
            Role(
                roleCode=role_code,
                roleName=role_name,
                status=status,
                createdAt=now,
                createdBy=by,
                updatedAt=now,
                updatedBy=by,
            )
        )
        append_audit(
            db,
            entityType="ROLE",
            entityId=role_code,
            action="ROLES_UPSERT",
            stageTag="ADMIN_ROLES_UPSERT",
            actor=auth,
            meta={"mode": "create", "roleCode": role_code, "status": status},
            at=now,
        )
    else:
        existing.roleName = role_name
        existing.status = status
        existing.updatedAt = now
        existing.updatedBy = by
        append_audit(
            db,
            entityType="ROLE",
            entityId=role_code,
            action="ROLES_UPSERT",
            stageTag="ADMIN_ROLES_UPSERT",
            actor=auth,
            meta={"mode": "update", "roleCode": role_code, "status": status},
            at=now,
        )

    return {"roleCode": role_code}


def permissions_list(data, auth: AuthContext | None, db, cfg):
    rows = db.execute(select(Permission)).scalars().all()
    items = []
    for p in rows:
        perm_type = str(p.permType or "").upper().strip()
        perm_key = str(p.permKey or "").upper().strip()
        if not perm_type or not perm_key:
            continue
        items.append(
            {
                "permType": perm_type,
                "permKey": perm_key,
                "rolesCsv": str(p.rolesCsv or ""),
                "enabled": bool(p.enabled),
                "updatedAt": p.updatedAt or "",
                "updatedBy": p.updatedBy or "",
            }
        )

    items.sort(key=lambda x: f"{x['permType']}:{x['permKey']}")
    return {"items": items}


def permissions_upsert(data, auth: AuthContext | None, db, cfg):
    items = data.get("items") if isinstance(data, dict) else None
    if items is None and isinstance(data, list):
        items = data
    if not isinstance(items, list):
        raise ApiError("BAD_REQUEST", "items must be an array")
    if len(items) > 200:
        raise ApiError("BAD_REQUEST", "Max 200 rules per batch")

    now = iso_utc_now()
    by = str(auth.userId or "") if auth else ""

    existing = {}
    for p in db.execute(select(Permission)).scalars().all():
        pt = str(p.permType or "").upper().strip()
        pk = str(p.permKey or "").upper().strip()
        if not pt or not pk:
            continue
        existing[f"{pt}:{pk}"] = p

    upserted = 0
    appended = 0
    for it in items:
        it = it or {}
        perm_type = str(it.get("permType") or "").upper().strip()
        perm_key = str(it.get("permKey") or "").upper().strip()
        roles_csv = str(it.get("rolesCsv") or "").strip()
        enabled = bool(it.get("enabled") is True)

        if perm_type not in {"ACTION", "UI"}:
            raise ApiError("BAD_REQUEST", "Invalid permType")
        if not perm_key:
            raise ApiError("BAD_REQUEST", "Missing permKey")

        key = f"{perm_type}:{perm_key}"
        row = existing.get(key)
        if not row:
            db.add(
                Permission(
                    permType=perm_type,
                    permKey=perm_key,
                    rolesCsv=roles_csv,
                    enabled=enabled,
                    updatedAt=now,
                    updatedBy=by,
                )
            )
            appended += 1
        else:
            row.rolesCsv = roles_csv
            row.enabled = enabled
            row.updatedAt = now
            row.updatedBy = by
        upserted += 1

    append_audit(
        db,
        entityType="PERMISSION",
        entityId="BATCH",
        action="PERMISSIONS_UPSERT",
        stageTag="ADMIN_PERMISSIONS_UPSERT",
        actor=auth,
        meta={"upserted": upserted, "appended": appended},
        at=now,
    )

    return {"upserted": upserted, "appended": appended}


def my_permissions_get(data, auth: AuthContext | None, db, cfg):
    if not auth or not auth.valid:
        raise ApiError("AUTH_INVALID", "Login required")
    return permissions_for_role(db, auth.role)


def template_list(data, auth: AuthContext | None, db, cfg):
    status = data.get("status") if isinstance(data, dict) else None
    q = data.get("q") if isinstance(data, dict) else None
    page = int(data.get("page") or 1) if isinstance(data, dict) else 1
    page_size = int(data.get("pageSize") or 200) if isinstance(data, dict) else 200
    page = max(1, page)
    page_size = max(1, min(500, page_size))

    status_uc = str(status).upper().strip() if status else None
    q_lc = str(q).lower().strip() if q else None

    rows = db.execute(select(JobTemplate)).scalars().all()
    items = []
    for t in rows:
        item = {
            "templateId": t.templateId or "",
            "jobRole": t.jobRole or "",
            "jobTitle": t.jobTitle or "",
            "jd": t.jd or "",
            "responsibilities": t.responsibilities or "",
            "skills": t.skills or "",
            "shift": t.shift or "",
            "payScale": t.payScale or "",
            "perks": t.perks or "",
            "notes": t.notes or "",
            "status": t.status or "ACTIVE",
            "createdAt": t.createdAt or "",
            "updatedAt": t.updatedAt or "",
        }
        if status_uc and str(item["status"]).upper() != status_uc:
            continue
        if q_lc:
            hay = f"{item['jobRole']} {item['jobTitle']}".lower()
            if q_lc not in hay:
                continue
        items.append(item)

    items.sort(key=lambda x: str(x.get("jobRole", "")))
    total = len(items)
    start = (page - 1) * page_size
    paged = items[start : start + page_size]
    return {"items": paged, "total": total}


def template_upsert(data, auth: AuthContext | None, db, cfg):
    template_id = str(data.get("templateId") or "").strip()
    job_role = str(data.get("jobRole") or "").strip()
    job_title = str(data.get("jobTitle") or "").strip()
    jd = str(data.get("jd") or "").strip()
    responsibilities = str(data.get("responsibilities") or "").strip()
    skills = str(data.get("skills") or "").strip()
    shift = str(data.get("shift") or "").strip()
    pay_scale = str(data.get("payScale") or "").strip()
    perks = str(data.get("perks") or "").strip()
    notes = str(data.get("notes") or "").strip()
    active = data.get("active") if isinstance(data.get("active"), bool) else None
    status = str(data.get("status") or "").upper().strip()

    if not job_role:
        raise ApiError("BAD_REQUEST", "Missing jobRole")
    if not job_title:
        raise ApiError("BAD_REQUEST", "Missing jobTitle")

    if active is True:
        final_status = "ACTIVE"
    elif active is False:
        final_status = "INACTIVE"
    elif status:
        final_status = status
    else:
        final_status = "ACTIVE"

    now = iso_utc_now()
    updated_by = str(auth.userId or "") if auth else ""

    found = None
    if template_id:
        found = db.execute(select(JobTemplate).where(JobTemplate.templateId == template_id)).scalar_one_or_none()
    if not found:
        found = db.execute(select(JobTemplate).where(JobTemplate.jobRole == job_role)).scalar_one_or_none()
        if found and not template_id:
            template_id = found.templateId

    if not found:
        existing_ids = [x for x in db.execute(select(JobTemplate.templateId)).scalars().all()]
        new_id = next_prefixed_id(db, counter_key="TPL", prefix="TPL-", pad=4, existing_ids=existing_ids)
        db.add(
            JobTemplate(
                templateId=new_id,
                jobRole=job_role,
                jobTitle=job_title,
                jd=jd,
                responsibilities=responsibilities,
                skills=skills,
                shift=shift,
                payScale=pay_scale,
                perks=perks,
                notes=notes,
                status=final_status,
                createdAt=now,
                createdBy=updated_by,
                updatedAt=now,
                updatedBy=updated_by,
            )
        )
        append_audit(
            db,
            entityType="TEMPLATE",
            entityId=new_id,
            action="TEMPLATE_UPSERT",
            stageTag="ADMIN_TEMPLATE_UPSERT",
            actor=auth,
            meta={"mode": "create", "jobRole": job_role, "status": final_status},
            at=now,
        )
        return {"templateId": new_id, "mode": "create"}

    found.jobRole = job_role
    found.jobTitle = job_title
    found.jd = jd
    found.responsibilities = responsibilities
    found.skills = skills
    found.shift = shift
    found.payScale = pay_scale
    found.perks = perks
    found.notes = notes
    found.status = final_status
    found.updatedAt = now
    found.updatedBy = updated_by

    append_audit(
        db,
        entityType="TEMPLATE",
        entityId=found.templateId,
        action="TEMPLATE_UPSERT",
        stageTag="ADMIN_TEMPLATE_UPSERT",
        actor=auth,
        meta={"mode": "update", "jobRole": job_role, "status": final_status},
        at=now,
    )
    return {"templateId": found.templateId, "mode": "update"}


def settings_get(data, auth: AuthContext | None, db, cfg):
    defaults = {
        "PASSING_MARKS": {"key": "PASSING_MARKS", "value": 6, "type": "number", "scope": "GLOBAL"},
        "NOT_PICK_THRESHOLD": {"key": "NOT_PICK_THRESHOLD", "value": 3, "type": "number", "scope": "GLOBAL"},
        "INTERVIEW_MESSAGE_TEMPLATE": {"key": "INTERVIEW_MESSAGE_TEMPLATE", "value": "", "type": "string", "scope": "GLOBAL"},
        "TEST_DEFAULT_FILL_OWNER_BY_TESTKEY": {"key": "TEST_DEFAULT_FILL_OWNER_BY_TESTKEY", "value": "{}", "type": "json", "scope": "GLOBAL"},
    }

    rows = db.execute(select(Setting)).scalars().all()
    if not rows:
        return {"items": [defaults["PASSING_MARKS"], defaults["NOT_PICK_THRESHOLD"], defaults["INTERVIEW_MESSAGE_TEMPLATE"], defaults["TEST_DEFAULT_FILL_OWNER_BY_TESTKEY"]]}

    mp = {}
    for r in rows:
        k = str(r.key or "").strip()
        if not k:
            continue
        mp[k] = {
            "key": k,
            "value": r.value,
            "type": r.type or "string",
            "scope": r.scope or "GLOBAL",
            "updatedAt": r.updatedAt or "",
            "updatedBy": r.updatedBy or "",
        }

    out = [
        mp.get("PASSING_MARKS", defaults["PASSING_MARKS"]),
        mp.get("NOT_PICK_THRESHOLD", defaults["NOT_PICK_THRESHOLD"]),
        mp.get("INTERVIEW_MESSAGE_TEMPLATE", defaults["INTERVIEW_MESSAGE_TEMPLATE"]),
        mp.get("TEST_DEFAULT_FILL_OWNER_BY_TESTKEY", defaults["TEST_DEFAULT_FILL_OWNER_BY_TESTKEY"]),
    ]
    return {"items": out}


def copy_template_data(data, auth: AuthContext | None, db, cfg):
    row = db.execute(select(Setting).where(Setting.key == "INTERVIEW_MESSAGE_TEMPLATE")).scalar_one_or_none()
    return {"interviewMessageTemplate": str(row.value) if row else ""}


def settings_upsert(data, auth: AuthContext | None, db, cfg):
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ApiError("BAD_REQUEST", "Missing items")

    now = iso_utc_now()
    updated_by = str(auth.userId or "") if auth else ""

    existing = {s.key: s for s in db.execute(select(Setting)).scalars().all()}
    updated = 0

    for it in items:
        it = it or {}
        key = str(it.get("key") or "").strip()
        if not key:
            continue
        type_ = str(it.get("type") or "string").strip()
        scope = str(it.get("scope") or "GLOBAL").strip()
        value = it.get("value")

        row = existing.get(key)
        if not row:
            db.add(
                Setting(
                    key=key,
                    value=value,
                    type=type_,
                    scope=scope,
                    updatedAt=now,
                    updatedBy=updated_by,
                )
            )
        else:
            row.value = value
            row.type = type_
            row.scope = scope
            row.updatedAt = now
            row.updatedBy = updated_by
        updated += 1

    append_audit(
        db,
        entityType="SETTINGS",
        entityId="GLOBAL",
        action="SETTINGS_UPSERT",
        stageTag="ADMIN_SETTINGS_UPSERT",
        actor=auth,
        meta={"updated": updated},
        at=now,
    )
    return {"updated": updated}


def logs_query(data, auth: AuthContext | None, db, cfg):
    log_type = str((data or {}).get("logType") or "AUDIT").upper().strip()
    from_raw = (data or {}).get("from")
    to_raw = (data or {}).get("to")
    stage_tag = str((data or {}).get("stageTag") or "").strip() or None
    actor_role = str((data or {}).get("actorRole") or "").upper().strip() or None
    entity_type = str((data or {}).get("entityType") or "").upper().strip() or None
    entity_id = str((data or {}).get("entityId") or "").strip() or None
    candidate_id = str((data or {}).get("candidateId") or "").strip() or None
    requirement_id = str((data or {}).get("requirementId") or "").strip() or None
    page = int((data or {}).get("page") or 1)
    page_size = int((data or {}).get("pageSize") or 100)
    page = max(1, page)
    page_size = max(1, min(500, page_size))

    # Stored as ISO-8601 UTC strings; lexicographic comparisons work.
    from_s = str(from_raw or "").strip() or None
    to_s = str(to_raw or "").strip() or None

    if log_type == "AUDIT":
        rows = db.execute(select(AuditLog)).scalars().all()
        items = []
        for r in rows:
            if from_s and r.at and str(r.at) < from_s:
                continue
            if to_s and r.at and str(r.at) > to_s:
                continue
            if actor_role and str(r.actorRole or "").upper() != actor_role:
                continue
            if stage_tag and str(r.stageTag or "") != stage_tag:
                continue
            if entity_type and str(r.entityType or "").upper() != entity_type:
                continue
            if entity_id and str(r.entityId or "") != entity_id:
                continue
            items.append(
                {
                    "logId": r.logId,
                    "entityType": r.entityType,
                    "entityId": r.entityId,
                    "action": r.action,
                    "fromState": r.fromState,
                    "toState": r.toState,
                    "stageTag": r.stageTag,
                    "remark": r.remark,
                    "actorUserId": r.actorUserId,
                    "actorRole": r.actorRole,
                    "at": r.at,
                    "metaJson": r.metaJson,
                }
            )
    else:
        from models import HoldLog, JoinLog, RejectionLog

        if log_type == "REJECTION":
            rows = db.execute(select(RejectionLog)).scalars().all()
        elif log_type == "HOLD":
            rows = db.execute(select(HoldLog)).scalars().all()
        elif log_type == "JOIN":
            rows = db.execute(select(JoinLog)).scalars().all()
        else:
            raise ApiError("BAD_REQUEST", "Invalid logType")

        items = []
        for r in rows:
            if from_s and r.at and str(r.at) < from_s:
                continue
            if to_s and r.at and str(r.at) > to_s:
                continue
            if actor_role and str(r.actorRole or "").upper() != actor_role:
                continue
            if stage_tag and str(getattr(r, "stageTag", "") or "") != stage_tag:
                continue
            if candidate_id and str(getattr(r, "candidateId", "") or "") != candidate_id:
                continue
            if requirement_id and str(getattr(r, "requirementId", "") or "") != requirement_id:
                continue
            d = {k: getattr(r, k) for k in r.__table__.columns.keys()}  # type: ignore[attr-defined]
            items.append(d)

    items.sort(key=lambda x: str(x.get("at") or ""), reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    paged = items[start : start + page_size]
    return {"items": paged, "total": total}
