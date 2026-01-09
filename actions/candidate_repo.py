from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from sqlalchemy import func, select

from actions.helpers import append_audit, append_rejection_log
from models import Candidate, FailCandidate, TestDecisionLog
from utils import ApiError, AuthContext, iso_utc_now
from pii import normalize_name, normalize_phone


def normalize_mobile(mobile: Any) -> str:
    return normalize_phone(str(mobile or ""))


def normalize_candidate_name(name: Any) -> str:
    return normalize_name(str(name or ""))


def has_duplicate_candidate_in_requirement(db, *, requirement_id: str, name_norm: str, mobile_norm: str) -> bool:
    nh = normalize_name(str(name_norm or ""))
    mh = normalize_phone(str(mobile_norm or ""))
    if not requirement_id or not nh or not mh:
        return False
    row = (
        db.execute(
            select(Candidate.candidateId)
            .where(Candidate.requirementId == requirement_id)
            .where(Candidate.candidateNameNorm == nh)
            .where(Candidate.mobileNorm == mh)
        )
        .scalars()
        .first()
    )
    return bool(row)


def find_candidate(db, *, candidate_id: str, requirement_id: str = "") -> Candidate:
    cand = db.execute(select(Candidate).where(Candidate.candidateId == str(candidate_id))).scalar_one_or_none()
    if not cand:
        raise ApiError("NOT_FOUND", "Candidate not found")
    if requirement_id and str(cand.requirementId or "") != str(requirement_id):
        raise ApiError("BAD_REQUEST", "Candidate does not belong to requirement")
    return cand


def update_candidate(db, *, cand: Candidate, patch: dict[str, Any], auth: AuthContext):
    def set_str(key: str, value: Any):
        setattr(cand, key, str(value or ""))

    def set_iso(key: str, value: Any):
        setattr(cand, key, str(value or ""))

    def set_int_or_none(key: str, value: Any):
        if value == "" or value is None:
            setattr(cand, key, None)
            return
        try:
            setattr(cand, key, int(value))
        except Exception:
            setattr(cand, key, None)

    if "jobRole" in patch and patch["jobRole"] is not None:
        set_str("jobRole", patch["jobRole"])
    if "status" in patch and patch["status"] is not None:
        set_str("status", patch["status"])
    if "holdUntil" in patch and patch["holdUntil"] is not None:
        set_iso("holdUntil", patch["holdUntil"])
    if "walkinAt" in patch and patch["walkinAt"] is not None:
        set_iso("walkinAt", patch["walkinAt"])
    if "walkinNotes" in patch and patch["walkinNotes"] is not None:
        set_str("walkinNotes", patch["walkinNotes"])
    if "notPickCount" in patch and patch["notPickCount"] is not None:
        try:
            cand.notPickCount = int(patch["notPickCount"] or 0)
        except Exception:
            cand.notPickCount = 0
    if "preCallAt" in patch and patch["preCallAt"] is not None:
        set_iso("preCallAt", patch["preCallAt"])
    if "preInterviewStatus" in patch and patch["preInterviewStatus"] is not None:
        set_str("preInterviewStatus", patch["preInterviewStatus"])
    if "preInterviewMarks" in patch and patch["preInterviewMarks"] is not None:
        cand.preInterviewMarks = "" if patch["preInterviewMarks"] == "" else patch["preInterviewMarks"]
    if "preInterviewMarksAt" in patch and patch["preInterviewMarksAt"] is not None:
        set_iso("preInterviewMarksAt", patch["preInterviewMarksAt"])
    if "testToken" in patch and patch["testToken"] is not None:
        set_str("testToken", patch["testToken"])
    if "testTokenExpiresAt" in patch and patch["testTokenExpiresAt"] is not None:
        set_iso("testTokenExpiresAt", patch["testTokenExpiresAt"])
    if "onlineTestScore" in patch and patch["onlineTestScore"] is not None:
        set_int_or_none("onlineTestScore", patch["onlineTestScore"])
    if "onlineTestResult" in patch and patch["onlineTestResult"] is not None:
        set_str("onlineTestResult", patch["onlineTestResult"])
    if "onlineTestSubmittedAt" in patch and patch["onlineTestSubmittedAt"] is not None:
        set_iso("onlineTestSubmittedAt", patch["onlineTestSubmittedAt"])
    if "testDecisionsJson" in patch and patch["testDecisionsJson"] is not None:
        set_str("testDecisionsJson", patch["testDecisionsJson"])
    if "candidate_test_failed_but_manually_continued" in patch and patch["candidate_test_failed_but_manually_continued"] is not None:
        cand.candidate_test_failed_but_manually_continued = bool(patch["candidate_test_failed_but_manually_continued"])
    if "inPersonMarks" in patch and patch["inPersonMarks"] is not None:
        set_int_or_none("inPersonMarks", patch["inPersonMarks"])
    if "inPersonMarksAt" in patch and patch["inPersonMarksAt"] is not None:
        set_iso("inPersonMarksAt", patch["inPersonMarksAt"])
    if "techSelectedTestsJson" in patch and patch["techSelectedTestsJson"] is not None:
        set_str("techSelectedTestsJson", patch["techSelectedTestsJson"])
    if "techSelectedAt" in patch and patch["techSelectedAt"] is not None:
        set_iso("techSelectedAt", patch["techSelectedAt"])
    if "tallyMarks" in patch and patch["tallyMarks"] is not None:
        set_int_or_none("tallyMarks", patch["tallyMarks"])
    if "voiceMarks" in patch and patch["voiceMarks"] is not None:
        set_int_or_none("voiceMarks", patch["voiceMarks"])
    if "techReview" in patch and patch["techReview"] is not None:
        set_str("techReview", patch["techReview"])
    if "excelMarks" in patch and patch["excelMarks"] is not None:
        set_int_or_none("excelMarks", patch["excelMarks"])
    if "excelReview" in patch and patch["excelReview"] is not None:
        set_str("excelReview", patch["excelReview"])
    if "techResult" in patch and patch["techResult"] is not None:
        set_str("techResult", patch["techResult"])
    if "techEvaluatedAt" in patch and patch["techEvaluatedAt"] is not None:
        set_iso("techEvaluatedAt", patch["techEvaluatedAt"])
    if "finalHoldAt" in patch and patch["finalHoldAt"] is not None:
        set_iso("finalHoldAt", patch["finalHoldAt"])
    if "finalHoldRemark" in patch and patch["finalHoldRemark"] is not None:
        set_str("finalHoldRemark", patch["finalHoldRemark"])
    if "joiningAt" in patch and patch["joiningAt"] is not None:
        set_iso("joiningAt", patch["joiningAt"])
    if "docsJson" in patch and patch["docsJson"] is not None:
        set_str("docsJson", patch["docsJson"])
    if "docsCompleteAt" in patch and patch["docsCompleteAt"] is not None:
        set_iso("docsCompleteAt", patch["docsCompleteAt"])
    if "joinedAt" in patch and patch["joinedAt"] is not None:
        set_iso("joinedAt", patch["joinedAt"])
    if "probationStartAt" in patch and patch["probationStartAt"] is not None:
        set_iso("probationStartAt", patch["probationStartAt"])
    if "probationEndsAt" in patch and patch["probationEndsAt"] is not None:
        set_iso("probationEndsAt", patch["probationEndsAt"])
    if "employeeId" in patch and patch["employeeId"] is not None:
        set_str("employeeId", patch["employeeId"])
    if "rejectedFromStatus" in patch and patch["rejectedFromStatus"] is not None:
        set_str("rejectedFromStatus", patch["rejectedFromStatus"])
    if "rejectedReasonCode" in patch and patch["rejectedReasonCode"] is not None:
        set_str("rejectedReasonCode", patch["rejectedReasonCode"])
    if "rejectedAt" in patch and patch["rejectedAt"] is not None:
        set_iso("rejectedAt", patch["rejectedAt"])

    cand.updatedAt = iso_utc_now()
    cand.updatedBy = auth.userId


def _parse_json_object(raw: Any, fallback: dict) -> dict:
    try:
        s = str(raw or "").strip()
        if not s:
            return fallback
        o = json.loads(s)
        if not isinstance(o, dict):
            return fallback
        return o
    except Exception:
        return fallback


def get_test_decisions(cand: Candidate) -> dict:
    return _parse_json_object(getattr(cand, "testDecisionsJson", "") or "", {})


def is_test_continued(cand: Candidate, test_type: str) -> bool:
    mp = get_test_decisions(cand)
    entry = mp.get(test_type) if isinstance(mp, dict) else None
    if not isinstance(entry, dict):
        return False
    return str(entry.get("decision") or "").upper() == "CONTINUE"


def upsert_test_decision(
    db,
    *,
    candidate_id: str,
    requirement_id: str,
    test_type: str,
    decision: str,
    remark: str,
    meta: dict,
    auth: AuthContext,
):
    cand = find_candidate(db, candidate_id=candidate_id, requirement_id=requirement_id)
    existing = get_test_decisions(cand)
    now = iso_utc_now()

    existing[test_type] = {
        "decision": str(decision or "").upper(),
        "at": now,
        "userId": auth.userId,
        "role": auth.role,
        "remark": remark or "",
    }

    continued = str(decision or "").upper() == "CONTINUE"

    update_candidate(
        db,
        cand=cand,
        patch={
            "testDecisionsJson": json.dumps(existing),
            "candidate_test_failed_but_manually_continued": True if continued else bool(cand.candidate_test_failed_but_manually_continued),
        },
        auth=auth,
    )

    db.add(
        TestDecisionLog(
            logId="LOG-" + os.urandom(16).hex(),
            candidateId=candidate_id,
            requirementId=requirement_id,
            testType=str(test_type or ""),
            marks=json.dumps(meta.get("marks")) if meta.get("marks") is not None else "",
            passFail=str(meta.get("passFail") or ""),
            hrDecision=str(decision or "").upper(),
            remark=str(remark or ""),
            overrideFlag=True if continued else False,
            actorUserId=auth.userId,
            actorRole=auth.role,
            at=now,
            metaJson=json.dumps(meta or {}),
        )
    )

    append_audit(
        db,
        entityType="CANDIDATE",
        entityId=candidate_id,
        action="TEST_FAIL_DECIDE",
        fromState=str(cand.status or ""),
        toState=str(cand.status or ""),
        stageTag=f"{str(test_type or 'TEST')} {str(decision or '')}",
        remark=str(remark or ""),
        actor=auth,
        at=now,
        meta={"requirementId": requirement_id, "testType": test_type, "decision": decision, "meta": meta or {}},
    )

    if str(test_type or "").upper().strip() == "ONLINE_TEST":
        row = (
            db.execute(
                select(FailCandidate)
                .where(FailCandidate.candidateId == candidate_id)
                .where(func.upper(FailCandidate.stageName) == "ONLINE_TEST")
                .where(FailCandidate.resolvedAt == "")
                .order_by(FailCandidate.failedAt.desc())
            )
            .scalars()
            .first()
        )
        if row:
            row.resolvedAt = now
            row.resolvedBy = str(auth.userId or auth.email or "")
            row.resolution = str(decision or "").upper().strip()

    return {"ok": True, "testType": test_type, "decision": str(decision or "").upper(), "at": now}


def reject_candidate_with_meta(
    db,
    *,
    candidate_id: str,
    requirement_id: str,
    stage_tag: str,
    remark: str,
    reason_code: str,
    auth: AuthContext | None,
):
    cand = find_candidate(db, candidate_id=candidate_id, requirement_id=requirement_id)
    current_status = str(cand.status or "").upper()
    if current_status == "REJECTED":
        return {"ok": True, "status": "REJECTED", "alreadyRejected": True}

    now = iso_utc_now()
    update_candidate(
        db,
        cand=cand,
        patch={
            "rejectedFromStatus": current_status,
            "rejectedReasonCode": str(reason_code or ""),
            "rejectedAt": now,
            "status": "REJECTED",
        },
        auth=auth if auth else AuthContext(valid=True, userId="SYSTEM", email="SYSTEM", role="SYSTEM", expiresAt=""),
    )

    append_rejection_log(
        db,
        candidateId=candidate_id,
        requirementId=requirement_id,
        stageTag=stage_tag,
        remark=remark,
        actor=auth,
        reasonCode=reason_code,
    )

    return {
        "ok": True,
        "status": "REJECTED",
        "rejectedFromStatus": current_status,
        "rejectedReasonCode": str(reason_code or ""),
        "rejectedAt": now,
    }
