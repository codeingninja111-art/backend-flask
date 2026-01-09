from __future__ import annotations

from sqlalchemy import Boolean, Column, Integer, String, Text, UniqueConstraint

from db import Base


class IdCounter(Base):
    __tablename__ = "id_counters"

    key = Column(String, primary_key=True)
    nextValue = Column(Integer, nullable=False, default=1)


class User(Base):
    __tablename__ = "users"

    userId = Column(String, primary_key=True)
    email = Column(String, nullable=False, unique=True, index=True)
    fullName = Column(Text, nullable=False, default="")
    # Optional encrypted-at-rest full values (AES-256-GCM).
    # When present, these can be used for recovery/migration.
    email_enc = Column(Text, nullable=False, default="")
    name_enc = Column(Text, nullable=False, default="")
    role = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="ACTIVE", index=True)
    lastLoginAt = Column(Text, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class Role(Base):
    __tablename__ = "roles"

    roleCode = Column(String, primary_key=True)
    roleName = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="ACTIVE")
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class Permission(Base):
    __tablename__ = "permissions"
    __table_args__ = (UniqueConstraint("permType", "permKey", name="uq_permissions_type_key"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    permType = Column(String, nullable=False)
    permKey = Column(String, nullable=False)
    rolesCsv = Column(Text, nullable=False, default="")
    enabled = Column(Boolean, nullable=False, default=True)
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False, default="")
    type = Column(String, nullable=False, default="")
    scope = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class JobTemplate(Base):
    __tablename__ = "job_templates"

    templateId = Column(String, primary_key=True)
    jobRole = Column(Text, nullable=False, default="")
    jobTitle = Column(Text, nullable=False, default="")
    jd = Column(Text, nullable=False, default="")
    responsibilities = Column(Text, nullable=False, default="")
    skills = Column(Text, nullable=False, default="")
    shift = Column(Text, nullable=False, default="")
    payScale = Column(Text, nullable=False, default="")
    perks = Column(Text, nullable=False, default="")
    notes = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="ACTIVE")
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class Requirement(Base):
    __tablename__ = "requirements"

    requirementId = Column(String, primary_key=True)
    templateId = Column(String, nullable=False, default="")
    jobRole = Column(Text, nullable=False, default="")
    jobTitle = Column(Text, nullable=False, default="")
    jd = Column(Text, nullable=False, default="")
    responsibilities = Column(Text, nullable=False, default="")
    skills = Column(Text, nullable=False, default="")
    shift = Column(Text, nullable=False, default="")
    payScale = Column(Text, nullable=False, default="")
    perks = Column(Text, nullable=False, default="")
    notes = Column(Text, nullable=False, default="")
    raisedFor = Column(Text, nullable=False, default="")
    concernedPerson = Column(Text, nullable=False, default="")
    requiredCount = Column(Integer, nullable=False, default=1)
    joinedCount = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="DRAFT", index=True)
    latestRemark = Column(Text, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class RequirementHistory(Base):
    __tablename__ = "requirement_history"

    historyId = Column(String, primary_key=True)
    requirementId = Column(String, nullable=False, index=True)
    fromStatus = Column(String, nullable=False, default="")
    toStatus = Column(String, nullable=False, default="")
    stageTag = Column(String, nullable=False, default="")
    remark = Column(Text, nullable=False, default="")
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    at = Column(Text, nullable=False, default="")
    metaJson = Column(Text, nullable=False, default="")


class JobPosting(Base):
    __tablename__ = "job_posting"

    requirementId = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="")
    checklistStateJson = Column(Text, nullable=False, default="")
    screenshotUploadId = Column(String, nullable=False, default="")
    completedAt = Column(Text, nullable=False, default="")
    completedBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class Candidate(Base):
    __tablename__ = "candidates"

    candidateId = Column(String, primary_key=True)
    requirementId = Column(String, nullable=False, index=True)
    candidateName = Column(Text, nullable=False, default="")
    jobRole = Column(Text, nullable=False, default="")
    mobile = Column(String, nullable=False, default="")
    # Non-hash normalized fields used for dedupe/lookups.
    candidateNameNorm = Column(Text, nullable=False, default="", index=True)
    mobileNorm = Column(String, nullable=False, default="", index=True)
    # Optional encrypted-at-rest full values (AES-256-GCM).
    name_enc = Column(Text, nullable=False, default="")
    mobile_enc = Column(Text, nullable=False, default="")
    source = Column(Text, nullable=False, default="")
    cvFileId = Column(String, nullable=False, default="")
    cvFileName = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="", index=True)
    holdUntil = Column(Text, nullable=False, default="")
    walkinAt = Column(Text, nullable=False, default="")
    walkinNotes = Column(Text, nullable=False, default="")
    notPickCount = Column(Integer, nullable=False, default=0)
    preCallAt = Column(Text, nullable=False, default="")
    preInterviewStatus = Column(Text, nullable=False, default="")
    preInterviewMarks = Column(Text, nullable=False, default="")
    preInterviewMarksAt = Column(Text, nullable=False, default="")
    testToken = Column(String, nullable=False, default="", index=True)
    testTokenExpiresAt = Column(Text, nullable=False, default="")
    onlineTestScore = Column(Integer, nullable=True)
    onlineTestResult = Column(Text, nullable=False, default="")
    onlineTestSubmittedAt = Column(Text, nullable=False, default="")
    testDecisionsJson = Column(Text, nullable=False, default="")
    candidate_test_failed_but_manually_continued = Column(Boolean, nullable=False, default=False)
    inPersonMarks = Column(Integer, nullable=True)
    inPersonMarksAt = Column(Text, nullable=False, default="")
    techSelectedTestsJson = Column(Text, nullable=False, default="")
    techSelectedAt = Column(Text, nullable=False, default="")
    tallyMarks = Column(Integer, nullable=True)
    voiceMarks = Column(Integer, nullable=True)
    techReview = Column(Text, nullable=False, default="")
    excelMarks = Column(Integer, nullable=True)
    excelReview = Column(Text, nullable=False, default="")
    techResult = Column(Text, nullable=False, default="")
    techEvaluatedAt = Column(Text, nullable=False, default="")
    finalHoldAt = Column(Text, nullable=False, default="")
    finalHoldRemark = Column(Text, nullable=False, default="")
    joiningAt = Column(Text, nullable=False, default="")
    docsJson = Column(Text, nullable=False, default="")
    docsCompleteAt = Column(Text, nullable=False, default="")
    joinedAt = Column(Text, nullable=False, default="")
    probationStartAt = Column(Text, nullable=False, default="")
    probationEndsAt = Column(Text, nullable=False, default="")
    employeeId = Column(String, nullable=False, default="", index=True)
    rejectedFromStatus = Column(Text, nullable=False, default="")
    rejectedReasonCode = Column(Text, nullable=False, default="")
    rejectedAt = Column(Text, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class TestDecisionLog(Base):
    __tablename__ = "logs_test_decision"

    logId = Column(String, primary_key=True)
    candidateId = Column(String, nullable=False, index=True)
    requirementId = Column(String, nullable=False, index=True)
    testType = Column(String, nullable=False, default="")
    marks = Column(Text, nullable=False, default="")
    passFail = Column(String, nullable=False, default="")
    hrDecision = Column(String, nullable=False, default="")
    remark = Column(Text, nullable=False, default="")
    overrideFlag = Column(Boolean, nullable=False, default=False)
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    at = Column(Text, nullable=False, default="")
    metaJson = Column(Text, nullable=False, default="")


class Session(Base):
    __tablename__ = "sessions"

    sessionId = Column(String, primary_key=True)
    tokenHash = Column(String, nullable=False, unique=True, index=True)
    tokenPrefix = Column(String, nullable=False, default="", index=True)
    userId = Column(String, nullable=False, default="")
    email = Column(String, nullable=False, default="")
    role = Column(String, nullable=False, default="", index=True)
    issuedAt = Column(Text, nullable=False, default="")
    expiresAt = Column(Text, nullable=False, default="")
    lastSeenAt = Column(Text, nullable=False, default="")
    revokedAt = Column(Text, nullable=False, default="")
    revokedBy = Column(String, nullable=False, default="")


class AuditLog(Base):
    __tablename__ = "audit_log"

    logId = Column(String, primary_key=True)
    entityType = Column(String, nullable=False, default="", index=True)
    entityId = Column(String, nullable=False, default="", index=True)
    action = Column(String, nullable=False, default="", index=True)
    fromState = Column(String, nullable=False, default="")
    toState = Column(String, nullable=False, default="")
    stageTag = Column(String, nullable=False, default="", index=True)
    remark = Column(Text, nullable=False, default="")
    actorUserId = Column(String, nullable=False, default="", index=True)
    actorRole = Column(String, nullable=False, default="", index=True)
    at = Column(Text, nullable=False, default="", index=True)
    metaJson = Column(Text, nullable=False, default="")


class RejectionLog(Base):
    __tablename__ = "logs_rejection"

    logId = Column(String, primary_key=True)
    candidateId = Column(String, nullable=False, index=True)
    requirementId = Column(String, nullable=False, index=True)
    rejectionType = Column(String, nullable=False, default="")
    autoRejectCode = Column(String, nullable=False, default="")
    stageTag = Column(String, nullable=False, default="", index=True)
    remark = Column(Text, nullable=False, default="")
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    at = Column(Text, nullable=False, default="", index=True)


class OnlineTest(Base):
    __tablename__ = "online_tests"

    testId = Column(String, primary_key=True)
    token = Column(String, nullable=False, unique=True, index=True)
    candidateId = Column(String, nullable=False, index=True)
    requirementId = Column(String, nullable=False, index=True)
    issuedAt = Column(Text, nullable=False, default="")
    expiresAt = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="")
    fullName = Column(Text, nullable=False, default="")
    applyingFor = Column(Text, nullable=False, default="")
    source = Column(Text, nullable=False, default="")
    questionsJson = Column(Text, nullable=False, default="")
    answersJson = Column(Text, nullable=False, default="")
    score = Column(Integer, nullable=True)
    result = Column(String, nullable=False, default="")
    submittedAt = Column(Text, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")


class HoldLog(Base):
    __tablename__ = "logs_hold"

    logId = Column(String, primary_key=True)
    candidateId = Column(String, nullable=False, index=True)
    requirementId = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False, default="")
    holdUntil = Column(Text, nullable=False, default="")
    stageTag = Column(String, nullable=False, default="", index=True)
    remark = Column(Text, nullable=False, default="")
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    at = Column(Text, nullable=False, default="", index=True)


class JoinLog(Base):
    __tablename__ = "logs_join"

    logId = Column(String, primary_key=True)
    candidateId = Column(String, nullable=False, index=True)
    requirementId = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False, default="")
    stageTag = Column(String, nullable=False, default="", index=True)
    remark = Column(Text, nullable=False, default="")
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    at = Column(Text, nullable=False, default="", index=True)


class Employee(Base):
    __tablename__ = "employees"

    employeeId = Column(String, primary_key=True)
    candidateId = Column(String, nullable=False, default="", index=True)
    requirementId = Column(String, nullable=False, default="", index=True)
    employeeName = Column(Text, nullable=False, default="")
    mobile = Column(String, nullable=False, default="")
    jobRole = Column(Text, nullable=False, default="")
    jobTitle = Column(Text, nullable=False, default="")
    source = Column(Text, nullable=False, default="")
    cvFileId = Column(String, nullable=False, default="")
    cvFileName = Column(Text, nullable=False, default="")
    joinedAt = Column(Text, nullable=False, default="")
    probationStartAt = Column(Text, nullable=False, default="")
    probationEndsAt = Column(Text, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    timelineJson = Column(Text, nullable=False, default="")


class TrainingMaster(Base):
    __tablename__ = "trainings_master"

    training_id = Column(String, primary_key=True)
    name = Column(Text, nullable=False, default="")
    department = Column(Text, nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    video_link = Column(Text, nullable=False, default="")
    videoLinksJson = Column(Text, nullable=False, default="")
    documentsJson = Column(Text, nullable=False, default="")
    created_by = Column(String, nullable=False, default="")
    created_on = Column(Text, nullable=False, default="")


class AssignedTraining(Base):
    __tablename__ = "assigned_trainings"

    assigned_id = Column(String, primary_key=True)
    candidate_id = Column(String, nullable=False, default="", index=True)
    training_id = Column(String, nullable=False, default="", index=True)
    training_name = Column(Text, nullable=False, default="")
    department = Column(Text, nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    video_link = Column(Text, nullable=False, default="")
    videoLinksJson = Column(Text, nullable=False, default="")
    documentsJson = Column(Text, nullable=False, default="")
    status = Column(String, nullable=False, default="", index=True)
    assigned_date = Column(Text, nullable=False, default="")
    due_date = Column(Text, nullable=False, default="")
    start_time = Column(Text, nullable=False, default="")
    completion_time = Column(Text, nullable=False, default="")
    assigned_by = Column(String, nullable=False, default="")


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Text, nullable=False, default="", index=True)
    candidate_id = Column(String, nullable=False, default="", index=True)
    training_id = Column(String, nullable=False, default="", index=True)
    assigned_id = Column(String, nullable=False, default="", index=True)
    action = Column(String, nullable=False, default="", index=True)
    performed_by = Column(String, nullable=False, default="")
    remarks = Column(Text, nullable=False, default="")
    metaJson = Column(Text, nullable=False, default="")


class TestMaster(Base):
    __tablename__ = "test_master"

    testKey = Column(String, primary_key=True)
    label = Column(Text, nullable=False, default="")
    fillRolesJson = Column(Text, nullable=False, default="[]")
    reviewRolesJson = Column(Text, nullable=False, default="[]")
    active = Column(Boolean, nullable=False, default=True, index=True)
    ordering = Column(Integer, nullable=False, default=0)
    createdAt = Column(Text, nullable=False, default="")
    createdBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class CandidateTest(Base):
    __tablename__ = "candidate_tests"
    __table_args__ = (UniqueConstraint("candidateId", "testKey", name="uq_candidate_tests_candidate_test"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidateId = Column(String, nullable=False, default="", index=True)
    requirementId = Column(String, nullable=False, default="", index=True)
    testKey = Column(String, nullable=False, default="", index=True)
    isRequired = Column(Boolean, nullable=False, default=False, index=True)
    status = Column(String, nullable=False, default="NOT_SELECTED", index=True)
    marksJson = Column(Text, nullable=False, default="")
    marksNumber = Column(Integer, nullable=True)
    # Multi-tenant isolation: assignee for filling the test (userId).
    fillOwnerUserId = Column(String, nullable=False, default="", index=True)
    filledBy = Column(String, nullable=False, default="")
    filledAt = Column(Text, nullable=False, default="")
    reviewedBy = Column(String, nullable=False, default="")
    reviewedAt = Column(Text, nullable=False, default="")
    remarks = Column(Text, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="", index=True)


class FailCandidate(Base):
    __tablename__ = "fail_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidateId = Column(String, nullable=False, default="", index=True)
    requirementId = Column(String, nullable=False, default="", index=True)
    stageName = Column(String, nullable=False, default="", index=True)
    reason = Column(Text, nullable=False, default="")
    score = Column(Integer, nullable=True)
    failedAt = Column(Text, nullable=False, default="", index=True)
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    resolvedAt = Column(Text, nullable=False, default="", index=True)
    resolvedBy = Column(String, nullable=False, default="")
    resolution = Column(String, nullable=False, default="", index=True)
    metaJson = Column(Text, nullable=False, default="")


class SLAConfig(Base):
    __tablename__ = "sla_config"

    stepName = Column(String, primary_key=True)
    plannedMinutes = Column(Integer, nullable=False, default=0)
    enabled = Column(Boolean, nullable=False, default=True)
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class StepMetric(Base):
    __tablename__ = "step_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requirementId = Column(String, nullable=False, default="", index=True)
    candidateId = Column(String, nullable=False, default="", index=True)
    stepName = Column(String, nullable=False, default="", index=True)
    plannedMinutes = Column(Integer, nullable=False, default=0)
    startTs = Column(Text, nullable=False, default="")
    endTs = Column(Text, nullable=False, default="")
    actualMinutes = Column(Integer, nullable=True)
    breached = Column(Boolean, nullable=False, default=False)
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="", index=True)


class CandidateTrainingState(Base):
    __tablename__ = "candidate_training_state"

    candidateId = Column(String, primary_key=True)
    requirementId = Column(String, nullable=False, default="", index=True)
    markedCompleteAt = Column(Text, nullable=False, default="")
    markedCompleteBy = Column(String, nullable=False, default="")
    closedAt = Column(Text, nullable=False, default="", index=True)
    closedBy = Column(String, nullable=False, default="")
    updatedAt = Column(Text, nullable=False, default="")
    updatedBy = Column(String, nullable=False, default="")


class ProbationLog(Base):
    __tablename__ = "probation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidateId = Column(String, nullable=False, default="", index=True)
    employeeId = Column(String, nullable=False, default="", index=True)
    requirementId = Column(String, nullable=False, default="", index=True)
    profileSnapshotJson = Column(Text, nullable=False, default="")
    trainingsSnapshotJson = Column(Text, nullable=False, default="")
    probationStartAt = Column(Text, nullable=False, default="")
    probationEndsAt = Column(Text, nullable=False, default="")
    decision = Column(String, nullable=False, default="")
    decidedAt = Column(Text, nullable=False, default="")
    actorUserId = Column(String, nullable=False, default="")
    actorRole = Column(String, nullable=False, default="")
    createdAt = Column(Text, nullable=False, default="", index=True)
