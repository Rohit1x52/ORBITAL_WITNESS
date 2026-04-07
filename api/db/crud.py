from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from .models import AnalysisHistory, AuditLog, User, UserSession


def get_or_create_user(db: Session, external_id: str | None = None, email: str | None = None) -> User | None:
    if not external_id and not email:
        return None

    query = db.query(User)
    if external_id:
        query = query.filter(User.external_id == external_id)
    elif email:
        query = query.filter(User.email == email)

    user = query.first()
    if user:
        return user

    user = User(external_id=external_id, email=email)
    db.add(user)
    db.flush()
    return user


def get_or_create_session(
    db: Session,
    session_id: str,
    user_agent: str | None,
    ip_address: str | None,
    user: User | None = None,
) -> UserSession:
    session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
    if session:
        session.last_seen = datetime.utcnow()
        if user_agent:
            session.user_agent = user_agent
        if ip_address:
            session.ip_address = ip_address
        if user and not session.user_id:
            session.user_id = user.id
        db.flush()
        return session

    session = UserSession(
        session_id=session_id,
        user_id=user.id if user else None,
        user_agent=user_agent,
        ip_address=ip_address,
    )
    db.add(session)
    db.flush()
    return session


def create_analysis_history(
    db: Session,
    *,
    task_id: str | None,
    session_id: str | None,
    user_id: int | None,
    location: tuple[float, float],
    before_date: str,
    after_date: str,
    workflow_mode: str,
    classification: dict[str, Any],
    summary: str,
    solutions: str,
    status: str,
    metadata: dict[str, Any] | None,
) -> AnalysisHistory:
    entry = AnalysisHistory(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
        location_lat=float(location[0]),
        location_lon=float(location[1]),
        before_date=before_date,
        after_date=after_date,
        workflow_mode=workflow_mode,
        classification_label=classification.get("label"),
        confidence=classification.get("confidence"),
        summary=summary,
        solutions=solutions,
        status=status,
        metadata=metadata,
    )
    db.add(entry)
    db.flush()
    return entry


def create_audit_log(
    db: Session,
    *,
    session_id: str | None,
    user_id: int | None,
    action: str,
    resource: str,
    status: str,
    details: dict[str, Any] | None,
) -> AuditLog:
    log = AuditLog(
        session_id=session_id,
        user_id=user_id,
        action=action,
        resource=resource,
        status=status,
        details=details,
    )
    db.add(log)
    db.flush()
    return log


def list_analysis_history(db: Session, session_id: str | None = None, limit: int = 50) -> list[AnalysisHistory]:
    query = db.query(AnalysisHistory).order_by(AnalysisHistory.created_at.desc())
    if session_id:
        query = query.filter(AnalysisHistory.session_id == session_id)
    return query.limit(limit).all()


def list_audit_logs(db: Session, session_id: str | None = None, limit: int = 50) -> list[AuditLog]:
    query = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if session_id:
        query = query.filter(AuditLog.session_id == session_id)
    return query.limit(limit).all()
