"""PostgreSQL persistence (SQLAlchemy).

What is stored, and why it is auditable:

* ``scan`` — the uploaded image's declared/measured resolution, paper
  rotation/scale and the *detector version* that produced them;
* ``plate_reading`` — per-plate candidate, 95% range, reading counts and
  status;
* ``confirmation`` — the human QC decision (confirmed / corrected / rejected)
  with the operator id and a free-text reason.  The machine result is never
  overwritten: corrections are separate rows.

The vision model is not used to close a control loop — these tables are a
record, never an actuator interface.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Iterator

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String,
                        Text, UniqueConstraint, create_engine, Index)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker, Session)


def database_url() -> str:
    return os.environ.get(
        "PRT1_DATABASE_URL",
        "postgresql+psycopg2://prt1:prt1@localhost:5432/prt1")


engine = create_engine(database_url(), pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width_px: Mapped[int] = mapped_column(Integer)
    height_px: Mapped[int] = mapped_column(Integer)
    declared_scan_ppi: Mapped[float | None] = mapped_column(Float, nullable=True)
    measured_ppi: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_rotation_deg: Mapped[float] = mapped_column(Float)
    frame_scale: Mapped[float] = mapped_column(Float)
    frame_quality: Mapped[float] = mapped_column(Float)
    residual_rotation_deg: Mapped[float] = mapped_column(Float)
    residual_scale: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32))
    detector_version: Mapped[str] = mapped_column(String(48))
    provenance: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=dt.datetime.utcnow)

    plates: Mapped[list["PlateReading"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan")
    confirmations: Mapped[list["Confirmation"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan")


class PlateReading(Base):
    __tablename__ = "plate_readings"
    __table_args__ = (UniqueConstraint("scan_id", "plate",
                                       name="uq_plate_per_scan"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    plate: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))  # ok|candidate|missing
    dx_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    dy_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    x_lo_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    x_hi_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_lo_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_hi_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    n_x_readings: Mapped[int] = mapped_column(Integer)
    n_y_readings: Mapped[int] = mapped_column(Integer)
    notes: Mapped[list] = mapped_column(JSON)

    scan: Mapped[Scan] = relationship(back_populates="plates")


class Confirmation(Base):
    __tablename__ = "confirmations"
    __table_args__ = (Index("ix_confirm_scan", "scan_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    operator_id: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))  # confirmed|corrected|rejected
    corrected_dx_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    corrected_dy_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=dt.datetime.utcnow)

    scan: Mapped[Scan] = relationship(back_populates="confirmations")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
