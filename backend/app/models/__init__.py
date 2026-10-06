import datetime as dt

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)

from ..core.config import settings

url = settings.database_url
connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
engine = create_engine(url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


class Scan(Base):
    """一次扫描分析会话。校准（纸张级变换）与色版偏移分开存。"""
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(String(400))
    declared_dpi: Mapped[float | None] = mapped_column(Float, nullable=True)
    dpi_source: Mapped[str] = mapped_column(String(20), default="none")
    status: Mapped[str] = mapped_column(String(40), default="new")

    angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle_uncertainty_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    measured_dpi: Mapped[float | None] = mapped_column(Float, nullable=True)
    scale_px_per_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    scale_uncertainty_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_residual_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_usable: Mapped[bool] = mapped_column(default=False)
    calibration_method: Mapped[str | None] = mapped_column(String(200), nullable=True)

    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    plates: Mapped[list["PlateResult"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan")
    confirmations: Mapped[list["Confirmation"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan")
    model_suggestions: Mapped[list["ModelSuggestion"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan")


class PlateResult(Base):
    """单个色版（C/M/Y/K）的测量/缺失结果。K 为参照版恒为 0。"""
    __tablename__ = "plate_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    plate: Mapped[str] = mapped_column(String(4))
    status: Mapped[str] = mapped_column(String(30))
    offset_x_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    offset_y_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_x_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_y_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_range_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    candidates_json: Mapped[list] = mapped_column(JSON, default=list)
    notes_json: Mapped[list] = mapped_column(JSON, default=list)

    scan: Mapped[Scan] = relationship(back_populates="plates")


class Confirmation(Base):
    """人工确认记录。人工确认不回写算法结果，只做裁决。"""
    __tablename__ = "confirmations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    plate: Mapped[str] = mapped_column(String(4))
    decision: Mapped[str] = mapped_column(String(20))  # confirmed/corrected/rejected
    offset_x_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    offset_y_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_x_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_y_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(100), default="operator")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    scan: Mapped[Scan] = relationship(back_populates="confirmations")


class ModelSuggestion(Base):
    """离线模型的复核建议（pending_review）。系统不自动采纳、不控制设备。"""
    __tablename__ = "model_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    plate: Mapped[str] = mapped_column(String(4))
    model_name: Mapped[str] = mapped_column(String(100), default="offline_review")
    model_version: Mapped[str] = mapped_column(String(40), default="offline-v1")
    suggested_x_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_y_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_range_um: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending_review")
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    reviewed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    scan: Mapped[Scan] = relationship(back_populates="model_suggestions")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
