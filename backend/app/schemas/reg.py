from pydantic import BaseModel, Field


class CandidateOut(BaseModel):
    plate: str
    centroid_px: tuple[float, float]
    area_px: float
    circularity: float
    color_distance: float
    complete: bool


class PlateOut(BaseModel):
    status: str
    nominal_mm: list[float] | None = None
    measured_mm: list[float] | None = None
    offset_um: list[float] | None = None
    uncertainty_um: list[float] | None = None
    confidence_range_um: float | None = None
    confidence: float
    candidates: list[CandidateOut]
    selected_index: int | None = None
    notes: list[str]


class CalibrationOut(BaseModel):
    angle_deg: float
    angle_uncertainty_deg: float
    scale_px_per_mm: float
    measured_dpi: float
    translation_px: list[float]
    residual_mm: float
    angle_method: str
    scale_uncertainty_percent: float
    method: str
    meta: dict
    usable: bool


class AnalysisOut(BaseModel):
    spec_version: str
    algo_version: str
    analyzed_at: str
    status: str
    spec_version_: str = Field(default="REG-TARGET/1", alias="spec_version_check")
    calibration: CalibrationOut | None = None
    plates: dict[str, PlateOut]
    reference_plate: str
    declared_dpi: float | None = None
    dpi_source: str
    warnings: list[dict]
    error: dict | None = None


class ConfirmationIn(BaseModel):
    plate: str
    decision: str  # confirmed / corrected / rejected
    offset_x_um: float | None = None
    offset_y_um: float | None = None
    uncertainty_x_um: float | None = None
    uncertainty_y_um: float | None = None
    comment: str | None = None
    reviewer: str = "operator"


class ConfirmationOut(ConfirmationIn):
    id: int
    scan_id: int
    created_at: str

    class Config:
        from_attributes = True


class ModelSuggestionIn(BaseModel):
    plate: str
    suggested_x_um: float | None = None
    suggested_y_um: float | None = None
    suggested_range_um: float | None = None
    rationale: str | None = None


class ModelSuggestionOut(BaseModel):
    id: int
    scan_id: int
    plate: str
    model_name: str
    model_version: str
    suggested_x_um: float | None
    suggested_y_um: float | None
    suggested_range_um: float | None
    status: str
    rationale: str | None

    class Config:
        from_attributes = True
