from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .models import init_db

app = FastAPI(
    title="印刷套准色标离线复核",
    description=(
        "只识别项目提供的合成/扫描 REG-TARGET/1 色标；校准（旋转/DPI/平移）"
        "与各色版偏移严格分离；缺色/污点/残缺返回候选与可信范围；"
        "模型仅离线复核、不控制任何印刷设备。"),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/")
def root():
    return {"service": "reg-review", "docs": "/docs",
            "boundary": "offline_review_only"}
