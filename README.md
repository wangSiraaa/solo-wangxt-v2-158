# 印刷套准色标位移复核系统（REG-TARGET/1）

质量人员从样张扫描图判断各色版（C/M/Y/K）相对黑色参照版的位移。核心约束：

- **纸张旋转、扫描尺度（DPI）校正与各色版偏移严格分开**：只用黑色校准要素
  （方框 + 两个非对称角点 pip + 两个 K 圆点）估纸张变换，因此整图歪斜不会被
  误判成"某色版失准"。
- **只接受随项目提供的合成色标和扫描图**，不识别任意图像。
- **缺色、污点、标记不完整时返回候选列表 + 可信范围（µm），标注 incomplete/
  ambiguous/missing，绝不从单个模糊点输出精确微米值。**
- **离线模型只做复核建议（pending_review），不自动回写、不控制任何印刷设备。**
- 校准尺检测方式、DPI 来源、算法版本、人工确认全程入库可查。

## 目录

```
docs/target-spec.md       色标格式规范（检测器与生成器共用 spec.py）
backend/                  FastAPI + OpenCV + SQLAlchemy（PG/SQLite）
  app/cv/                 spec / 几何 / K校准 / 色版圆点 / 主编排
  app/api,models,...      API、PostgreSQL 持久化、人工确认、离线复核
  scripts/generate_fixtures.py  合成夹具与 ground truth
  data/fixtures/          10 个合成扫描案例 + manifest
  tests/                  28 项验算（已知平移/旋转/DPI、缺色、污点、残缺）
frontend/                 React + Canvas：局部放大、测量叠层、候选/范围、人工确认
docker-compose.yml        Postgres + 后端（无 PG 时后端自动回退 SQLite）
```

## 快速开始

```bash
# 后端（无 Postgres/Venv 时）
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/generate_fixtures.py     # 生成合成色标+ground truth
.venv/bin/python -m pytest                         # 用已知变换验算
.venv/bin/uvicorn app.main:app --reload           # http://localhost:8000/docs

# 前端
cd frontend && npm install && npm run dev         # http://localhost:5173

# 或：PostgreSQL + 后端
docker compose up --build
```

## 色标几何（详见 docs/target-spec.md）

- 黑色校准：9.4mm 中心线边长的方框；两个非对称角点方块 P0(+3.85,+3.85)、
  P1(−3.85,−3.85)（确定方向，消除 180°/镜像歧义）；K 圆点 dK(1.75,0)、
  dK1(3.5,0)（沿 x 轴，用于自校验）。
- 色版圆点（直径 0.55mm）排在 x 轴上，远离 pip 对角线，避免与角标相撞：
  C(−3.5,0)、M(−1.75,0)、Y(0,0)、K(1.75,0)、K1(3.5,0)。

## 检测管线（`app/cv/`）

1. `calibrate.py`：Canny + 概率霍夫 → 线段共线拼接 → 两正交方向聚类
   → 覆盖/等距约束装配矩形（排除"污点碰巧连成假框"）→ 暗度剖面联合精修
   中心与边长；pip/K 圆点用 Lab 软暗度质心；离散假设枚举消歧方向。
   **全程只用 K 油墨，不含 C/M/Y 位移信息。**
2. `detect_dots.py`：Lab 油墨近邻分类 + 软隶属质心 + 圆度/面积过滤，
   输出**候选列表**（颜色距离、完整度、bbox），缺/残时降级而非丢弃。
3. `detect_target.py`：纸张变换逆算各点到印刷坐标，报偏移（µm）与逐分量
   不确定度（亚像素质心、校准残差、尺度杠杆），以及状态/来源/告警。

## 测量诚实性（重要设计）

- 状态机：`measured / incomplete / ambiguous / missing / low_confidence /
  reference`。找不到点时 `offset_um=null`，只给搜索半径并注明"不是测量精度"。
- 不确定度必须覆盖测量误差（测试据此断言），不允许报自相矛盾的高精度。
- 校准残差超门限 → `calibration_poor`，偏移仅作参考。
- 声明 DPI 与算法尺度偏差 >2% → `dpi_mismatch` 告警；算法尺度始终单独保留。

## 安全边界

- 没有任何指向印刷机控制的接口；`/health` 显式声明 `offline_review_only`。
- 人工确认、模型建议都**不回写**算法原始结果（测试验证）。
- 模型建议只能被人工"部分采纳/驳回"，系统不会自动改色版结果。

## 验算案例（ground truth）

| 案例 | 变换/缺陷 |
|---|---|
| 01 baseline | 无平移、200DPI（整图歪斜≠色版偏移）|
| 02 translations | C/M/Y 已知平移 |
| 03/04 rotated | +2.5° / −3.2° 纸张旋转 |
| 05/06 dpi | 300 / 150 DPI 分辨率变化 |
| 07 missing_c | C 整版缺失 |
| 08 spots | 暗/彩色污点 |
| 09 occluded_m | M 点局部残缺 |
| 10 all_defects | 缺 M + 污点 + 残缺 + 240DPI + JPEG |

## API 摘要

- `POST /api/scans/analyze`（multipart：image、可选 declared_dpi/dpi_source）
- `POST /api/scans/{id}/confirmations`（人工确认/修正/驳回，不回写算法）
- `GET  /api/scans/{id}/model-suggestions`
- `POST /api/scans/{id}/model-suggestions/{sid}/review`
- `POST /api/scans/overlay`（即时 RGBA 测量叠层 PNG）
