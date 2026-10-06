# PRT-1 印刷套准色标质检（离线分析，不控制设备）

从样张上的 **PRT-1 合成套准色标** 判断 C/M/Y/K 各色版相对黑版的位移。
纸张旋转/尺度校正与色版偏移**分开求解**；缺色、污点、标记残缺时输出
**候选值 + 95% 可信范围**，不从一个模糊点给出精确微米值。

## 技术栈

| 层 | 技术 | 作用 |
|---|---|---|
| 视觉定位 | Python + OpenCV（HSV 分色、相似度变换、霍夫/质心） | 色标点定位、纸张校正、色版位移估计 |
| 服务 | FastAPI + SQLAlchemy | 上传分析、证据叠层、人工确认、离线复核 |
| 存储 | PostgreSQL（测试用 SQLite） | 扫描尺度、色版读数、人工确认、来源 |
| 前端 | React + Canvas/SVG | 样张展示、局部放大、测量叠层、确认表单 |
| 复核 | 独立的简单臂轮廓中值模型 | **离线建议**，与 OpenCV 结果交叉核对 |

> 本服务没有任何设备控制接口；模型只做离线复核
>（`mode=offline_advisory_only`，`control_authority=none`）。

## 色标格式（PRT-1，仅限随项目提供的合成样张）

- 5×4 个标记，间距 8 mm；每个标记：
  - **黑 K**：中心圆环（基准中心）；
  - **C/M/Y**：位于 0°/90°/180° 轨道上的小十字（空间分色，避免同心叠印变黑）；
- 四角 3 mm 黑色实心圆点 = 纸张定位点；
- 30 mm 校准尺（1 mm 小刻度、5 mm 大刻度、端部括号）= 独立尺度来源。

格式常量全部在 `backend/app/markspec.py`，检测器只认这一种明确格式。

## 算法管线（每一步都写入 `provenance`）

1. **HSV 分色**得到 C/M/Y/K 墨色掩膜（黑色 = 低饱和且低明度）。
2. **四角定位点**（最大实心圆 + 半径簇 + 矩形/内角点几何约束）→
   纸张相似度变换（旋转、均匀尺度、平移）。
3. 透视校正到 300 PPI 规范坐标系；**校准尺 30 mm 括号跨度**（亚像素）
   为主尺度、1 mm 刻度为交叉校验（>3% 不一致则降级为候选）。
4. 每个标记各读各的色：黑环用椭圆拟合取中心；彩色小十字用带内墨色
   加权质心定位，横臂管 x、竖臂管 y（残一臂仍有一个轴的读数）。
5. **扣除四色共有的残差相似度（旋转/尺度/平移）后**再估计各色版平移——
   整张图歪斜由此被分离，不可能被报成色版失准。
6. 输出候选值与 95% 区间（综合像素量化、读数 MAD、纸张残差）。
   - 读点不足/残缺/污点高 → `candidate`；
   - 整版缺墨 → `missing`，数值为 **null 而不是 0 µm**；
   - 旋转 > ±8° 或定位点 <3 → 拒绝输出色版结论。

## 验算（`backend/tests/test_detect.py`，真值来自合成器 sidecar JSON）

用**已知平移、扫描旋转和分辨率变化**生成样张，再反算：

- 参考样张：四版位移 ≈ 0；
- 已知偏移：干净图误差 ≤30 µm，扫描图 ≤45 µm，真值全部落在 95% 区间；
- **纯纸张变换**（2.3° + 260 PPI、色版零偏移）：四版仍在 ±45 µm 内；
- 240/260/306 PPI 扫描：分辨率误差 <2 PPI，旋转误差 <0.1°；
- 缺色样张：返回 `missing` + null；缺陷样张：返回候选与放宽区间；
- 12° 旋转样张：被旋转护栏拒绝；
- 6 组随机平移/旋转/PPI 参数网格回归。

校准尺读数、定位点像素坐标、检测器版本、每一步变换都可通过
`GET /api/runs/{run_id}` 的 `provenance` 字段回溯。

## 运行

```bash
# 1. PostgreSQL（可选；也可用 SQLite）
docker compose up -d postgres
export PRT1_DATABASE_URL="postgresql+psycopg2://prt1:prt1@localhost:5432/prt1"

# 2. 后端
cd backend
python3 -m pip install -r requirements.txt
python3 -m app.synth generate        # 重新生成 data/ 下的合成样张与真值
python3 -m pytest                    # 17 个测试
python3 -m uvicorn app.main:app --reload --port 8000

# 3. 前端
cd ../frontend
npm install && npm run dev           # http://localhost:5173
```

## API 摘要

- `GET  /api/health`
- `GET  /api/datasets` — 随项目提供的样张及真值
- `POST /api/analyze` — `multipart` 上传或 `dataset=<名称>`（可附 `declared_ppi`）
- `GET  /api/runs/{id}` — 完整结果、读数、provenance、人工确认历史
- `GET  /api/runs/{id}/overlay.png` — 校正图 + 证据/测量叠层
- `POST /api/runs/{id}/review` — 离线模型复核（仅建议）
- `POST /api/runs/{id}/confirm` — 人工 confirmed/corrected/rejected

机器读数与人工修正**分别存表**（`plate_readings` vs `confirmations`），
修正不会覆盖原始检测值。

## 前端要点

- 纸张校正（旋转角、扫描 PPI、尺度、残差共同项）独立面板展示，
  与色版位移视觉分离；
- Canvas 上可缩放、点选标记、按光标读取校正坐标系下的 µm；
- 局部放大窗含 0.5 mm 比例尺和每个色版本点读数/σ；
- 色版表显示候选值、95% 区间、读点数、置信度和降级原因；
- 缺色显示红色 `missing` 且无数值；候选显示黄色并提示需人工判定。

## 明确不做的事

- 不识别项目外任意格式的标记/真实机台输出（只有 `backend/data/` 的合成图）；
- 不把模糊/残缺标记报成精确微米值；
- 不用模型结果闭环控制任何印刷设备。
