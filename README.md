# FinanceGPT VN — Hệ thống Multi-Agent AI Phân tích Chứng khoán

> **Nền tảng AI phân tích chứng khoán Việt Nam theo thời gian thực**, được xây dựng trên kiến trúc Multi-Agent với Gemini LLM, cung cấp dữ liệu từ TCBS, VNDirect, FireAnt, SSI và vnstock.

Tác giả: Phan Hoàng Dũng

---

## Mục lục

- [Tổng quan hệ thống](#tổng-quan-hệ-thống)
- [Kiến trúc Multi-Agent](#kiến-trúc-multi-agent)
- [Cấu trúc file](#cấu-trúc-file)
- [Chức năng chi tiết](#chức-năng-chi-tiết)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Nguồn dữ liệu](#Nguồn-dữ-liệu)

---

## Tổng quan hệ thống

FinanceGPT VN là chatbot tài chính dạng **streaming SSE** (Server-Sent Events), cho phép người dùng đặt câu hỏi bằng tiếng Việt về bất kỳ mã cổ phiếu nào trên sàn HOSE/HNX/UPCOM và nhận phân tích chuyên sâu theo thời gian thực.

**Điểm nổi bật:**
- Phân tích kỹ thuật với 7 chỉ báo: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR
- Dữ liệu tài chính từ nhiều nguồn với cơ chế fallback tự động
- Sentiment analysis bằng mô hình ViSoBERT (Vietnamese NLP)
- Giao diện chat thời gian thực với lịch sử cuộc trò chuyện
- Hỗ trợ tiếng Việt hoàn toàn

---

## Kiến trúc Multi-Agent

Hệ thống bao gồm **4 agent chuyên biệt** phối hợp theo pipeline bất đồng bộ:

```
┌─────────────────────────────────────────────────────────┐
│                  GolineOrchestrator                      │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐                    │
│  │ RouterAgent  │───▶│ExecutionAgent│                    │
│  │             │    │              │                    │
│  │ • Phân tích │    │ • Gọi tool   │                    │
│  │   intent    │    │   song song  │                    │
│  │ • Lập kế    │    │ • Cache 10   │                    │
│  │   hoạch     │    │   phút       │                    │
│  └─────────────┘    └──────┬───────┘                    │
│                             │                            │
│  ┌─────────────┐    ┌──────▼───────┐                    │
│  │SynthesisAgent│◀──│ AdvisorAgent │                    │
│  │             │    │              │                    │
│  │ • Định dạng │    │ • Đánh giá   │                    │
│  │   câu trả   │    │   Bullish /  │                    │
│  │   lời theo  │    │   Bearish /  │                    │
│  │   intent    │    │   Neutral    │                    │
│  └─────────────┘    └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### Luồng xử lý một câu hỏi

```
1. RouterAgent.plan()        → Xác định intent + danh sách tool cần gọi
2. ExecutionAgent.execute()  → Gọi tool song song (asyncio + ThreadPool)
3. AdvisorAgent.analyze()    → Đánh giá tổng hợp dữ liệu thô
4. SynthesisAgent.synthesize() → Tạo câu trả lời theo đúng intent
5. SSE Stream                → Gửi về client từng chunk
```

## Cấu trúc file

```
project/
├── templates/chat.html    # Giao diện chat (template Flask)
├── chat_app.py            # Flask web server + API endpoints
├── agent_orchestrator.py  # Multi-agent pipeline (Router/Executor/Advisor/Synthesis)
├── financial_tools.py     # 11 LangChain tools thu thập dữ liệu tài chính
├── data_sources.py        # Lớp data access: TCBS, VNDirect, SSI, FireAnt
├── realtime_loader.py     # Loader OHLCV realtime qua vnstock + TCBS API
├── sentiment_agent.py     # Phân tích sentiment tiếng Việt bằng ViSoBERT
└── requirements.txt       # Danh sách thư viện Python
```

---

## Chức năng chi tiết

### `chat_app.py` — Flask Web Server

Entry point của ứng dụng. Xử lý:
- **`GET /`** — Trả về giao diện chat HTML
- **`POST /api/chat`** — Nhận câu hỏi, chạy pipeline, stream kết quả qua SSE
- **`GET/POST /api/conversations/*`** — Quản lý lịch sử hội thoại (lưu, tải, đổi tên)
- **`POST /api/key`** — Cấu hình Gemini API key
- **`GET /api/status`** — Kiểm tra trạng thái hệ thống
- **`GET /api/suggestions`** — Danh sách câu hỏi gợi ý theo danh mục
- Session management in-memory với TTL 60 phút, giữ tối đa 24 tin nhắn/session

### `agent_orchestrator.py` — Điều phối Multi-Agent

**RouterAgent:** Dùng Gemini LLM (`gemma-3-12b-it`) với `temperature=0.1` để phân tích câu hỏi → JSON chứa `intent`, `tasks[]`, `ticker`. Có safety-net phát hiện câu hỏi kỹ thuật bị router miss thông qua keyword matching.

**ExecutionAgent:** Chạy song song các tool bằng `asyncio.gather()` + `ThreadPoolExecutor(max_workers=5)`. Cache kết quả tool trong 10 phút để tránh gọi API lặp lại.

**AdvisorAgent:** Đánh giá nhanh dữ liệu thô → `Bullish / Bearish / Neutral` với reasoning.

**SynthesisAgent:** Chọn prompt template theo intent (9 template khác nhau), format câu trả lời phù hợp với từng loại dữ liệu. Tự parse JSON string trong result trước khi đưa vào prompt để tránh double-encoding.

### `financial_tools.py` — 11 LangChain Tools

Toàn bộ tool dùng decorator `@tool` của LangChain, hỗ trợ function calling:

| Tool | Mô tả | Nguồn dữ liệu |
|------|--------|---------------|
| `get_company_info` | Thông tin đầy đủ công ty, giá hiện tại, cổ đông, lãnh đạo | TCBS, VNDirect, FireAnt, SSI, vnstock |
| `get_price_history` | OHLCV lịch sử, hỗ trợ nhiều interval và period | vnstock (KBS→VCI→realtime) |
| `calculate_technical_indicators` | SMA/EMA/RSI/MACD/BB/Stoch/ATR | realtime_loader |
| `get_news_and_sentiment` | Tin tức + điểm sentiment ViSoBERT | CafeF |
| `get_financial_statements` | KQKD/CDKT/LCTT/Tỷ số tài chính | vnstock (KBS→VCI→TCBS Direct API) |
| `get_brokerage_research_reports` | Báo cáo CTCK, khuyến nghị, giá mục tiêu | CafeF, VNDirect, SSI, DNSE, MBS |
| `compare_stocks` | So sánh hiệu suất nhiều mã cùng kỳ | realtime_loader |
| `get_market_overview` | VN-Index, HNX-Index, VN30 | realtime_loader |
| `screen_stocks` | Lọc cổ phiếu theo % tăng/giảm | realtime_loader |
| `get_valuation_metrics` | P/E, P/B, ROE, EPS, lịch sử 4 quý, cổ tức | TCBS, VNDirect, FireAnt |
| `get_comprehensive_analysis` | Phân tích toàn diện: tài chính + định giá + broker | TCBS, VNDirect |

### `data_sources.py` — Lớp Data Access

Module tập trung toàn bộ logic gọi API tài chính bên ngoài:

- **TCBS API** (`apipubaws.tcbs.com.vn`): KQKD, CDKT, LCTT, tỷ số tài chính, thông tin công ty
- **VNDirect finfo** (`api-finfo.vndirect.com.vn/v4`): Ratios, khuyến nghị analyst, cổ đông, cổ tức
- **SSI iBoard** (`iboard-query.ssi.com.vn/v2`): Dữ liệu tài chính, báo cáo nghiên cứu
- **FireAnt** (`restv2.fireant.vn`): Snapshot, profile công ty, chỉ số cơ bản
- **CafeF** (scraping): Bài phân tích chứng khoán, báo cáo CTCK
- **vnstock** (fallback): Kết hợp nhiều nguồn qua thư viện Python

Cơ chế fallback tự động: TCBS → VNDirect → FireAnt → SSI → vnstock

### `realtime_loader.py` — OHLCV Loader

- Tải dữ liệu lịch sử OHLCV qua vnstock với thứ tự nguồn: `VCI → TCBS → KBS → FMP`
- Cache in-memory TTL 5 phút
- Chuẩn hóa cột tự động (xử lý camelCase, snake_case, tên tiếng Việt)
- Lấy thông tin giá realtime từ TCBS stock-info API

### `sentiment_agent.py` — Phân tích Sentiment Tiếng Việt

- **Thu thập**: Scrape bài viết từ CafeF theo mã cổ phiếu (tối đa 30 bài, 4 trang)
- **Mô hình**: `5CD-AI/Vietnamese-Sentiment-visobert` (HuggingFace) — load lazy khi cần
- **Fallback**: Lexicon-based scoring khi không tải được ViSoBERT
- **Lọc bài**: Kiểm tra URL + keyword công ty để loại bài không liên quan
- **Công ty liên quan**: Dùng LLM trích xuất mã cổ phiếu từ bài báo
- Output: `positive / neutral / negative` + điểm số từ -1.0 đến +1.0

### `chat.html` — Giao diện Chat

Single-page app thuần HTML/CSS/JS (không framework):
- Dark theme với color palette chuyên nghiệp
- Sidebar trái: lịch sử cuộc trò chuyện (lưu, tải, đổi tên)
- Sidebar phải: câu hỏi gợi ý theo danh mục
- Streaming SSE: hiển thị từng chunk text với typing cursor
- Markdown rendering bằng `marked.js`
- Ticker strip trên topbar, status indicator API key

---

## Yêu cầu hệ thống

- **Python** 3.10+
- **Gemini API Key** (Google AI Studio): [https://aistudio.google.com](https://aistudio.google.com)
- RAM: 4GB+ (8GB+ nếu dùng ViSoBERT)
- Kết nối internet để gọi API dữ liệu tài chính

---

## Cài đặt

### 1. Clone và tạo môi trường ảo

```bash
git clone <repo_url>
cd financegpt-vn

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý về ViSoBERT:** Mô hình sentiment (~400MB) sẽ tự tải về lần đầu chạy. Nếu không cần sentiment analysis, hệ thống vẫn hoạt động với lexicon fallback.

### 3. Cấu hình API Key

**Cách 1 — Biến môi trường (khuyến nghị):**

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

**Cách 2 — Qua giao diện web:**

Truy cập ứng dụng → click "Cấu hình API key" → dán key → "Kết nối Hệ thống"

### 4. Khởi động server

```bash
python chat_app.py
```

Truy cập: [http://127.0.0.1:5001](http://127.0.0.1:5001)

---

## Nguồn dữ liệu

| Nguồn | Loại dữ liệu | Ghi chú |
|-------|-------------|---------|
| **TCBS** | KQKD, CDKT, LCTT, tỷ số, giá realtime | Nguồn chính, độ tin cậy cao |
| **VNDirect finfo** | Analyst recs, cổ đông, cổ tức, ratios | Bổ sung và cross-check |
| **FireAnt** | Profile, snapshot, market cap | Fallback cho thông tin công ty |
| **SSI iBoard** | BCTC, báo cáo nghiên cứu | Nguồn dự phòng |
| **vnstock** | OHLCV lịch sử (VCI/KBS/TCBS) | Wrapper open-source |
| **CafeF** | Tin tức, bài phân tích CTCK | Scraping HTML |

---

## Lưu ý

- Dữ liệu chỉ mang tính **tham khảo**, không phải khuyến nghị đầu tư
- ViSoBERT cần ~400MB RAM/VRAM, có thể tắt nếu cần nhẹ hơn
- API TCBS có giới hạn request; cache giúp giảm tải đáng kể
- Một số endpoint CafeF/VNDirect có thể thay đổi cấu trúc theo thời gian
