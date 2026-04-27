import os
import json
import asyncio
import time
import threading
from datetime import datetime
import pytz
from typing import List, Dict, Any, Optional, AsyncGenerator
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor

from financial_tools import (
    get_company_info,
    get_price_history,
    calculate_technical_indicators,
    get_news_and_sentiment,
    get_financial_statements,
    get_brokerage_research_reports,
    get_market_overview,
    compare_stocks,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemma-3-12b-it"
API_VERSION   = "v1beta"

TOOL_MAP = {
    "get_company_info":              get_company_info,
    "get_price_history":             get_price_history,
    "calculate_technical_indicators":calculate_technical_indicators,
    "get_news_and_sentiment":        get_news_and_sentiment,
    "get_financial_statements":      get_financial_statements,
    "get_brokerage_research_reports":get_brokerage_research_reports,
    "get_market_overview":           get_market_overview,
    "compare_stocks":                compare_stocks,
}

# Tool → intent mapping (dùng để fallback khi router không trả intent)
TOOL_INTENT_MAP = {
    "get_company_info":               "company_info",
    "get_price_history":              "price_history",
    "calculate_technical_indicators": "technical",
    "get_news_and_sentiment":         "news",
    "get_financial_statements":       "financials",
    "get_brokerage_research_reports": "research",
    "get_market_overview":            "market",
    "compare_stocks":                 "compare",
}

# ── Helper ────────────────────────────────────────────────────────────────────

def get_now_vn():
    return datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime("%d/%m/%Y %H:%M")

# ── Caching ───────────────────────────────────────────────────────────────────

_tool_cache  = {}
_cache_lock  = threading.Lock()
CACHE_TTL    = 600


def clear_tool_cache():
    with _cache_lock:
        _tool_cache.clear()


def _get_cached(tool_name: str, args: dict) -> Optional[str]:
    key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
    with _cache_lock:
        entry = _tool_cache.get(key)
        if entry and (time.time() - entry["ts"]) < CACHE_TTL:
            return entry["result"]
    return None


def _set_cache(tool_name: str, args: dict, result: str):
    key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
    with _cache_lock:
        _tool_cache[key] = {"result": result, "ts": time.time()}

# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client  = genai.Client(api_key=api_key, http_options={"api_version": API_VERSION})

    async def _call_model_retry(
        self,
        prompt: str,
        config: types.GenerateContentConfig = None,
        retries: int = 3,
    ) -> str:
        for i in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=DEFAULT_MODEL, contents=prompt, config=config
                )
                return response.text
            except Exception as e:
                if "429" in str(e) and i < retries - 1:
                    await asyncio.sleep((i + 1) * 2)
                    continue
                raise e
        return ""

# ── Router Agent ──────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """Bạn là Router Agent. Nhiệm vụ: Phân tích câu hỏi, xác định intent và danh sách tool cần gọi.
Thời gian thực tế hiện tại: {current_time} (HÃY DÙNG NGÀY NÀY, KHÔNG DÙNG DỮ LIỆU CŨ).

Các tool:
- get_company_info: Thông tin cơ bản doanh nghiệp (ngành, sàn, vốn, mô tả, giá hiện tại).
  Args: {{"ticker": "MÃ"}}

- get_price_history: Giá lịch sử OHLCV.
  Args: {{"ticker": "MÃ", "period": "3m"}} hoặc {{"ticker": "MÃ", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}

- calculate_technical_indicators: Chỉ báo kỹ thuật RSI/SMA/EMA/MACD/BB/STOCH/ATR.
  Args bắt buộc: {{"ticker": "MÃ", "indicators": "SMA,RSI,MACD"}}
  Args tuỳ chọn:
    - "indicators": mặc định "SMA,RSI,MACD" - ghi đè khi user chỉ định rõ chỉ báo khác
    - "sma_windows": mặc định "20,50,200" - ghi đè khi user chỉ định rõ, VD: "10,30"
    - "rsi_window": mặc định 14 - ghi đè khi user chỉ định rõ, VD: 21
    - "period": mặc định "1y" - ghi đè khi user chỉ định, VD: "6m"

- get_news_and_sentiment: Tin tức & sentiment. Args: {{"ticker": "MÃ"}}

- get_financial_statements: Báo cáo tài chính.
  Args: {{"ticker": "MÃ", "statement_type": "income|balance|cashflow|ratio"}}

- get_market_overview: Tổng quan thị trường VNIndex/HNX/VN30. Args: {{"period": "1m"}}

- compare_stocks: So sánh nhiều mã. Args: {{"tickers": "MÃ1,MÃ2", "period": "3m"}}

- get_brokerage_research_reports: Báo cáo phân tích từ CTCK. Args: {{"ticker": "MÃ"}}

INTENT (chọn 1):
- "company_info"   : hỏi thông tin doanh nghiệp, giới thiệu, hoạt động, ngành nghề, sàn, vốn
- "price_history"  : hỏi giá, biến động, lịch sử giá
- "technical"      : hỏi chỉ báo kỹ thuật, RSI, SMA, EMA, MACD, Bollinger, tín hiệu kỹ thuật, tín hiệu mua bán, phân tích kỹ thuật
- "news"           : hỏi tin tức, sentiment, sự kiện
- "financials"     : hỏi kết quả kinh doanh, doanh thu, lợi nhuận, BCTC, tài sản, nợ, ROE
- "research"       : hỏi khuyến nghị, báo cáo CTCK, giá mục tiêu
- "market"         : hỏi VNIndex, HNX, thị trường chứng khoán chung
- "compare"        : so sánh nhiều cổ phiếu
- "general"        : chào hỏi, hỏi ngày giờ, câu hỏi chung không liên quan tài chính

QUY TẮC QUAN TRỌNG:
1. "Cho tôi thông tin về VIC" → intent: company_info, tool: get_company_info
2. "FPT hoạt động trong lĩnh vực gì?" → intent: company_info, tool: get_company_info
3. "VNIndex hôm nay?" → intent: market, tool: get_market_overview
4. Chỉ gọi ĐÚNG tool cần thiết, KHÔNG gọi thừa.
5. Chỉ chào hỏi/hỏi ngày tháng → tasks: [], is_general: true, intent: "general"
6. KHI HỎI GIÁ "HÔM NAY" / "HIỆN TẠI": dùng period ngắn ("5d"), KHÔNG dùng start_date=end_date=hôm nay.
7. VỚI calculate_technical_indicators - LUÔN truyền "indicators" trong args:
   - Câu hỏi CHUNG ("chỉ báo kỹ thuật", "tín hiệu kỹ thuật", "phân tích kỹ thuật") → indicators: "SMA,RSI,MACD"
   - Chỉ hỏi RSI → indicators: "RSI"
   - Chỉ hỏi SMA/MA → indicators: "SMA"
   - Hỏi RSI và SMA → indicators: "SMA,RSI"
   - Hỏi MACD, Bollinger → indicators: "MACD,BB"
   - Hỏi tất cả / toàn diện → indicators: "SMA,EMA,RSI,MACD,BB"
   - Window size không chỉ định → không truyền (dùng default)
   - Window size chỉ định rõ ("RSI 21", "SMA 10,50") → truyền rsi_window/sma_windows tương ứng

VÍ DỤ CỤ THỂ:
- "HPG đang có tín hiệu các chỉ báo kỹ thuật như nào?" →
  intent: technical, tasks: [{{"tool":"calculate_technical_indicators","args":{{"ticker":"HPG","indicators":"SMA,RSI,MACD"}}}}]
- "Chỉ báo kỹ thuật VNM?" →
  intent: technical, tasks: [{{"tool":"calculate_technical_indicators","args":{{"ticker":"VNM","indicators":"SMA,RSI,MACD"}}}}]
- "RSI của ACB?" →
  intent: technical, tasks: [{{"tool":"calculate_technical_indicators","args":{{"ticker":"ACB","indicators":"RSI"}}}}]
- "SMA 10, 50 và RSI 21 của FPT?" →
  intent: technical, tasks: [{{"tool":"calculate_technical_indicators","args":{{"ticker":"FPT","indicators":"SMA,RSI","sma_windows":"10,50","rsi_window":21}}}}]

Output JSON:
{{
  "ticker": "MÃ hoặc null",
  "intent": "intent_name",
  "query_summary": "tóm tắt câu hỏi bằng tiếng Việt ngắn gọn (≤10 từ)",
  "tasks": [{{"tool": "tên_tool", "args": {{...}}}}],
  "is_general": false
}}
"""


class RouterAgent(BaseAgent):
    # Từ khóa nhận diện câu hỏi kỹ thuật - dùng làm safety-net khi LLM miss
    _TECHNICAL_KW = {
        "chỉ báo", "kỹ thuật", "rsi", "sma", "ema", "macd", "bollinger",
        "stochastic", "atr", "tín hiệu", "moving average", "trung bình động",
        "phân tích kỹ thuật", "indicator", "oscillator", "momentum",
    }
    # Từ khóa nhận diện tên ticker VN (uppercase 2-5 ký tự)
    _TICKER_RE = __import__("re").compile(r'\b([A-Z]{2,5})\b')

    async def plan(self, query: str) -> Dict[str, Any]:
        prompt = f"{ROUTER_SYSTEM.format(current_time=get_now_vn())}\n\nCâu hỏi: {query}\n\nJSON:"
        try:
            res_text = await self._call_model_retry(
                prompt, config=types.GenerateContentConfig(temperature=0.1)
            )
            if "{" in res_text:
                res_text = res_text[res_text.find("{") : res_text.rfind("}") + 1]
            plan = json.loads(res_text)

            # ── Fallback intent từ tool nếu router không trả về ────────────
            if not plan.get("intent") and plan.get("tasks"):
                first_tool = plan["tasks"][0].get("tool", "")
                plan["intent"] = TOOL_INTENT_MAP.get(first_tool, "general")

            # ── Safety-net: phát hiện câu hỏi kỹ thuật bị router miss ─────
            # Xảy ra khi: is_general=True hoặc tasks=[] nhưng câu hỏi CÓ từ kỹ thuật
            q_lower = query.lower()
            is_technical_query = any(kw in q_lower for kw in self._TECHNICAL_KW)

            if is_technical_query and (plan.get("is_general") or not plan.get("tasks")):
                # Tìm ticker từ câu hỏi (ưu tiên uppercase word)
                ticker = plan.get("ticker")
                if not ticker:
                    matches = self._TICKER_RE.findall(query)
                    # Lọc bỏ stopwords không phải ticker
                    _STOP = {"RSI","SMA","EMA","MACD","ATR","BB","VN","HNX","VNĐ","USD","ETF","IPO"}
                    for m in matches:
                        if m not in _STOP and len(m) >= 2:
                            ticker = m
                            break

                if ticker:
                    # Xác định indicators từ câu hỏi
                    indicators = "SMA,RSI,MACD"  # default toàn diện
                    has_sma  = any(k in q_lower for k in ("sma", "ma ", "moving", "trung bình động"))
                    has_ema  = "ema" in q_lower
                    has_rsi  = "rsi" in q_lower
                    has_macd = "macd" in q_lower
                    has_bb   = any(k in q_lower for k in ("bollinger", " bb ", "bb,", ",bb"))
                    has_stoch= "stoch" in q_lower
                    has_atr  = "atr" in q_lower
                    specific = [k for k, v in {
                        "SMA": has_sma, "EMA": has_ema, "RSI": has_rsi,
                        "MACD": has_macd, "BB": has_bb,
                        "STOCH": has_stoch, "ATR": has_atr,
                    }.items() if v]
                    if specific:
                        indicators = ",".join(specific)

                    plan = {
                        "ticker":       ticker,
                        "intent":       "technical",
                        "query_summary": f"Chỉ báo kỹ thuật {ticker}",
                        "tasks": [{
                            "tool": "calculate_technical_indicators",
                            "args": {"ticker": ticker, "indicators": indicators},
                        }],
                        "is_general": False,
                    }

            return plan
        except Exception:
            return {
                "ticker": None, "intent": "general",
                "query_summary": query[:40],
                "tasks": [], "is_general": True,
            }

# ── Execution Agent ───────────────────────────────────────────────────────────

class ExecutionAgent:
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tasks(self, tasks: List[dict]) -> List[dict]:
        loop    = asyncio.get_event_loop()
        futures = []
        for task in tasks:
            tool_name = task.get("tool")
            args      = task.get("args", {})
            if tool_name in TOOL_MAP:
                futures.append(
                    loop.run_in_executor(self.executor, self._run_tool, tool_name, args)
                )
        return await asyncio.gather(*futures)

    def _run_tool(self, tool_name: str, args: dict) -> dict:
        cached = _get_cached(tool_name, args)
        if cached:
            return {"tool": tool_name, "result": cached, "cached": True}
        try:
            tool_fn = TOOL_MAP[tool_name]
            result  = tool_fn.func(**args) if hasattr(tool_fn, "func") else tool_fn(**args)
            _set_cache(tool_name, args, result)
            return {"tool": tool_name, "result": result, "cached": False}
        except Exception as e:
            return {"tool": tool_name, "error": str(e)}

# ── Advisor Agent ─────────────────────────────────────────────────────────────

ADVISOR_SYSTEM = """Bạn là Advisor. Phân tích dữ liệu tài chính.
Thời gian thực tế hiện tại: {current_time}.
JSON: {{"recommendation": "Bullish|Bearish|Neutral", "reasoning": ["..."]}}
"""


class AdvisorAgent(BaseAgent):
    async def analyze(self, results: List[dict]) -> Dict[str, Any]:
        prompt = (
            f"{ADVISOR_SYSTEM.format(current_time=get_now_vn())}\n"
            f"Dữ liệu: {json.dumps(results, ensure_ascii=False)[:3000]}\n\nJSON:"
        )
        try:
            res_text = await self._call_model_retry(
                prompt, config=types.GenerateContentConfig(temperature=0.2)
            )
            if "{" in res_text:
                res_text = res_text[res_text.find("{") : res_text.rfind("}") + 1]
            return json.loads(res_text)
        except Exception:
            return {"recommendation": "Neutral", "reasoning": ["Lỗi xử lý phân tích."]}

# ── Synthesis Prompts theo Intent ─────────────────────────────────────────────

# Dùng cho: company_info
PROMPT_COMPANY_INFO = """Bạn là trợ lý tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

Dữ liệu nằm trong phần DỮ LIỆU, trong khóa "company_data".

NHIỆM VỤ: Hiển thị ĐÚNG những gì có trong dữ liệu. TUYỆT ĐỐI không tự thêm trường không có trong dữ liệu.

BẢNG TRA TÊN TRƯỜNG → NHÃN TIẾNG VIỆT (chỉ dùng khi trường đó TỒN TẠI trong dữ liệu):
company_name             = Tên công ty
ten_cong_ty              = Tên công ty
business_model           = Mô hình kinh doanh
symbol                   = Mã chứng khoán
founded_date             = Ngày thành lập
charter_capital          = Vốn điều lệ
number_of_employees      = Số lượng nhân viên
listing_date             = Ngày niêm yết
par_value                = Mệnh giá
exchange                 = Sàn giao dịch
listing_price            = Giá niêm yết
listed_volume            = Khối lượng niêm yết
ceo_name                 = Tên CEO
ceo_position             = Vị trí CEO
inspector_name           = Tên kiểm soát viên
inspector_position       = Vị trí kiểm soát viên
establishment_license    = Giấy phép thành lập
business_code            = Mã ngành kinh doanh
tax_id                   = Mã số thuế
auditor                  = Kiểm toán viên
company_type             = Loại hình công ty
address                  = Địa chỉ
phone                    = Điện thoại
fax                      = Fax
email                    = Email
website                  = Website
branches                 = Chi nhánh
history                  = Lịch sử công ty
free_float_percentage    = Tỷ lệ free float
free_float               = Số lượng free float
outstanding_shares       = Số cổ phiếu đang lưu hành
as_of_date               = Ngày cập nhật dữ liệu
id                       = ID công ty
issue_share              = Số cổ phiếu phát hành
company_profile          = Hồ sơ công ty
icb_name3                = Ngành ICB cấp 3
icb_name2                = Ngành ICB cấp 2
icb_name4                = Ngành ICB cấp 4
financial_ratio_issue_share = Tỷ lệ tài chính/cổ phiếu

CÁCH XUẤT KẾT QUẢ - BẮT BUỘC:

**Bước 1:** Đọc kbs_overview (hoặc vci_overview nếu không có KBS) từ company_data.
**Bước 2:** Với MỖI KEY có trong object đó, tra bảng trên để lấy nhãn tiếng Việt, rồi in ra:
  - **<Nhãn tiếng Việt>:** <giá trị>
**Bước 3:** Nếu KEY không có trong bảng tra, bỏ qua.
**Bước 4:** Không in trường nào không có trong dữ liệu. Không in "N/A", "Không có", "Chưa có".

---
QUY TẮC FORMAT SỐ:
- Số lớn: dấu phẩy phân nghìn (1,234,567)
- charter_capital: thêm "tỷ đồng" nếu giá trị > 1000
- Tỷ lệ %: 2 chữ số thập phân (25.34%)
- Giá cổ phiếu (listing_price): thêm "VNĐ"
- free_float_percentage: thêm "%"

TUYỆT ĐỐI KHÔNG bịa thêm bất kỳ trường nào không có trong dữ liệu.
Viết hoàn toàn bằng tiếng Việt.
"""

# Dùng cho: price_history
PROMPT_PRICE_HISTORY = """Bạn là trợ lý tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

Dữ liệu nằm trong phần DỮ LIỆU. Trình bày theo format gạch đầu dòng, tên trường in đậm:

XỬ LÝ KHI THỊ TRƯỜNG ĐÓNG CỬA (ƯU TIÊN CAO NHẤT):
- Nếu dữ liệu có "thi_truong_dong_cua": true HOẶC trường "ghi_chu" chứa "đóng cửa":
  → Bắt đầu câu trả lời bằng: "⚠️ Thị trường đang đóng cửa. Dữ liệu dưới đây là giá đóng cửa phiên giao dịch gần nhất (<ngay_phien_gan_nhat>)."
  → Hiển thị đầy đủ thông tin giá như bình thường (KHÔNG bỏ qua, KHÔNG viết "không có dữ liệu").
- Nếu "trang_thai" = "Realtime" hoặc không có cờ đóng cửa: trình bày bình thường, không cần ghi chú.

---
THÔNG TIN CHÍNH:
- **Mã cổ phiếu:** <ticker>
- **Phiên giao dịch:** <tu_ngay> → <den_ngay>  *(nếu là 1 phiên duy nhất thì chỉ ghi ngày đó)*
- **Khung nến:** <interval>

Giá (đơn vị VNĐ, dùng dấu phẩy phân nghìn):
- **Giá đóng cửa:** <gia_dong_cuoi_ky> VNĐ
- **Giá mở cửa:** <gia_mo_dau_ky> VNĐ  *(nếu có)*
- **Thay đổi so với phiên trước:** <thay_doi_vnd> VNĐ (<thay_doi_pct>%)  *(▲ tăng / ▼ giảm)*
- **Cao nhất phiên/kỳ:** <gia_cao_nhat> VNĐ
- **Thấp nhất phiên/kỳ:** <gia_thap_nhat> VNĐ
- **KLGD trung bình/phiên:** <klgd_tb_phien> cổ phiếu  *(nếu có)*

QUY TẮC FORMAT:
- Giá: dùng dấu phẩy phân nghìn (45,200 VNĐ)
- Tỷ lệ %: 2 chữ số thập phân (+3.25% hoặc -1.50%)
- Thay đổi dương: ký hiệu ▲; thay đổi âm: ký hiệu ▼
- KHÔNG thêm phân tích hay khuyến nghị trừ khi được yêu cầu.
Viết bằng tiếng Việt.
"""


# Dùng cho: technical
PROMPT_TECHNICAL = """Bạn là chuyên gia phân tích kỹ thuật. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

Đọc DỮ LIỆU từ tool calculate_technical_indicators và trình bày theo cấu trúc sau:

**[ticker] - Phân tích kỹ thuật ngày [ngay]**
- **Giá đóng cửa**: [gia_dong_cua] VNĐ

Với MỖI chỉ báo có trong dữ liệu (trường "indicators"), trình bày:

SMA (nếu có indicators.SMA):
- Mỗi chu kỳ 1 dòng: **SMA[chu_ky]**: [value_dong] VNĐ - [signal] ([chenh_lech_pct]%)
- Nếu có Golden/Death Cross → ghi rõ

EMA (nếu có indicators.EMA):
- Mỗi chu kỳ 1 dòng: **EMA[chu_ky]**: [value_dong] VNĐ - [signal]

RSI (nếu có indicators.RSI):
- **RSI([chu_ky])**: [gia_tri] - [signal]
- Lịch sử 5 phiên: [lich_su_5_phien]

MACD (nếu có indicators.MACD):
- **MACD**: [macd] | Signal: [signal_line] | Histogram: [histogram] - [interpretation]

Bollinger Bands (nếu có indicators.BB):
- **BB trên/giữa/dưới**: [dai_tren_dong] / [trung_binh_dong] / [dai_duoi_dong] VNĐ
- %B = [pct_b] | Băng rộng: [bang_rong_pct]% - [signal]

Stochastic (nếu có indicators.Stochastic):
- **Stoch %K/D**: [K] / [D] - [signal]

ATR (nếu có indicators.ATR):
- **ATR(14)**: [value_dong] VNĐ ([atr_pct]%) - [ghi_chu]

**Tín hiệu tổng hợp** (từ trường "signals" trong dữ liệu):
[Liệt kê từng signal]
→ Nhận định chung: Bullish / Bearish / Neutral

QUY TẮC:
- Giá: dùng dấu phẩy phân nghìn (45,200 VNĐ)
- Chỉ hiển thị chỉ báo CÓ trong dữ liệu, bỏ qua nếu không có
- KHÔNG bịa số liệu
Viết bằng tiếng Việt.
"""

# Dùng cho: news / sentiment
PROMPT_NEWS = """Bạn là trợ lý tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

Tóm tắt tin tức & sentiment:
- **Sentiment tổng thể**: Tích cực / Tiêu cực / Trung tính (điểm: ...)
- **Tin nổi bật** (5-7 bài): tên bài, ngày, link bài viết, nhận định ngắn
- **Nhận xét**: 1-2 câu về xu hướng tin tức gần đây

KHÔNG thêm tiêu đề ngoài yêu cầu. Viết bằng tiếng Việt.
"""

# Dùng cho: financials (KQKD, BCTC, tỷ số)
PROMPT_FINANCIALS = """Bạn là chuyên gia tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

Trình bày kết quả tài chính theo đúng loại báo cáo người dùng yêu cầu:
- Tiêu đề: dùng đúng tên loại báo cáo (VD: "Kết quả kinh doanh HPG - Q1/2024")
- Các chỉ tiêu chính theo bảng hoặc danh sách có cấu trúc
- Ghi rõ đơn vị: tỷ đồng cho tiền tệ, % cho tỷ lệ

LƯU Ý:
1. Giá cổ phiếu dùng VNĐ đầy đủ.
2. Ghi rõ kỳ báo cáo (Q?/YYYY hoặc năm YYYY).
3. KHÔNG thêm khuyến nghị mua bán trừ khi người dùng hỏi.
Viết bằng tiếng Việt.
"""

# Dùng cho: research (báo cáo CTCK)
PROMPT_RESEARCH = """Bạn là trợ lý tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

Tổng hợp khuyến nghị từ các công ty chứng khoán:
- **Đồng thuận**: MUA / BÁN / GIỮ (bao nhiêu CTCK đồng thuận)
- **Giá mục tiêu**: range giá mục tiêu từ các CTCK
- **Báo cáo gần nhất** (3-5 báo): CTCK, khuyến nghị, giá mục tiêu, ngày

KHÔNG bịa khuyến nghị. Nếu thiếu dữ liệu, ghi rõ "Chưa có đủ dữ liệu".
Viết bằng tiếng Việt.
"""

# Dùng cho: market (VNIndex, HNX)
PROMPT_MARKET = """Bạn là trợ lý tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

LƯU Ý QUAN TRỌNG: Chỉ số VN-Index, HNX-Index, VN30 tính bằng ĐIỂM (points), KHÔNG phải VNĐ. 
Chỉ trình bày về thị trường mà người dùng hỏi (VD: nếu hỏi về VN-Index thì chỉ trình bày VN-Index, không cần đề cập HNX nếu không có dữ liệu), còn nếu hỏi chung chung về thị trường thì trình bày cả 3 nếu có dữ liệu.

Format trả lời cho từng chỉ số:
- **VN-Index**: ... điểm | thay đổi: ±... điểm (±...%) trong kỳ
- **HNX-Index**: ... điểm | thay đổi: ...
- **VN30**: ... điểm | thay đổi: ...
- Cao nhất / Thấp nhất trong kỳ (ghi rõ khoảng thời gian)
- Nhận xét ngắn về xu hướng

Nếu hôm nay thị trường đóng cửa, ghi rõ "Dữ liệu chốt phiên ngày [Ngày]".
Viết bằng tiếng Việt.
"""

# Dùng cho: compare
PROMPT_COMPARE = """Bạn là trợ lý tài chính. Người dùng hỏi: "{query_summary}"
Thời gian hiện tại: {current_time}.

So sánh hiệu suất các cổ phiếu theo bảng:
| Mã | Giá đầu kỳ | Giá cuối kỳ | Thay đổi (%) | RSI | Biến động |
|-----|-----------|-------------|-------------|-----|-----------|
| ... | ...       | ...         | ...         | ... | ...       |

Ghi rõ kỳ so sánh. Nhận xét ngắn về cổ phiếu tốt nhất / kém nhất trong kỳ.
Giá tính bằng VNĐ đầy đủ. Viết bằng tiếng Việt.
"""

# Dùng cho: general (chào hỏi, câu hỏi không liên quan tài chính)
PROMPT_GENERAL = """Bạn là trợ lý tài chính FinanceGPT VN, thân thiện và hữu ích.
Thời gian hiện tại tại Việt Nam: {current_time}.
Trả lời ngắn gọn, tự nhiên bằng tiếng Việt.
"""

# Map intent → prompt template
INTENT_PROMPT_MAP = {
    "company_info":  PROMPT_COMPANY_INFO,
    "price_history": PROMPT_PRICE_HISTORY,
    "technical":     PROMPT_TECHNICAL,
    "news":          PROMPT_NEWS,
    "financials":    PROMPT_FINANCIALS,
    "research":      PROMPT_RESEARCH,
    "market":        PROMPT_MARKET,
    "compare":       PROMPT_COMPARE,
    "general":       PROMPT_GENERAL,
}

# Nhiệt độ theo intent (company_info cần chính xác → thấp)
INTENT_TEMP_MAP = {
    "company_info":  0.1,
    "price_history": 0.1,
    "technical":     0.15,
    "news":          0.2,
    "financials":    0.15,
    "research":      0.2,
    "market":        0.15,
    "compare":       0.15,
    "general":       0.4,
}

# ── Synthesis Agent ───────────────────────────────────────────────────────────

class SynthesisAgent(BaseAgent):
    async def synthesize(
        self,
        plan: dict,
        raw_data: list,
        advisor_insight: dict,
        original_query: str = "",
    ) -> str:
        intent        = plan.get("intent", "general")
        query_summary = plan.get("query_summary", original_query[:60])
        current_time  = get_now_vn()

        # Chọn prompt theo intent
        prompt_template = INTENT_PROMPT_MAP.get(intent, PROMPT_GENERAL)
        system_prompt   = prompt_template.format(
            query_summary=query_summary,
            current_time=current_time,
        )
        temperature = INTENT_TEMP_MAP.get(intent, 0.2)

        # ── Parse result JSON strings → tránh double-encoding ────────────────
        # raw_data mỗi entry: {"tool": "...", "result": "<json_string>", ...}
        # Nếu để nguyên, json.dumps() sẽ escape result thành string trong string
        # → LLM thấy: "result": "{\"ticker\": ...}" thay vì "result": {"ticker": ...}
        parsed_data = []
        for item in raw_data:
            entry = dict(item)
            if isinstance(entry.get("result"), str):
                try:
                    entry["result"] = json.loads(entry["result"])
                except (json.JSONDecodeError, ValueError):
                    pass  # Giữ nguyên nếu không parse được
            parsed_data.append(entry)

        # Giới hạn data để không vượt context (5000 chars sau khi parse)
        data_str = json.dumps(parsed_data, ensure_ascii=False)[:5000]

        # Yêu cầu định dạng: tùy theo intent
        if intent == "company_info":
            formatting_instr = (
                "ĐỊNH DẠNG BẮT BUỘC - Toàn bộ tiếng Việt:\n"
                "- Mỗi trường là 1 dòng gạch đầu dòng: '- **Tên trường tiếng Việt:** Giá trị'\n"
                "- Giữa các nhóm (Thông tin công ty / Cổ đông lớn / Ban lãnh đạo / Công ty con) "
                "phải có 1 dòng trống (2 ký tự xuống dòng liên tiếp).\n"
                "- Chỉ hiển thị trường có dữ liệu thực; bỏ qua hoàn toàn trường null/rỗng.\n"
                "- Số lớn: dấu phẩy phân nghìn. Vốn điều lệ: tỷ đồng. Tỷ lệ: 2 chữ số thập phân%.\n"
                "- KHÔNG bịa thông tin. KHÔNG thêm phân tích. Viết bằng tiếng Việt."
            )
        elif intent == "technical":
            formatting_instr = (
                "QUAN TRỌNG: Dữ liệu nằm trong parsed_data[0].result (đã là object JSON, không phải string).\n"
                "Đọc trực tiếp các trường: ticker, ngay, gia_dong_cua, indicators, signals.\n"
                "KHÔNG nói 'không có dữ liệu' nếu result object có các trường trên.\n"
                "Trình bày từng chỉ báo trong indicators theo cấu trúc đã hướng dẫn.\n"
                "Viết bằng tiếng Việt."
            )
        else:
            formatting_instr = (
                "Định dạng trả lời: Sử dụng danh sách gạch đầu dòng '-' cho từng mục; mỗi mục trên một dòng; "
                "nếu cần mô tả ngắn, viết 1-2 câu ngay sau dấu gạch đầu dòng. Tránh đoạn văn dài và tiêu đề thừa. "
                "Viết bằng tiếng Việt."
            )

        prompt = (
            f"{system_prompt}\n\n"
            f"=== DỮ LIỆU (parsed_data) ===\n{data_str}\n\n"
            f"=== NHẬN ĐỊNH ===\n{json.dumps(advisor_insight, ensure_ascii=False)}\n\n"
            f"Câu hỏi gốc: {original_query}\n\n"
            f"{formatting_instr}\n\nTrả lời:"
        )

        return await self._call_model_retry(
            prompt, config=types.GenerateContentConfig(temperature=temperature)
        )

# ── Main Orchestrator ─────────────────────────────────────────────────────────

class GolineOrchestrator(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.router    = RouterAgent(api_key)
        self.executor  = ExecutionAgent()
        self.advisor   = AdvisorAgent(api_key)
        self.synthesis = SynthesisAgent(api_key)

    async def chat(self, query: str) -> AsyncGenerator[dict, None]:
        plan = await self.router.plan(query)

        if plan.get("is_general") or not plan.get("tasks"):
            try:
                now       = get_now_vn()
                sys_p     = PROMPT_GENERAL.format(current_time=now)
                resp_text = await self._call_model_retry(
                    f"{sys_p}\n\nNgười dùng: {query}\n\nTrả lời:"
                )
                yield {"type": "text", "text": resp_text}
            except Exception as e:
                yield {"type": "error", "error": f"Lỗi: {str(e)}"}
            yield {"type": "done"}
            return

        yield {"type": "plan", "plan": plan}
        raw_data       = await self.executor.execute_tasks(plan["tasks"])
        advisor_insight = await self.advisor.analyze(raw_data)
        final_answer   = await self.synthesis.synthesize(
            plan, raw_data, advisor_insight, original_query=query
        )
        yield {"type": "text", "text": final_answer}
        yield {"type": "done"}


# ── Singleton ─────────────────────────────────────────────────────────────────

_orchestrator = None
_orch_lock    = threading.Lock()


def get_orchestrator(api_key: str = None) -> GolineOrchestrator:
    global _orchestrator
    with _orch_lock:
        if _orchestrator is None or (api_key and api_key != _orchestrator.api_key):
            _orchestrator = GolineOrchestrator(
                api_key or os.environ.get("GEMINI_API_KEY", "")
            )
    return _orchestrator