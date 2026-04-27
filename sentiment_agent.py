import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.messages import HumanMessage, SystemMessage

# ── Constants ──────────────────────────────────────────────────────────────────

CAFEF_BASE_URL = "https://cafef.vn"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Referer":         "https://cafef.vn/",
    "Cache-Control":   "max-age=0",
}

MAX_ARTICLES         = 30
MAX_RELATED_ARTICLES = 10
MAX_PAGES            = 4
BATCH_SIZE           = 20
TOP_RELATED          = 10
REQUEST_TIMEOUT      = 15
REQUEST_DELAY        = 1.0
CONTENT_MAX_CHARS    = 1200

SENTIMENT_SCORES = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

# ── Lazy-load ViSoBERT ─────────────────────────────────────────────────────────

_visobert_pipeline = None
_visobert_load_err: Optional[str] = None


def _load_visobert():
    global _visobert_pipeline, _visobert_load_err
    if _visobert_pipeline is not None:
        return _visobert_pipeline
    if _visobert_load_err:
        return None
    try:
        import torch
        from transformers import BertTokenizer, AutoModelForSequenceClassification

        MODEL_NAME = "5CD-AI/Vietnamese-Sentiment-visobert"
        print("[SentimentAgent] Đang tải ViSoBERT (slow tokenizer)...")

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        id2label = model.config.id2label

        def _predict(text: str):
            inputs = tokenizer(
                text[:512],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            probs    = torch.softmax(logits, dim=-1)[0]
            best_idx = int(probs.argmax())
            return [{"label": id2label[best_idx], "score": float(probs[best_idx])}]

        _visobert_pipeline = _predict
        print("[SentimentAgent] ViSoBERT đã tải xong ✓")
        return _visobert_pipeline
    except Exception as e:
        _visobert_load_err = str(e)
        print(f"[SentimentAgent] Không tải được ViSoBERT: {e}")
        return None


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            print(f"[CafeF] HTTP {resp.status_code}: {url}")
            if resp.status_code in (403, 404):
                return None
        except requests.RequestException as e:
            print(f"[CafeF] Request error attempt {attempt+1}: {e}")
        if attempt < retries - 1:
            time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


# ── CafeF listing page parser ──────────────────────────────────────────────────

def _parse_listing_page(html: str) -> List[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    for h3 in soup.find_all("h3"):
        a_tag = h3.find("a", href=True)
        if not a_tag:
            continue

        href = a_tag.get("href", "").strip()

        if not href.endswith(".chn"):
            continue
        if not re.search(r'\d{6,}', href):
            continue

        if href.startswith("/"):
            href = CAFEF_BASE_URL + href
        elif not href.startswith("http"):
            href = CAFEF_BASE_URL + "/" + href

        title = a_tag.get("title", "") or a_tag.get_text(strip=True)
        title = title.strip()
        if len(title) < 10:
            continue

        date_str = ""
        parent = h3.parent
        if parent:
            text_parent = parent.get_text(" ", strip=True)
            date_match = re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', text_parent)
            if date_match:
                date_str = date_match.group(0)

        snippet = ""
        if parent:
            for p_tag in parent.find_all("p"):
                t = p_tag.get_text(strip=True)
                if len(t) > 30 and title[:15].lower() not in t.lower()[:50]:
                    snippet = t[:300]
                    break

        articles.append({
            "title":   title,
            "url":     href,
            "snippet": snippet,
            "date":    date_str,
        })

    return articles


# ── Article content fetcher ────────────────────────────────────────────────────

def _fetch_article_content(url: str) -> str:
    resp = _get(url)
    if resp is None:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "button", "iframe", "figure"]):
            tag.decompose()

        content_el = (
            soup.find("div", class_=re.compile(
                r"detail-content|article-content|content-detail|"
                r"maincontent|entry-content|post-content", re.I))
            or soup.find("div", attrs={"id": re.compile(
                r"content|article|detail|main", re.I)})
            or soup.find("article")
        )

        if content_el:
            h2 = content_el.find("h2")
            sapo = (h2.get_text(" ", strip=True) + " ") if h2 else ""
            paras = [
                p.get_text(" ", strip=True)
                for p in content_el.find_all("p")
                if len(p.get_text(strip=True)) > 20
            ]
            text = sapo + " ".join(paras)
        else:
            paras = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 30
            ]
            text = " ".join(paras)

        text = re.sub(r'\s+', ' ', text).strip()
        return text[:CONTENT_MAX_CHARS]
    except Exception as e:
        print(f"[CafeF] Parse error {url}: {e}")
        return ""


# ── Main article collector ─────────────────────────────────────────────────────

def _collect_articles(ticker: str, max_articles: int = MAX_ARTICLES) -> List[dict]:
    ticker_lower = ticker.lower()

    url_patterns = [
        ticker_lower,
        f"co-phieu-{ticker_lower}",
    ]

    all_articles: List[dict] = []
    seen_urls:    set         = set()

    for pattern in url_patterns:
        if len(all_articles) >= max_articles:
            break

        pattern_found = False
        for page in range(1, MAX_PAGES + 1):
            if len(all_articles) >= max_articles:
                break

            url = (
                f"{CAFEF_BASE_URL}/{pattern}.html"
                if page == 1
                else f"{CAFEF_BASE_URL}/{pattern}-p{page}.html"
            )

            print(f"[CafeF] GET {url}")
            resp = _get(url)
            if resp is None:
                break

            if "Không tìm thấy" in resp.text[:500] or len(resp.text) < 3000:
                print(f"[CafeF] Trang không hợp lệ hoặc 404: {url}")
                break

            page_arts = _parse_listing_page(resp.text)
            if not page_arts:
                print(f"[CafeF] Không parse được bài từ {url}")
                break

            pattern_found = True
            new_count = 0
            for art in page_arts:
                if art["url"] not in seen_urls and len(all_articles) < max_articles:
                    seen_urls.add(art["url"])
                    all_articles.append(art)
                    new_count += 1

            print(f"[CafeF] Trang {page}: +{new_count} bài → tổng {len(all_articles)}")
            if new_count == 0:
                break

            time.sleep(REQUEST_DELAY)

        if pattern_found and all_articles:
            break

    enriched = 0
    for art in all_articles:
        combined = (art.get("title", "") + " " + art.get("snippet", "")).strip()
        if len(combined) < 80 and art.get("url") and enriched < 10:
            print(f"[CafeF] Fetch content: {art['url'][:70]}...")
            content = _fetch_article_content(art["url"])
            if content:
                art["snippet"] = content[:CONTENT_MAX_CHARS]
                enriched += 1
            time.sleep(REQUEST_DELAY * 0.5)

    print(f"[CafeF] ✓ Tổng {len(all_articles)} bài cho '{ticker}'")
    return all_articles


# ── Sentiment scoring ──────────────────────────────────────────────────────────

def _score_text(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"label": "neutral", "confidence": 0.5, "numeric_score": 0.0}

    pipe = _load_visobert()
    if pipe is not None:
        try:
            result   = pipe(text[:512])[0]
            raw      = result.get("label", "NEUTRAL").upper()
            conf     = float(result.get("score", 0.5))
            if "NEG" in raw or raw in ("LABEL_0", "0"):
                label = "negative"
            elif "POS" in raw or raw in ("LABEL_2", "2"):
                label = "positive"
            else:
                label = "neutral"
            return {
                "label":         label,
                "confidence":    round(conf, 4),
                "numeric_score": round(SENTIMENT_SCORES[label] * conf, 4),
            }
        except Exception as e:
            print(f"[ViSoBERT] Error: {e}")

    return _lexicon_fallback(text)


def _lexicon_fallback(text: str) -> Dict[str, Any]:
    POS = ["tăng","tích cực","khởi sắc","vượt","đỉnh","lợi nhuận","tốt","mạnh",
           "phục hồi","bứt phá","kỳ vọng","lạc quan","cao","tăng trưởng","sinh lời",
           "kỷ lục","thuận lợi","hưởng lợi","cổ tức","hiệu quả","cải thiện","khả quan"]
    NEG = ["giảm","sụt","lao dốc","rủi ro","lo ngại","thua lỗ","xấu","khó khăn",
           "áp lực","cảnh báo","thấp","giảm sâu","tiêu cực","nợ xấu","bán tháo",
           "cắt lỗ","tranh cãi","vi phạm","phạt","kiện","tụt","suy giảm","bất ổn"]
    lower = text.lower()
    pos = sum(1 for w in POS if w in lower)
    neg = sum(1 for w in NEG if w in lower)
    if pos > neg:
        return {"label":"positive","confidence":0.5,"numeric_score":round(min(0.65,0.3+0.05*pos),4)}
    if neg > pos:
        return {"label":"negative","confidence":0.5,"numeric_score":round(max(-0.65,-0.3-0.05*neg),4)}
    return {"label":"neutral","confidence":0.5,"numeric_score":0.0}


def _aggregate_sentiment(scored: List[Dict]) -> Dict[str, Any]:
    if not scored:
        return {"label":"neutral","avg_score":0.0,"article_count":0,
                "positive":0,"negative":0,"neutral_count":0}
    scores = [s["numeric_score"] for s in scored]
    labels = [s["label"] for s in scored]
    avg    = round(sum(scores)/len(scores), 4)
    return {
        "label":         "positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral",
        "avg_score":     avg,
        "article_count": len(scored),
        "positive":      labels.count("positive"),
        "negative":      labels.count("negative"),
        "neutral_count": labels.count("neutral"),
    }


# ── Related company extraction ─────────────────────────────────────────────────

def _is_article_relevant(article: dict, ticker: str, company_keywords: List[str] = None) -> bool:
    """
    Kiểm tra bài báo có liên quan đến mã cổ phiếu không.
    
    Lý do cần: CafeF URL /hvn.html trả về bài về Honda Vietnam (HVN)
    lẫn với Vietnam Airlines (HVN) — cần lọc bài báo không liên quan.
    
    Logic:
    - Nếu ticker ngắn (≤3 ký tự): kiểm tra URL có chứa ticker không
    - Bài từ URL chứa ticker → luôn relevant
    - Bài có title/snippet chứa ticker hoặc keyword công ty → relevant
    - Bài không chứa gì → không relevant (bị loại)
    """
    if not article:
        return False
    
    ticker_lower = ticker.lower()
    url   = article.get("url", "").lower()
    title = article.get("title", "").lower()
    snip  = article.get("snippet", "").lower()
    text  = f"{title} {snip}"
    
    # URL chứa mã cổ phiếu → relevant
    # Ví dụ: cafef.vn/hvn-tang-manh.html → relevant cho HVN
    if f"/{ticker_lower}" in url or f"-{ticker_lower}-" in url or f"-{ticker_lower}." in url:
        return True
    
    # Text chứa ticker dưới dạng từ độc lập → relevant
    import re as _re
    if _re.search(r'\b' + ticker_lower + r'\b', text):
        return True
    
    # Kiểm tra keyword công ty nếu có
    if company_keywords:
        for kw in company_keywords:
            if kw.lower() in text:
                return True
    
    # Nếu ticker dài (≥4 ký tự) → ít nhầm hơn, chấp nhận nếu URL liên quan
    if len(ticker) >= 4 and ticker_lower in url:
        return True
    
    # Không khớp gì → không relevant
    return False


def _get_company_keywords(ticker: str) -> List[str]:
    """
    Trả về các từ khóa tên công ty phổ biến để filter bài báo.
    Chỉ cần match một trong các từ khóa là relevant.
    """
    # Mapping một số ticker phổ biến → từ khóa nhận dạng
    KNOWN_KEYWORDS = {
        "HVN": ["vietnam airlines", "hàng không việt nam", "hvn"],
        "VNM": ["vinamilk", "sữa việt nam", "vnm"],
        "VIC": ["vingroup", "tập đoàn vin"],
        "VHM": ["vinhomes"],
        "FPT": ["fpt corporation", "công ty fpt"],
        "HPG": ["hòa phát", "hoa phat"],
        "MBB": ["mb bank", "quân đội"],
        "ACB": ["á châu", "ngân hàng á châu"],
        "TCB": ["techcombank", "kỹ thương"],
        "VCB": ["vietcombank", "ngoại thương"],
        "BID": ["bidv", "đầu tư và phát triển"],
        "CTG": ["vietinbank", "công thương"],
        "VPB": ["vpbank", "thịnh vượng"],
        "STB": ["sacombank", "sài gòn thương tín"],
        "TPB": ["tpbank", "tiên phong"],
        "MWG": ["thế giới di động", "the gioi di dong"],
        "PNJ": ["phú nhuận", "phu nhuan"],
        "MSN": ["masan"],
        "SSI": ["ssi chứng khoán"],
        "VRE": ["vincom retail"],
        "GAS": ["pv gas", "khí việt nam"],
        "SAB": ["sabeco", "bia sài gòn"],
        "PLX": ["petrolimex", "xăng dầu việt nam"],
        "POW": ["pv power", "điện lực dầu khí"],
    }
    return KNOWN_KEYWORDS.get(ticker.upper(), [ticker.lower()])


def _extract_related_batch(llm, batch: List[dict], target: str, idx: int) -> str:
    n    = len(batch)
    body = ""
    for i, a in enumerate(batch, 1):
        snip = (a.get("snippet") or "")[:400]
        body += f"\n[Bài {i}] {a.get('title','')}\n{snip}\n"

    try:
        resp = llm.invoke([
            SystemMessage(content=(
                "Bạn là trợ lý AI chuyên phân tích tài chính chứng khoán Việt Nam. "
                "Nhiệm vụ: trích xuất MÃ CỔ PHIẾU (2-5 chữ cái in hoa, VD: VNM, HPG, ACB) "
                "của các công ty liên quan đến mã mục tiêu. "
                "CHỈ trả về mã cổ phiếu hợp lệ trên sàn HOSE/HNX/UPCOM, "
                "KHÔNG trả về tên công ty, vai trò, hay từ mô tả chung. "
                "Định dạng: mỗi dòng một mã, không có gì khác."
            )),
            HumanMessage(content=(
                f"Từ {n} bài báo về {target} (lô {idx}), "
                f"hãy liệt kê MÃ CỔ PHIẾU (chỉ mã, không tên) của tối đa 10 công ty "
                f"ảnh hưởng đến giá {target} (đối thủ, đối tác, cổ đông lớn, ngân hàng cho vay...).\n\n"
                f"Ví dụ output hợp lệ:\nVHM\nACB\nSTB\nNVL\n\n"
                f"=== {n} BÀI BÁO (LÔ {idx}) ===\n{body}\n\n"
                f"Chỉ liệt kê mã cổ phiếu, mỗi mã một dòng:"
            )),
        ])
        return resp.content or ""
    except Exception as e:
        print(f"[SentimentAgent] LLM batch {idx} error: {e}")
        return ""


def _parse_codes_from_table(text: str) -> List[str]:
    BLACKLIST = {
        "N/A","STT","MÃ","TÊN","VAI","LÝ","DO","VỚI","CÔNG","TY","CỔ","PHIẾU",
        "NGÂN","HÀNG","ĐỐI","TÁC","BÁO","CÁO","LÔ","BÀI","TOP","SỐ",
        "ROLE","NAME","CODE","TICKER","COMPANY","REASON","VÀ","CÁC","LÀ",
        "HNX","HOSE","UPCOM","VND","USD","VN","DN","CT","CP","TNHH",
    }

    codes = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[\|\s\-:]+$', line):
            continue
        found = re.findall(r'\b([A-Z]{2,5})\b', line)
        for code in found:
            if code not in BLACKLIST:
                codes.append(code)

    return codes


def _find_related_companies(llm, articles: List[dict], target: str) -> List[str]:
    try:
        from realtime_loader import fetch_dstock_all_symbols
        valid_symbols: set = {s["code"] for s in fetch_dstock_all_symbols()}
    except Exception:
        valid_symbols = set()

    all_codes: List[str] = []
    for idx, start in enumerate(range(0, len(articles), BATCH_SIZE), 1):
        batch  = articles[start: start + BATCH_SIZE]
        result = _extract_related_batch(llm, batch, target, idx)
        all_codes.extend(_parse_codes_from_table(result))
        time.sleep(0.5)

    counter = Counter(all_codes)
    clean   = []
    seen    = set()
    for code, _ in counter.most_common():
        if code == target:
            continue
        if len(code) < 2:
            continue
        if code in seen:
            continue
        if valid_symbols and code not in valid_symbols:
            continue
        seen.add(code)
        clean.append(code)
        if len(clean) >= TOP_RELATED:
            break

    print(f"[SentimentAgent] Related (validated): {clean}")
    return clean


# ── Sentiment bar ──────────────────────────────────────────────────────────────

def _sentiment_bar(score: float) -> str:
    clamped = max(-1.0, min(1.0, score))
    bars    = int(abs(clamped) * 10)
    if clamped > 0.1:
        return f"TÍCH CỰC ({score:+.3f}) {'█'*bars}"
    if clamped < -0.1:
        return f"TIÊU CỰC ({score:+.3f}) {'█'*bars}"
    return f"TRUNG TÍNH ({score:+.3f}) {'▒'*max(1,bars)}"


# ── Report builder ─────────────────────────────────────────────────────────────

def _build_report(
    llm, target: str, time_frame: str,
    scored_articles: List[dict],
    main_agg: Dict,
    related_companies: List[str],
    related_sentiment: Dict[str, Dict],
) -> str:
    art_sum = ""
    for i, a in enumerate(scored_articles[:15], 1):
        icon = "🟢" if a["label"]=="positive" else ("🔴" if a["label"]=="negative" else "⚪")
        art_sum += f"{i}. {icon} [{a['label'].upper()}] {a['title'][:80]}\n   Điểm: {a['numeric_score']:+.3f} | {a.get('date','')}\n"

    rel_sum = ""
    for co in related_companies:
        s    = related_sentiment.get(co, {})
        icon = "🟢" if s.get("label")=="positive" else ("🔴" if s.get("label")=="negative" else "⚪")
        rel_sum += f"- **{co}**: {icon} {s.get('label','N/A').upper()} | điểm: {s.get('avg_score',0):+.3f} | {s.get('article_count',0)} bài\n"

    visobert_note = ("✅ 5CD-AI/Vietnamese-Sentiment-visobert"
                     if _visobert_pipeline else "⚠️ Lexicon fallback")

    try:
        resp = llm.invoke([
            SystemMessage(content=(
                "Bạn là chuyên gia phân tích sentiment thị trường chứng khoán Việt Nam. "
                "Tổng hợp dữ liệu từ báo chí, trả lời tiếng Việt, markdown."
            )),
            HumanMessage(content=f"""Phân tích sentiment cho **{target}** ({time_frame}).

=== KẾT QUẢ ===
Số bài: {main_agg.get('article_count',0)} | Điểm TB: {main_agg.get('avg_score',0):+.4f} | Nhận định: {main_agg.get('label','neutral').upper()}
Phân bổ: 🟢{main_agg.get('positive',0)} / ⚪{main_agg.get('neutral_count',0)} / 🔴{main_agg.get('negative',0)}
{_sentiment_bar(main_agg.get('avg_score',0))} | Model: {visobert_note}

=== 15 BÀI GẦN NHẤT ===
{art_sum or '(Không có bài)'}

=== CÔNG TY LIÊN QUAN ===
{rel_sum or '(Không tìm được)'}

---
Viết báo cáo theo template:

**📊 Tóm tắt Sentiment — {target}**
**Nhận định chính:** [1-2 câu]
**Phân tích chi tiết:**
- Xu hướng tin tức: [...]
- Tín hiệu tích cực: [...]
- Tín hiệu tiêu cực: [...]
- Áp lực từ công ty liên quan: [...]
"""),
        ])
        llm_text = resp.content or ""
    except Exception as e:
        llm_text = f"Lỗi tổng hợp: {e}"

    return (
        f"## 📰 Phân tích Sentiment — {target}\n\n"
        f"### Kết quả tổng hợp\n"
        f"| Chỉ số | Giá trị |\n|--------|----------|\n"
        f"| Số bài phân tích | {main_agg.get('article_count',0)} |\n"
        f"| Điểm sentiment TB | **{main_agg.get('avg_score',0):+.4f}** |\n"
        f"| Nhận định | **{main_agg.get('label','neutral').upper()}** |\n"
        f"| Tích cực / Trung tính / Tiêu cực | "
        f"{main_agg.get('positive',0)} / {main_agg.get('neutral_count',0)} / {main_agg.get('negative',0)} |\n"
        f"| Thanh cảm xúc | {_sentiment_bar(main_agg.get('avg_score',0))} |\n"
        f"### Công ty liên quan (Top {len(related_companies)})\n"
        f"{rel_sum or '_Không tìm được_'}\n\n"
        f"---\n\n{llm_text}\n"
    )


# ── Standalone function for Alpha Agent ───────────────────────────────────────

def run_sentiment_for_alpha(
    llm, stock_name: str, time_frame: str
) -> Tuple[Dict[str, Any], str]:
    """
    Chạy phân tích sentiment và trả về (sentiment_data, sentiment_report).
    Được gọi trực tiếp bởi Alpha Agent thay vì chạy như một node pipeline.
    """
    print(f"\n{'='*55}")
    print(f"[SentimentAgent] Bắt đầu thu thập sentiment: {stock_name}")
    print(f"{'='*55}")

    empty_data = {
        "target_stock":       stock_name,
        "main_sentiment":     _aggregate_sentiment([]),
        "scored_articles":    [],
        "related_companies":  [],
        "related_sentiment":  {},
        "visobert_available": _visobert_pipeline is not None,
        "model_used": "N/A",
    }

    # Bước 1: Thu thập bài báo
    articles = _collect_articles(stock_name, max_articles=MAX_ARTICLES)

    if not articles:
        print(f"[SentimentAgent] ⚠️ Không lấy được bài nào cho {stock_name}")
        report = (
            f"## 📰 Sentiment — {stock_name}\n\n"
            "⚠️ Không thu thập được bài báo từ CafeF.\n\n"
            f"CafeF có thể chưa có trang riêng cho mã **{stock_name}** "
            "hoặc lỗi kết nối mạng tạm thời.\n\n"
            "_Sentiment score = 0 (neutral) sẽ được dùng làm đầu vào cho Alpha Agent._"
        )
        return empty_data, report

    # Bước 2: Lọc bài báo không liên quan (giải quyết vấn đề HVN ← Honda)
    company_kws = _get_company_keywords(stock_name)
    relevant_articles = [
        art for art in articles
        if _is_article_relevant(art, stock_name, company_kws)
    ]
    n_filtered = len(articles) - len(relevant_articles)
    if n_filtered > 0:
        print(f"[SentimentAgent] Lọc {n_filtered}/{len(articles)} bài không liên quan đến {stock_name}")
    articles = relevant_articles if relevant_articles else articles

    # Bước 3: Tính sentiment từng bài liên quan
    print(f"[SentimentAgent] Tính sentiment {len(articles)} bài liên quan...")
    scored: List[dict] = []
    for art in articles:
        text = (art.get("title","") + " " + art.get("snippet","")).strip()
        if text:
            s = _score_text(text)
            scored.append({
                "title":         art.get("title",""),
                "url":           art.get("url",""),
                "date":          art.get("date",""),
                "label":         s["label"],
                "numeric_score": s["numeric_score"],
                "confidence":    s["confidence"],
                "content":       text[:200],
            })

    main_agg = _aggregate_sentiment(scored)
    print(f"[SentimentAgent] Tổng hợp: {main_agg['label'].upper()} "
          f"(điểm={main_agg['avg_score']:+.3f}, {main_agg['article_count']} bài liên quan)")

    # Bước 3: Tìm công ty liên quan
    related: List[str] = []
    try:
        related = _find_related_companies(llm, articles, stock_name)
        related = [c for c in related if c != stock_name][:TOP_RELATED]
    except Exception as e:
        print(f"[SentimentAgent] Lỗi tìm related: {e}")

    # Bước 4: Sentiment công ty liên quan
    related_sentiment: Dict[str, Dict] = {}
    for co in related[:7]:
        try:
            print(f"[SentimentAgent] Related: {co}...")
            co_arts    = _collect_articles(co, max_articles=MAX_RELATED_ARTICLES)
            co_scored  = []
            for a in co_arts:
                txt = (a.get("title","") + " " + a.get("snippet","")).strip()
                if txt:
                    co_scored.append(_score_text(txt))
            related_sentiment[co] = _aggregate_sentiment(co_scored)
        except Exception as e:
            print(f"[SentimentAgent] Error {co}: {e}")
            related_sentiment[co] = {
                "label":"neutral","avg_score":0.0,"article_count":0,
                "positive":0,"negative":0,"neutral_count":0,
            }

    # Bước 5: Tổng hợp báo cáo
    report = _build_report(
        llm, stock_name, time_frame,
        scored, main_agg, related, related_sentiment,
    )

    sentiment_data = {
        "target_stock":       stock_name,
        "main_sentiment":     main_agg,
        "scored_articles":    scored[:15],
        "related_companies":  related,
        "related_sentiment":  related_sentiment,
        "visobert_available": _visobert_pipeline is not None,
        "model_used": (
            "5CD-AI/Vietnamese-Sentiment-visobert"
            if _visobert_pipeline else "lexicon-fallback"
        ),
    }

    print(f"[SentimentAgent] ✓ Hoàn thành ({len(scored)} bài scored, {len(related)} related)")
    return sentiment_data, report


# ── Legacy pipeline node (kept for compatibility) ──────────────────────────────

def create_sentiment_agent(llm):
    """
    Legacy: tạo nút Sentiment Agent cho pipeline LangGraph.
    Hiện không dùng trong pipeline chính — sentiment chạy trong Alpha Agent.
    """

    def sentiment_agent_node(state: dict) -> dict:
        stock_name = state.get("stock_name", "")
        time_frame = state.get("time_frame", "1 day")

        sentiment_data, report = run_sentiment_for_alpha(llm, stock_name, time_frame)

        return {
            "messages":         state.get("messages", []),
            "sentiment_report": report,
            "sentiment_data":   sentiment_data,
        }

    return sentiment_agent_node