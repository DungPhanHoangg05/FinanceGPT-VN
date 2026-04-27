import re
import time
from typing import List, Optional, Tuple

import requests

# ── Shared headers ────────────────────────────────────────────────────────────

_HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Connection": "keep-alive",
}

_HEADERS_TCBS = {
    **_HEADERS_BASE,
    "Origin":  "https://tcinvest.tcbs.com.vn",
    "Referer": "https://tcinvest.tcbs.com.vn/",
}

_HEADERS_VND = {
    **_HEADERS_BASE,
    "Origin":  "https://dstock.vndirect.com.vn",
    "Referer": "https://dstock.vndirect.com.vn/",
}

_HEADERS_SSI = {
    **_HEADERS_BASE,
    "Origin":  "https://iboard.ssi.com.vn",
    "Referer": "https://iboard.ssi.com.vn/",
}

_HEADERS_HTML = {
    **_HEADERS_BASE,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}


def _get(url: str, params: dict = None, headers: dict = None,
         timeout: int = 12, retries: int = 2) -> Optional[requests.Response]:
    """Shared GET with retry."""
    h = headers or _HEADERS_BASE
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404, 401):
                return None        # no retry on auth/not-found
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
    return None


# ═════════════════════════════════════════════════════════════════════════════
# TCBS FINANCIAL DATA API
# Endpoint: apipubaws.tcbs.com.vn/stock-insight/v1
# ═════════════════════════════════════════════════════════════════════════════

TCBS_BASE = "https://apipubaws.tcbs.com.vn/stock-insight"

# Tên tiếng Việt cho các trường TCBS Income Statement
TCBS_INCOME_LABELS = {
    "revenue":                 "Doanh thu thuần",
    "yearRevenueGrowth":       "Tăng trưởng doanh thu YoY (%)",
    "quarterRevenueGrowth":    "Tăng trưởng doanh thu QoQ (%)",
    "costsOfGoodsSold":        "Giá vốn hàng bán",
    "grossProfit":             "Lợi nhuận gộp",
    "operationExpense":        "Chi phí hoạt động",
    "operationProfit":         "Lợi nhuận từ hoạt động KD",
    "ebitda":                  "EBITDA",
    "interestExpense":         "Chi phí lãi vay",
    "preTaxProfit":            "Lợi nhuận trước thuế",
    "postTaxProfit":           "Lợi nhuận sau thuế",
    "shareHolderIncome":       "LNST thuộc cổ đông công ty mẹ",
    "yearShareHolderIncomeGrowth": "Tăng trưởng LNST YoY (%)",
    "quarterShareHolderIncomeGrowth": "Tăng trưởng LNST QoQ (%)",
}

TCBS_BALANCE_LABELS = {
    "shortAsset":        "Tài sản ngắn hạn",
    "cash":              "Tiền & tương đương tiền",
    "shortInvest":       "Đầu tư ngắn hạn",
    "shortReceivable":   "Phải thu ngắn hạn",
    "inventory":         "Hàng tồn kho",
    "longAsset":         "Tài sản dài hạn",
    "fixedAsset":        "Tài sản cố định",
    "asset":             "Tổng tài sản",
    "debt":              "Tổng nợ phải trả",
    "shortDebt":         "Nợ ngắn hạn",
    "longDebt":          "Nợ dài hạn",
    "equity":            "Vốn chủ sở hữu",
    "capital":           "Vốn điều lệ",
    "centralBankDeposit":"Tiền gửi NHNN",
    "otherBankDeposit":  "Tiền gửi TCTD khác",
}

TCBS_CASHFLOW_LABELS = {
    "investCost":   "Chi đầu tư tài sản cố định",
    "fromInvest":   "Lưu chuyển tiền từ đầu tư",
    "fromFinancial":"Lưu chuyển tiền từ tài chính",
    "fromSale":     "Lưu chuyển tiền từ hoạt động KD",
    "freeCashFlow": "Dòng tiền tự do (FCF)",
}

TCBS_RATIO_LABELS = {
    "priceToEarning":      "P/E",
    "priceToBook":         "P/B",
    "valueBeforeEbitda":   "EV/EBITDA",
    "roe":                 "ROE (%)",
    "roa":                 "ROA (%)",
    "earningPerShare":     "EPS (đồng)",
    "bookValuePerShare":   "BVPS (đồng)",
    "dividendYield":       "Tỷ suất cổ tức (%)",
    "grossProfitMargin":   "Biên lợi nhuận gộp (%)",
    "operatingProfitMargin":"Biên lợi nhuận hoạt động (%)",
    "postTaxMargin":       "Biên lợi nhuận ròng (%)",
    "debtOnEquity":        "Nợ/Vốn chủ sở hữu (D/E)",
    "debtOnAsset":         "Nợ/Tổng tài sản",
    "debtOnEbitda":        "Nợ/EBITDA",
    "currentPayment":      "Hệ số thanh toán hiện hành",
    "quickPayment":        "Hệ số thanh toán nhanh",
    "epsChange":           "Tăng trưởng EPS YoY (%)",
    "revenueOnAsset":      "Vòng quay tài sản",
    "assetOnEquity":       "Đòn bẩy tài chính",
    "ebitOnInterest":      "EBIT/Lãi vay (ICR)",
    "daysReceivable":      "Kỳ thu tiền bình quân (ngày)",
    "daysInventory":       "Số ngày tồn kho (ngày)",
    "daysPayable":         "Kỳ thanh toán bình quân (ngày)",
}


def _tcbs_financial(endpoint: str, ticker: str, yearly: bool = False,
                    periods: int = 8) -> List[dict]:
    """Generic TCBS financial report fetcher (tries v1 then v2 then vnstock lib)."""
    ticker = ticker.upper()
    
    # ── Try direct API first ──────────────────────────────────────────────────
    versions = ["v1", "v2"]
    for ver in versions:
        url = f"{TCBS_BASE}/{ver}/financial-report/{endpoint}"
        r = _get(url, params={
            "ticker": ticker,
            "yearly": 1 if yearly else 0,
            "page":   0,
            "size":   periods,
        }, headers=_HEADERS_TCBS)
        if r and r.status_code == 200:
            try:
                data = r.json()
                res = data.get("data", [])
                if res: return res
            except:
                continue

    # ── Fallback to vnstock library ───────────────────────────────────────────
    try:
        from vnstock import Vnstock
        vn = Vnstock()
        # vnstock supports different source providers; TCBS may not be supported by vnstock
        # Try a list of commonly supported sources and return the first successful one
        period = 'year' if yearly else 'quarter'
        df = None
        supported_sources = ["KBS", "VCI", "MSN", "FMP"]
        for src in supported_sources:
            try:
                stock = vn.stock(symbol=ticker, source=src)
                if endpoint == "income-statement":
                    df = stock.finance.income_statement(period=period)
                elif endpoint == "balance-sheet":
                    df = stock.finance.balance_sheet(period=period)
                elif endpoint == "cash-flow":
                    df = stock.finance.cash_flow(period=period)
                elif endpoint == "financial-ratio":
                    df = stock.finance.ratio(period=period)

                if df is not None and not df.empty:
                    return df.to_dict('records')
            except Exception:
                # try next source
                continue
    except Exception as e:
        print(f"[DataSources] vnstock fallback failed for {ticker}/{endpoint}: {e}")

    return []


def fetch_tcbs_income_statement(ticker: str, yearly: bool = False, periods: int = 8) -> List[dict]:
    """
    Lấy kết quả kinh doanh từ TCBS.
    Fields: revenue, costsOfGoodsSold, grossProfit, operationProfit,
            ebitda, interestExpense, preTaxProfit, postTaxProfit,
            shareHolderIncome, yearRevenueGrowth, yearShareHolderIncomeGrowth
    Đơn vị: tỷ đồng (chia 1e9)
    """
    return _tcbs_financial("income-statement", ticker, yearly, periods)


def fetch_tcbs_balance_sheet(ticker: str, yearly: bool = False, periods: int = 8) -> List[dict]:
    """
    Lấy bảng cân đối kế toán từ TCBS.
    Fields: asset, shortAsset, longAsset, fixedAsset, cash, inventory,
            debt, shortDebt, longDebt, equity, capital
    """
    return _tcbs_financial("balance-sheet", ticker, yearly, periods)


def fetch_tcbs_cashflow(ticker: str, yearly: bool = False, periods: int = 8) -> List[dict]:
    """
    Lấy báo cáo lưu chuyển tiền tệ từ TCBS.
    Fields: fromSale, fromInvest, fromFinancial, investCost, freeCashFlow
    """
    return _tcbs_financial("cash-flow", ticker, yearly, periods)


def fetch_tcbs_financial_ratios(ticker: str, yearly: bool = False, periods: int = 8) -> List[dict]:
    """
    Lấy các chỉ số tài chính từ TCBS.
    Fields: priceToEarning(P/E), priceToBook(P/B), roe, roa,
            earningPerShare, grossProfitMargin, postTaxMargin,
            debtOnEquity, currentPayment, dividendYield, ...
    """
    return _tcbs_financial("financial-ratio", ticker, yearly, periods)


def fetch_tcbs_company_overview(ticker: str) -> dict:
    """Lấy thông tin tổng quan công ty từ TCBS — thử nhiều endpoint."""
    ticker = ticker.upper()

    # ── Try TCBS stock-info (endpoint này trả về organName, charterCapital, ...) ──
    r_si = _get(
        f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/{ticker}/stock-info",
        headers=_HEADERS_TCBS,
    )
    if r_si:
        try:
            d = r_si.json() or {}
            if any(d.get(k) for k in ["organName", "companyName", "charterCapital", "industryVi"]):
                return d
        except Exception:
            pass

    # ── Try TCBS company profile variants ────────────────────────────────────
    endpoints = [
        (f"{TCBS_BASE}/v2/company/profile", {"ticker": ticker}),
        (f"{TCBS_BASE}/v1/company/profile",  {"ticker": ticker}),
        (f"{TCBS_BASE}/v2/company/{ticker}",  {}),
        (f"{TCBS_BASE}/v1/company/{ticker}",  {}),
    ]
    for url, params in endpoints:
        r = _get(url, params=params or None, headers=_HEADERS_TCBS)
        if r is None:
            continue
        try:
            data = r.json()
            if not isinstance(data, dict):
                continue
            if any(data.get(k) for k in [
                "companyName", "organName", "organ_name", "shortName",
                "ticker", "companyProfile", "charterCapital",
            ]):
                return data
        except Exception:
            continue

    # ── Fallback: vnstock company.overview() ─────────────────────────────────
    try:
        from vnstock import Vnstock
        vn = Vnstock()
        for src in ["VCI", "KBS"]:
            try:
                stock = vn.stock(symbol=ticker, source=src)
                df_ov = stock.company.overview()
                if df_ov is not None and len(df_ov) > 0:
                    row = df_ov.iloc[0].to_dict() if hasattr(df_ov, "iloc") else {}
                    if row:
                        return row
            except Exception:
                continue
    except Exception:
        pass

    return {}


def fetch_tcbs_market_eval(ticker: str) -> dict:
    """
    Lấy định giá thị trường hiện tại từ TCBS:
    market cap, P/E, P/B, EV/EBITDA, beta, giá mục tiêu nếu có.
    """
    r = _get(f"{TCBS_BASE}/v1/company/{ticker.upper()}",
             headers=_HEADERS_TCBS)
    if r is None:
        return {}
    try:
        d = r.json()
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def format_tcbs_income(records: List[dict]) -> List[dict]:
    """Chuyển đổi TCBS income statement sang format đọc được (tỷ đồng)."""
    out = []
    for r in records:
        period = f"Q{r.get('quarter','?')}/{r.get('year','?')}"
        formatted = {"kỳ": period}
        for key, label in TCBS_INCOME_LABELS.items():
            val = r.get(key)
            if val is None:
                continue
            if key in ("yearRevenueGrowth", "quarterRevenueGrowth",
                       "yearShareHolderIncomeGrowth", "quarterShareHolderIncomeGrowth"):
                formatted[label] = f"{round(val * 100, 2)}%"
            else:
                # Convert to billion VND
                formatted[label] = round(val / 1e9, 2) if abs(val) > 1e6 else round(val, 4)
        out.append(formatted)
    return out


def format_tcbs_ratios(records: List[dict]) -> List[dict]:
    """Chuyển đổi TCBS financial ratios sang format đọc được."""
    out = []
    for r in records:
        period = f"Q{r.get('quarter','?')}/{r.get('year','?')}"
        formatted = {"kỳ": period}
        for key, label in TCBS_RATIO_LABELS.items():
            val = r.get(key)
            if val is None:
                continue
            formatted[label] = round(float(val), 4)
        out.append(formatted)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# VNDIRECT FINANCIAL DATA API
# Endpoint: api-finfo.vndirect.com.vn/v4
# ═════════════════════════════════════════════════════════════════════════════

VND_BASE = "https://api-finfo.vndirect.com.vn/v4"


def _vnd_get(path: str, params: dict = None) -> Optional[dict]:
    r = _get(f"{VND_BASE}/{path}", params=params, headers=_HEADERS_VND)
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def fetch_vndirect_ratios(ticker: str, periods: int = 8) -> List[dict]:
    """
    Lấy chỉ số tài chính từ VNDirect finfo:
    pe, pb, eps, roe, roa, debtToEquity, currentRatio,
    revenueGrowth, profitGrowth, ...
    """
    data = _vnd_get("financial-indicators", {
        "q":      f"code:{ticker.upper()}",
        "sort":   "reportDate:desc",
        "size":   periods,
        "fields": (
            "code,reportDate,pe,pb,eps,roe,roa,debtToEquity,"
            "currentRatio,revenueGrowth,profitGrowth,"
            "grossMargin,netMargin,operatingMargin,"
            "evToEbitda,dividendYield,bookValuePerShare"
        ),
    })
    if data is None:
        return []
    return data.get("data", [])


def fetch_vndirect_financial_summary(ticker: str, periods: int = 8) -> List[dict]:
    """Lấy tóm tắt báo cáo tài chính từ VNDirect."""
    data = _vnd_get("financial-statements", {
        "q":    f"code:{ticker.upper()}",
        "sort": "reportDate:desc",
        "size": periods,
    })
    if data is None:
        return []
    return data.get("data", [])


def fetch_vndirect_analyst_recs(ticker: str) -> List[dict]:
    """
    Lấy khuyến nghị phân tích từ VNDirect finfo:
    recommendation (MUA/BÁN/GIỮ), targetPrice, analystName, publishedDate
    """
    data = _vnd_get("recommendations", {
        "q":    f"code:{ticker.upper()}",
        "sort": "publishedDate:desc",
        "size": 10,
    })
    if data is None:
        return []
    return data.get("data", [])


def fetch_vndirect_price_target(ticker: str) -> dict:
    """Lấy giá mục tiêu đồng thuận từ VNDirect."""
    data = _vnd_get("price-target", {
        "q":    f"code:{ticker.upper()}",
        "size": 1,
    })
    if data is None:
        return {}
    items = data.get("data", [])
    return items[0] if items else {}


def fetch_vndirect_ownership(ticker: str) -> List[dict]:
    """Lấy thông tin cổ đông lớn từ VNDirect finfo."""
    data = _vnd_get("ownership", {
        "q":    f"code:{ticker.upper()}",
        "sort": "ownedRate:desc",
        "size": 10,
    })
    if data is None:
        return []
    return data.get("data", [])


def fetch_vndirect_dividends(ticker: str, periods: int = 10) -> List[dict]:
    """Lấy lịch sử chi trả cổ tức từ VNDirect finfo."""
    data = _vnd_get("dividends", {
        "q":    f"code:{ticker.upper()}",
        "sort": "exerciseDate:desc",
        "size": periods,
    })
    if data is None:
        return []
    return data.get("data", [])


# ═════════════════════════════════════════════════════════════════════════════
# SSI iBOARD FINANCIAL DATA
# Endpoint: iboard-query.ssi.com.vn/v2
# ═════════════════════════════════════════════════════════════════════════════

SSI_BASE = "https://iboard-query.ssi.com.vn/v2"


def _ssi_get(path: str, params: dict = None) -> Optional[dict]:
    r = _get(f"{SSI_BASE}/{path}", params=params, headers=_HEADERS_SSI)
    if r is None:
        return None
    try:
        return r.json()
    except Exception:
        return None


def fetch_ssi_financial_data(ticker: str) -> dict:
    """Lấy dữ liệu tài chính từ SSI iBoard."""
    return _ssi_get("stock/financial", {
        "symbol":     ticker.upper(),
        "reportType": "QUARTERLY",
        "reportSize": 8,
    }) or {}


def fetch_ssi_company_info(ticker: str) -> dict:
    """Lấy thông tin công ty từ SSI iBoard."""
    return _ssi_get("stock/company", {"symbol": ticker.upper()}) or {}


def fetch_ssi_research_list(ticker: str, max_items: int = 10) -> List[dict]:
    """
    Lấy danh sách báo cáo nghiên cứu từ SSI Research portal.
    Trả về: title, date, analyst, recommendation, targetPrice, reportUrl
    """
    results: List[dict] = []

    # Method 1: iBoard research API
    data = _ssi_get("research/list", {
        "stockCode": ticker.upper(),
        "size":      max_items,
        "page":      0,
    })
    if data:
        for item in data.get("data", data.get("items", [])):
            results.append({
                "title":          item.get("title", item.get("reportTitle", "")),
                "date":           item.get("publishDate", item.get("date", "")),
                "analyst":        item.get("analystName", item.get("author", "SSI Research")),
                "recommendation": item.get("recommendation", ""),
                "target_price":   item.get("targetPrice", ""),
                "url":            item.get("reportUrl", item.get("url", "")),
                "source":         "SSI Research",
            })

    # Method 2: SSI Research portal scrape
    if not results:
        results = _scrape_ssi_research_portal(ticker, max_items)

    return results[:max_items]


def _scrape_ssi_research_portal(ticker: str, max_items: int = 10) -> List[dict]:
    """Scrape SSI Research portal cho báo cáo phân tích."""
    results = []
    try:
        from bs4 import BeautifulSoup
        url = f"https://research.ssi.com.vn/search?keyword={ticker.upper()}"
        r = _get(url, headers={**_HEADERS_HTML, "Referer": "https://research.ssi.com.vn/"})
        if r is None:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        for card in soup.select(".report-item, .research-card, article.item")[:max_items]:
            title_el = card.select_one("h2, h3, .title, .report-title")
            link_el  = card.select_one("a[href]")
            date_el  = card.select_one(".date, time, .published")
            rec_el   = card.select_one(".recommendation, .rating, .action")

            title = title_el.get_text(strip=True) if title_el else ""
            href  = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = "https://research.ssi.com.vn" + href

            if title and len(title) > 8:
                results.append({
                    "title":          title,
                    "date":           date_el.get_text(strip=True) if date_el else "",
                    "recommendation": rec_el.get_text(strip=True) if rec_el else "",
                    "analyst":        "SSI Research",
                    "url":            href,
                    "source":         "SSI Research",
                })
    except Exception as e:
        print(f"[SSI Scrape] Error: {e}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# VNDIRECT RESEARCH PORTAL
# ═════════════════════════════════════════════════════════════════════════════

def fetch_vndirect_research_reports(ticker: str, max_items: int = 10) -> List[dict]:
    """
    Lấy báo cáo phân tích từ VNDIRECT Research.
    Trả về: title, date, analyst, recommendation, targetPrice, url
    """
    results: List[dict] = []

    # Method 1: finfo research API
    data = _vnd_get("research-reports", {
        "q":    f"code:{ticker.upper()}",
        "sort": "publishedDate:desc",
        "size": max_items,
    })
    if data:
        for rp in data.get("data", []):
            results.append({
                "title":          rp.get("title", ""),
                "date":           rp.get("publishedDate", rp.get("reportDate", "")),
                "analyst":        rp.get("analystName", "VNDIRECT Research"),
                "recommendation": rp.get("recommendation", rp.get("rating", "")),
                "target_price":   rp.get("targetPrice", ""),
                "url":            rp.get("reportUrl", rp.get("fileUrl", "")),
                "source":         "VNDIRECT Research",
            })

    # Method 2: dstock API
    if not results:
        try:
            r = _get("https://dstock.vndirect.com.vn/api/research", params={
                "code":  ticker.upper(),
                "size":  max_items,
                "page":  0,
                "sort":  "publishedDate,desc",
            }, headers=_HEADERS_VND)
            if r:
                for rp in r.json().get("data", []):
                    results.append({
                        "title":          rp.get("title", ""),
                        "date":           rp.get("publishedDate", ""),
                        "analyst":        rp.get("author", "VNDIRECT Research"),
                        "recommendation": rp.get("recommendation", ""),
                        "target_price":   rp.get("targetPrice", ""),
                        "url":            rp.get("pdfUrl", rp.get("url", "")),
                        "source":         "VNDIRECT Research",
                    })
        except Exception:
            pass

    # Method 3: scrape VNDIRECT research portal
    if not results:
        results = _scrape_vndirect_research(ticker, max_items)

    return results[:max_items]


def _scrape_vndirect_research(ticker: str, max_items: int = 10) -> List[dict]:
    """Scrape báo cáo từ portal VNDIRECT."""
    results = []
    try:
        from bs4 import BeautifulSoup
        url = f"https://dstock.vndirect.com.vn/bao-cao-phan-tich?q={ticker.upper()}"
        r = _get(url, headers={**_HEADERS_HTML, "Referer": "https://dstock.vndirect.com.vn/"})
        if r is None:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        for card in soup.select(".report-item, .analysis-item, .research-item")[:max_items]:
            title_el = card.select_one("h2, h3, .title, a")
            link_el  = card.select_one("a[href]")
            date_el  = card.select_one(".date, .time, time")

            title = title_el.get_text(strip=True) if title_el else ""
            href  = link_el.get("href", "") if link_el else ""
            if href and not href.startswith("http"):
                href = "https://dstock.vndirect.com.vn" + href

            if title and len(title) > 8:
                results.append({
                    "title":  title,
                    "date":   date_el.get_text(strip=True) if date_el else "",
                    "url":    href,
                    "source": "VNDIRECT Research",
                })
    except Exception as e:
        print(f"[VNDIRECT Scrape] Error: {e}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# ENTRADE / DNSE RESEARCH
# Endpoint: services.entrade.com.vn
# ═════════════════════════════════════════════════════════════════════════════

def fetch_entrade_research(ticker: str, max_items: int = 8) -> List[dict]:
    """Lấy bài phân tích từ DNSE/Entrade blog."""
    results = []
    try:
        r = _get("https://services.entrade.com.vn/dnse-blog/api/article/search", params={
            "q":        ticker.upper(),
            "size":     max_items,
            "page":     0,
            "category": "analysis",
        }, headers={**_HEADERS_BASE, "Referer": "https://dnse.com.vn/"})
        if r:
            for item in r.json().get("data", r.json().get("items", [])):
                results.append({
                    "title":   item.get("title", ""),
                    "date":    item.get("publishedDate", item.get("createdAt", "")),
                    "url":     item.get("url", item.get("link", "")),
                    "snippet": (item.get("summary") or item.get("description") or "")[:250],
                    "source":  "DNSE/Entrade",
                })
    except Exception as e:
        print(f"[Entrade] Error: {e}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# CAFEF PHÂN TÍCH — SCRAPING BÀI PHÂN TÍCH CHỨNG KHOÁN
# ═════════════════════════════════════════════════════════════════════════════

BROKER_KEYWORDS = {
    "ssi":         "SSI Research",
    "vci":         "Vietcap (VCI)",
    "tcbs":        "TCBS Research",
    "vndirect":    "VNDIRECT Research",
    "hsc":         "HSC Research",
    "bsc":         "BSC Research",
    "mb securities":"MBS Research",
    "mbs":         "MBS Research",
    "acbs":        "ACBS Research",
    "kbs":         "KBS Research",
    "kis":         "KIS Việt Nam",
    "shs":         "SHS Research",
    "maybank":     "Maybank Securities",
    "mirae":       "Mirae Asset",
    "yuanta":      "Yuanta Securities",
    "pinetree":    "Pinetree Securities",
    "fpt securities":"FPT Securities",
    "rồng việt":   "Rồng Việt Securities",
    "agriseco":    "Agriseco Research",
    "dsc":         "DSC Research",
    "vcbs":        "Vietcombank Securities",
    "phân tích":   "",
    "báo cáo":     "",
    "khuyến nghị": "",
    "định giá":    "",
    "mua vào":     "",
    "bán ra":      "",
}

ANALYSIS_PATTERNS = [
    r"khuyến nghị\s*(mua|bán|giữ|trung lập|tích lũy)",
    r"giá mục tiêu[:\s]+[\d,\.]+",
    r"tăng\s*[\d,\.]+%",
    r"p/e\s*[\d,\.]+",
    r"p/b\s*[\d,\.]+",
    r"roe\s*[\d,\.]+",
    r"eps\s*[\d,\.]+",
    r"doanh thu.*tỷ",
    r"lợi nhuận.*tỷ",
    r"upside\s*[\d,\.]+",
    r"tiềm năng tăng giá",
]


def _extract_broker_from_text(text: str) -> Tuple[str, str]:
    """Tìm tên công ty chứng khoán và loại báo cáo từ text."""
    text_lower = text.lower()
    broker = ""
    for kw, name in BROKER_KEYWORDS.items():
        if kw in text_lower and name:
            broker = name
            break

    # Detect report type
    report_type = "Bài phân tích"
    if any(re.search(p, text_lower) for p in ANALYSIS_PATTERNS):
        report_type = "Báo cáo phân tích chứng khoán"
    elif "khuyến nghị" in text_lower:
        report_type = "Khuyến nghị đầu tư"

    return broker, report_type


def _extract_recommendation(text: str) -> Tuple[str, str]:
    """Tìm khuyến nghị mua/bán và giá mục tiêu từ text."""
    text_lower = text.lower()
    recommendation = ""
    target_price = ""

    # Recommendation
    rec_match = re.search(
        r"khuyến nghị\s*(mua|bán|giữ|trung lập|tích lũy|nắm giữ|outperform|buy|sell|hold)",
        text_lower
    )
    if rec_match:
        action = rec_match.group(1).upper()
        action_map = {
            "MUA": "MUA ✅", "BUY": "MUA ✅",
            "BÁN": "BÁN ❌", "SELL": "BÁN ❌",
            "GIỮ": "GIỮ ⚖️", "HOLD": "GIỮ ⚖️",
            "NẮM GIỮ": "GIỮ ⚖️",
            "TRUNG LẬP": "TRUNG LẬP ⚖️",
            "TÍCH LŨY": "TÍCH LŨY 🟡",
            "OUTPERFORM": "OUTPERFORM 📈",
        }
        recommendation = action_map.get(action, action)

    # Target price
    tp_match = re.search(
        r"giá mục tiêu[:\s]+([\d,\.]+)\s*(nghìn|ngàn|đồng|vnd)?",
        text_lower
    )
    if tp_match:
        target_price = tp_match.group(1).replace(",", "") + "000đ"

    return recommendation, target_price


def scrape_cafef_analysis_articles(ticker: str, max_items: int = 15) -> List[dict]:
    """
    Scrape bài phân tích chứng khoán từ CafeF.
    Tìm kiếm trong chuyên mục phân tích + search kết quả.
    """
    from bs4 import BeautifulSoup

    results: List[dict] = []
    seen_urls: set = set()
    ticker_lower = ticker.lower()

    # URL patterns để tìm bài phân tích trên CafeF
    search_urls = [
        f"https://cafef.vn/phan-tich-bao-cao/{ticker_lower}.html",
        f"https://cafef.vn/co-phieu-{ticker_lower}.html",
        f"https://cafef.vn/{ticker_lower}.html",
        # CafeF search
        f"https://cafef.vn/tim-kiem.html?keywords={ticker.upper()}+phân+tích",
    ]

    h = {**_HEADERS_HTML, "Referer": "https://cafef.vn/"}

    for url in search_urls:
        if len(results) >= max_items:
            break
        try:
            r = _get(url, headers=h, timeout=12, retries=1)
            if r is None or len(r.text) < 2000:
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            for h3 in soup.find_all("h3"):
                a = h3.find("a", href=True)
                if not a:
                    continue
                href = a.get("href", "")
                if not href.endswith(".chn") or not re.search(r'\d{5,}', href):
                    continue
                if href.startswith("/"):
                    href = "https://cafef.vn" + href
                if href in seen_urls:
                    continue
                seen_urls.add(href)

                title = (a.get("title") or a.get_text(strip=True)).strip()
                if len(title) < 12:
                    continue

                # Get snippet
                snippet = ""
                parent = h3.parent
                if parent:
                    for p in parent.find_all("p"):
                        t = p.get_text(strip=True)
                        if len(t) > 30:
                            snippet = t[:300]
                            break

                # Date
                date_str = ""
                if parent:
                    text_all = parent.get_text(" ", strip=True)
                    dm = re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', text_all)
                    if dm:
                        date_str = dm.group(0)

                combined = f"{title} {snippet}"
                broker, rtype = _extract_broker_from_text(combined)
                recommendation, target_price = _extract_recommendation(combined)

                # Filter: prefer analysis articles
                is_analysis = any(kw in combined.lower() for kw in [
                    "phân tích", "báo cáo", "khuyến nghị", "định giá",
                    "mục tiêu", "upside", "ssi", "vci", "hsc", "bsc",
                    "tcbs", "vndirect", "acbs", "mbs", "kis", "shs"
                ])

                results.append({
                    "title":          title,
                    "url":            href,
                    "date":           date_str,
                    "snippet":        snippet,
                    "source":         "CafeF",
                    "broker":         broker,
                    "report_type":    rtype,
                    "recommendation": recommendation,
                    "target_price":   target_price,
                    "is_analysis":    is_analysis,
                })

                if len(results) >= max_items:
                    break

            time.sleep(0.5)

        except Exception as e:
            print(f"[CafeF Analysis] Error {url}: {e}")

    # Sort: ưu tiên bài phân tích
    results.sort(key=lambda x: (not x["is_analysis"], not bool(x["broker"])))
    return results[:max_items]


# ═════════════════════════════════════════════════════════════════════════════
# VIETSTOCK FINANCIAL DATA
# ═════════════════════════════════════════════════════════════════════════════

def fetch_vietstock_financial(ticker: str) -> dict:
    """
    Lấy chỉ số tài chính từ Vietstock.
    Cung cấp: P/E, P/B, EPS, BVPS, ROE, ROA, vốn hóa thị trường.
    """
    try:
        r = _get(
            f"https://finance.vietstock.vn/data/financeinfo",
            params={"code": ticker.upper(), "type": "SUMMARY"},
            headers={
                **_HEADERS_BASE,
                "Referer":    "https://finance.vietstock.vn/",
                "Origin":     "https://finance.vietstock.vn",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=10,
        )
        if r:
            return r.json() or {}
    except Exception as e:
        print(f"[Vietstock] Error: {e}")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# FIREANT FINANCIAL DATA (Public endpoints)
# ═════════════════════════════════════════════════════════════════════════════

FIREANT_BASE = "https://restv2.fireant.vn"


def fetch_fireant_financial_report(ticker: str, report_type: str = "IS",
                                   year: int = None, quarter: int = None) -> List[dict]:
    """
    Lấy báo cáo tài chính từ FireAnt (public endpoint).
    report_type: IS=Income Statement, BS=Balance Sheet, CF=Cash Flow
    """
    try:
        params = {
            "type":     report_type,
            "page":     0,
            "pageSize": 8,
        }
        if year:
            params["year"] = year
        if quarter:
            params["quarter"] = quarter

        r = _get(
            f"{FIREANT_BASE}/symbols/{ticker.upper()}/financial-reports",
            params=params,
            headers={**_HEADERS_BASE, "Referer": "https://fireant.vn/"},
            timeout=10,
        )
        if r:
            data = r.json()
            return data if isinstance(data, list) else data.get("data", [])
    except Exception as e:
        print(f"[FireAnt] Error: {e}")
    return []


def fetch_fireant_snapshot(ticker: str) -> dict:
    """Lấy snapshot thị trường hiện tại từ FireAnt."""
    try:
        r = _get(
            f"{FIREANT_BASE}/symbols/{ticker.upper()}/snapshot",
            headers={**_HEADERS_BASE, "Referer": "https://fireant.vn/"},
            timeout=10,
        )
        if r:
            return r.json() or {}
    except Exception as e:
        print(f"[FireAnt Snapshot] Error: {e}")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# MBS (MBBank Securities) RESEARCH
# ═════════════════════════════════════════════════════════════════════════════

def fetch_mbs_research(ticker: str, max_items: int = 5) -> List[dict]:
    """Lấy báo cáo phân tích từ MBBank Securities."""
    results = []
    try:
        r = _get(
            "https://mbs.com.vn/api/research/search",
            params={"ticker": ticker.upper(), "size": max_items},
            headers={**_HEADERS_BASE, "Referer": "https://mbs.com.vn/"},
            timeout=10,
        )
        if r:
            for item in r.json().get("data", []):
                results.append({
                    "title":          item.get("title", ""),
                    "date":           item.get("publishDate", ""),
                    "recommendation": item.get("recommendation", ""),
                    "target_price":   item.get("targetPrice", ""),
                    "analyst":        item.get("analyst", "MBS Research"),
                    "url":            item.get("url", item.get("fileUrl", "")),
                    "source":         "MBS Research",
                })
    except Exception as e:
        print(f"[MBS] Error: {e}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATOR: MULTI-SOURCE FINANCIAL FETCHER
# ═════════════════════════════════════════════════════════════════════════════

def get_best_financial_statements(
    ticker: str,
    report_type: str = "income",  # income | balance | cashflow | ratio
    yearly: bool = False,
    periods: int = 8,
) -> Tuple[List[dict], str]:
    """
    Lấy báo cáo tài chính từ nguồn tốt nhất.
    Ưu tiên: TCBS → VNDirect → vnstock
    Trả về: (records, source_name)
    """
    ticker = ticker.strip().upper()

    # ── TCBS (nguồn tốt nhất — data rõ ràng, công thức chuẩn) ───────────────
    if report_type == "income":
        records = fetch_tcbs_income_statement(ticker, yearly, periods)
        if records:
            return records, "TCBS"
    elif report_type == "balance":
        records = fetch_tcbs_balance_sheet(ticker, yearly, periods)
        if records:
            return records, "TCBS"
    elif report_type == "cashflow":
        records = fetch_tcbs_cashflow(ticker, yearly, periods)
        if records:
            return records, "TCBS"
    elif report_type == "ratio":
        records = fetch_tcbs_financial_ratios(ticker, yearly, periods)
        if records:
            return records, "TCBS"

    # ── VNDirect finfo (fallback) ─────────────────────────────────────────────
    if report_type == "ratio":
        records = fetch_vndirect_ratios(ticker, periods)
        if records:
            return records, "VNDirect"

    vnd_fs = fetch_vndirect_financial_summary(ticker, periods)
    if vnd_fs:
        return vnd_fs, "VNDirect"

    # ── SSI iBoard (last resort) ──────────────────────────────────────────────
    ssi_data = fetch_ssi_financial_data(ticker)
    if ssi_data.get("data"):
        return ssi_data["data"], "SSI"

    # ── FireAnt (fallback cho BCTC) ───────────────────────────────────────────
    if report_type in ("income", "balance", "cashflow"):
        fa_map = {"income": "IS", "balance": "BS", "cashflow": "CF"}
        fa_data = fetch_fireant_financial_report(ticker, report_type=fa_map[report_type])
        if fa_data:
            return fa_data, "FireAnt"

    return [], "none"


def get_multi_source_research_reports(ticker: str, max_per_source: int = 6) -> List[dict]:
    """
    Tổng hợp báo cáo phân tích từ nhiều nguồn:
    CafeF + VNDirect Research + SSI Research + DNSE/Entrade + MBS
    Dedup và sắp xếp theo mức độ phân tích.
    """
    all_reports: List[dict] = []
    seen_titles: set = set()

    def _add(reports: List[dict]):
        for rp in reports:
            title = rp.get("title", "").strip()
            if not title or len(title) < 10:
                continue
            key = re.sub(r'\s+', ' ', title.lower())[:80]
            if key not in seen_titles:
                seen_titles.add(key)
                all_reports.append(rp)

    # 1. CafeF analysis articles (best coverage for VN stocks)
    try:
        cafef = scrape_cafef_analysis_articles(ticker, max_items=max_per_source)
        _add(cafef)
    except Exception as e:
        print(f"[MultiSource] CafeF error: {e}")

    # 2. VNDirect Research
    try:
        vnd = fetch_vndirect_research_reports(ticker, max_items=max_per_source)
        _add(vnd)
    except Exception as e:
        print(f"[MultiSource] VNDirect error: {e}")

    # 3. SSI Research
    try:
        ssi = fetch_ssi_research_list(ticker, max_items=max_per_source)
        _add(ssi)
    except Exception as e:
        print(f"[MultiSource] SSI error: {e}")

    # 4. DNSE/Entrade
    try:
        dnse = fetch_entrade_research(ticker, max_items=max_per_source)
        _add(dnse)
    except Exception as e:
        print(f"[MultiSource] DNSE error: {e}")

    # 5. MBS Research
    try:
        mbs = fetch_mbs_research(ticker, max_items=3)
        _add(mbs)
    except Exception as e:
        print(f"[MultiSource] MBS error: {e}")

    # Sort: ưu tiên báo cáo có broker, recommendation, target_price
    def _score(rp: dict) -> int:
        score = 0
        if rp.get("broker") or rp.get("analyst"):
            score += 4
        if rp.get("recommendation"):
            score += 3
        if rp.get("target_price"):
            score += 2
        if rp.get("is_analysis"):
            score += 1
        return score

    all_reports.sort(key=_score, reverse=True)
    return all_reports


def get_valuation_snapshot(ticker: str) -> dict:
    """
    Lấy snapshot định giá tổng hợp:
    P/E, P/B, EV/EBITDA, ROE, ROA, EPS, vốn hóa từ nhiều nguồn.
    """
    result = {"ticker": ticker, "sources": [], "metrics": {}}

    # TCBS ratios (most recent quarter)
    tcbs_ratios = fetch_tcbs_financial_ratios(ticker, yearly=False, periods=1)
    if tcbs_ratios:
        r = tcbs_ratios[0]
        result["sources"].append("TCBS")
        result["period"] = f"Q{r.get('quarter','?')}/{r.get('year','?')}"
        result["metrics"].update({
            "P/E":              r.get("priceToEarning"),
            "P/B":              r.get("priceToBook"),
            "EV/EBITDA":        r.get("valueBeforeEbitda"),
            "ROE (%)":          r.get("roe"),
            "ROA (%)":          r.get("roa"),
            "EPS (đồng)":       r.get("earningPerShare"),
            "BVPS (đồng)":      r.get("bookValuePerShare"),
            "Cổ tức (%)":       r.get("dividendYield"),
            "Biên LN gộp (%)":  r.get("grossProfitMargin"),
            "Biên LN ròng (%)": r.get("postTaxMargin"),
            "D/E":              r.get("debtOnEquity"),
            "Current Ratio":    r.get("currentPayment"),
        })

    # VNDirect ratios (complement)
    vnd_ratios = fetch_vndirect_ratios(ticker, periods=1)
    if vnd_ratios:
        r = vnd_ratios[0]
        result["sources"].append("VNDirect")
        for k, vk in [("P/E", "pe"), ("P/B", "pb"), ("EPS", "eps"),
                      ("ROE (%)", "roe"), ("ROA (%)", "roa"),
                      ("D/E", "debtToEquity"), ("Current Ratio", "currentRatio")]:
            if k not in result["metrics"] or result["metrics"][k] is None:
                result["metrics"][k] = r.get(vk)

    # FireAnt snapshot (market cap, real-time valuation)
    fa_snap = fetch_fireant_snapshot(ticker)
    if fa_snap:
        result["sources"].append("FireAnt")
        result["metrics"]["Vốn hóa (tỷ đồng)"] = fa_snap.get("marketCap")
        if not result["metrics"].get("P/E"):
            result["metrics"]["P/E"] = fa_snap.get("pe")

    # Remove None metrics
    result["metrics"] = {k: v for k, v in result["metrics"].items() if v is not None}
    return result

# ═════════════════════════════════════════════════════════════════════════════
# FIREANT COMPANY PROFILE & FUNDAMENTAL
# ═════════════════════════════════════════════════════════════════════════════

def fetch_fireant_company_profile(ticker: str) -> dict:
    """Lấy profile công ty từ FireAnt (website, sector, description)."""
    try:
        r = _get(
            f"{FIREANT_BASE}/symbols/{ticker.upper()}/profile",
            headers={**_HEADERS_BASE, "Referer": "https://fireant.vn/"},
            timeout=10,
        )
        if r:
            return r.json() or {}
    except Exception as e:
        print(f"[FireAnt Profile] Error: {e}")
    return {}


def fetch_fireant_fundamental(ticker: str) -> dict:
    """Lấy chỉ số cơ bản từ FireAnt (P/E, P/B, ROE, market cap)."""
    try:
        r = _get(
            f"{FIREANT_BASE}/symbols/{ticker.upper()}/fundamental",
            headers={**_HEADERS_BASE, "Referer": "https://fireant.vn/"},
            timeout=10,
        )
        if r:
            return r.json() or {}
    except Exception as e:
        print(f"[FireAnt Fundamental] Error: {e}")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE MULTI-SOURCE COMPANY INFO
# ═════════════════════════════════════════════════════════════════════════════

def _first_val(d: dict, *keys) -> str:
    """Lấy giá trị đầu tiên không rỗng theo danh sách keys."""
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def fetch_comprehensive_company_info(ticker: str) -> dict:
    """
    Tổng hợp thông tin đầy đủ của công ty từ TCBS, VNDirect, FireAnt, SSI.
    Trả về dict với field names nhất quán.
    """
    ticker = ticker.upper()
    info: dict = {"ticker": ticker}

    # ── 1. TCBS Company Profile ──────────────────────────────────────────
    try:
        tcbs = fetch_tcbs_company_overview(ticker)
        if tcbs:
            info["ten_cong_ty"]    = _first_val(tcbs, "companyName", "organName", "organ_name",
                                                "company_name", "short_name")
            info["ten_tieng_anh"]  = _first_val(tcbs, "companyNameEng", "enOrganName",
                                                "en_organ_name")
            info["ten_viet_tat"]   = _first_val(tcbs, "shortName", "short_name")
            info["nganh"]          = _first_val(tcbs, "industryVi", "industry_vi",
                                                "industryEn", "industry",
                                                "icbName3", "icbName2", "sector",
                                                "industry_name")
            info["website"]        = _first_val(tcbs, "website", "webUrl", "web_url", "homeUrl")
            info["san_niem_yet"]   = _first_val(tcbs, "exchange", "floor", "exchangeCode")
            info["ngay_niem_yet"]  = _first_val(tcbs, "firstIssueDate", "listedDate",
                                                "listingDate", "listed_date")
            profile_text = _first_val(tcbs, "companyProfile", "businessActivities",
                                      "profile", "introduction", "description")
            if profile_text:
                info["gioi_thieu"] = profile_text[:600]
            # Charter capital — handle both raw VND and already-in-billion
            for k in ("charterCapital", "charter_capital", "capitalAmount", "capital_amount"):
                raw = tcbs.get(k)
                if raw is not None:
                    try:
                        fv = float(raw)
                        info["von_dieu_le_ty"] = round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1)
                    except Exception:
                        pass
                    break
            # Outstanding shares
            for k in ("outstandingShare", "outstanding_share", "shareAmount",
                       "listedShare", "listed_share"):
                raw = tcbs.get(k)
                if raw is not None:
                    try:
                        fv = float(raw)
                        info["co_phieu_luu_hanh_trieu"] = (
                            round(fv / 1e6, 2) if fv > 1e6 else round(fv, 2))
                    except Exception:
                        pass
                    break
            # Market cap
            for k in ("marketCap", "market_cap", "marketCapitalization"):
                raw = tcbs.get(k)
                if raw is not None:
                    try:
                        fv = float(raw)
                        info["market_cap_ty_dong"] = round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1)
                    except Exception:
                        pass
                    break
    except Exception as e:
        print(f"[CompanyInfo] TCBS error: {e}")

    # ── 2. VNDirect stocks API ───────────────────────────────────────────
    try:
        fields = (
            "code,companyName,shortName,companyNameEng,isin,floor,"
            "listedDate,taxCode,status,industryName,website,companyProfile"
        )
        r = _get(
            f"{VND_BASE}/stocks",
            params={"q": f"code:{ticker}", "fields": fields},
            headers=_HEADERS_VND,
        )
        if r:
            items = (r.json() or {}).get("data", [])
            if items:
                item = items[0]
                if not info.get("ten_cong_ty"):
                    info["ten_cong_ty"]   = item.get("companyName", "")
                if not info.get("ten_tieng_anh"):
                    info["ten_tieng_anh"] = item.get("companyNameEng", "")
                if not info.get("san_niem_yet"):
                    info["san_niem_yet"]  = item.get("floor", "")
                if not info.get("ngay_niem_yet"):
                    info["ngay_niem_yet"] = item.get("listedDate", "")
                if not info.get("nganh"):
                    info["nganh"]         = item.get("industryName", "")
                if not info.get("website"):
                    info["website"]       = item.get("website", "")
                if not info.get("gioi_thieu"):
                    p = item.get("companyProfile", "")
                    if p:
                        info["gioi_thieu"] = str(p)[:600]
                info.setdefault("isin",       item.get("isin", ""))
                info.setdefault("ma_so_thue", item.get("taxCode", ""))
    except Exception as e:
        print(f"[CompanyInfo] VNDirect error: {e}")

    # ── 3. FireAnt Company Profile ───────────────────────────────────────
    try:
        fa = fetch_fireant_company_profile(ticker)
        if fa:
            if not info.get("ten_cong_ty"):
                info["ten_cong_ty"] = _first_val(fa, "companyName", "name", "shortName")
            if not info.get("nganh"):
                info["nganh"] = _first_val(fa, "industryName", "sector", "icbName", "industry")
            if not info.get("website"):
                info["website"] = _first_val(fa, "website", "webUrl", "web_url")
            if not info.get("gioi_thieu"):
                desc = _first_val(fa, "description", "companyProfile", "businessActivities", "profile")
                if desc:
                    info["gioi_thieu"] = desc[:600]
            if not info.get("san_niem_yet"):
                info["san_niem_yet"] = _first_val(fa, "exchange", "floor")
            if not info.get("von_dieu_le_ty"):
                raw = fa.get("charterCapital") or fa.get("capitalAmount")
                if raw:
                    try:
                        fv = float(raw)
                        info["von_dieu_le_ty"] = round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[CompanyInfo] FireAnt profile error: {e}")

    # ── 4. SSI Company Info ──────────────────────────────────────────────
    try:
        ssi = fetch_ssi_company_info(ticker)
        if ssi:
            inner = ssi.get("data", ssi) if isinstance(ssi, dict) else {}
            if isinstance(inner, dict):
                if not info.get("gioi_thieu"):
                    desc = _first_val(inner, "companyProfile", "description", "businessActivities")
                    if desc:
                        info["gioi_thieu"] = desc[:600]
                if not info.get("nganh"):
                    info["nganh"] = _first_val(inner, "industryName", "sector", "industry")
                if not info.get("website"):
                    info["website"] = _first_val(inner, "website", "webUrl")
                so_nv = inner.get("employee") or inner.get("numberOfEmployee")
                if so_nv:
                    info.setdefault("so_nhan_vien", so_nv)
    except Exception as e:
        print(f"[CompanyInfo] SSI error: {e}")

    # ── 5. vnstock company.overview() + company.profile() fallback ──────────
    if not info.get("ten_cong_ty") or not info.get("nganh") or not info.get("gioi_thieu"):
        try:
            from vnstock import Vnstock
            vn = Vnstock()
            for src in ["VCI", "KBS"]:
                try:
                    _stock = vn.stock(symbol=ticker, source=src)
                    try:
                        df_ov = _stock.company.overview()
                        if df_ov is not None and len(df_ov) > 0:
                            row = df_ov.iloc[0].to_dict() if hasattr(df_ov, "iloc") else {}
                            if not info.get("ten_cong_ty"):
                                info["ten_cong_ty"] = str(
                                    row.get("company_name") or row.get("short_name") or "")
                            if not info.get("nganh"):
                                info["nganh"] = str(
                                    row.get("industry_name") or row.get("icb_name3") or
                                    row.get("icb_name2") or "")
                            if not info.get("san_niem_yet"):
                                info["san_niem_yet"] = str(row.get("exchange") or row.get("floor") or "")
                            if not info.get("website"):
                                info["website"] = str(row.get("website") or "")
                            for ck in ("charter_capital", "charterCapital"):
                                if row.get(ck) and not info.get("von_dieu_le_ty"):
                                    try:
                                        fv = float(row[ck])
                                        info["von_dieu_le_ty"] = round(fv/1e9,1) if fv>1e9 else round(fv,1)
                                    except Exception: pass
                                    break
                            for ok in ("outstanding_share", "outstandingShare", "listed_share"):
                                if row.get(ok) and not info.get("co_phieu_luu_hanh_trieu"):
                                    try:
                                        fv = float(row[ok])
                                        info["co_phieu_luu_hanh_trieu"] = round(fv/1e6,2) if fv>1e6 else round(fv,2)
                                    except Exception: pass
                                    break
                    except Exception: pass
                    try:
                        df_pr = _stock.company.profile()
                        if df_pr is not None and len(df_pr) > 0:
                            row = df_pr.iloc[0].to_dict() if hasattr(df_pr, "iloc") else {}
                            if not info.get("gioi_thieu"):
                                desc = (row.get("company_profile") or row.get("history_dev") or
                                        row.get("business_strategies") or "")
                                if desc:
                                    info["gioi_thieu"] = str(desc)[:600]
                    except Exception: pass
                    if info.get("ten_cong_ty") and info["ten_cong_ty"] != ticker:
                        break
                except Exception: continue
        except Exception as e:
            print(f"[CompanyInfo] vnstock fallback: {e}")

    # Clean empty values
    return {k: v for k, v in info.items() if v is not None and v != ""}


def fetch_comprehensive_financial_ratios(ticker: str) -> dict:
    """
    Tổng hợp chỉ số tài chính chuẩn hóa từ TCBS → VNDirect → FireAnt.
    ROE/ROA/Biên lợi nhuận: đã chuyển sang %, không cần nhân 100 thêm.
    """
    ticker = ticker.upper()
    m: dict = {}

    def _pct(v):
        """Chuyển giá trị 0-1 sang %, nếu đã là % giữ nguyên."""
        if v is None:
            return None
        try:
            fv = float(v)
            return round(fv * 100, 2) if abs(fv) <= 1.5 else round(fv, 2)
        except Exception:
            return None

    # ── 1. TCBS ratios (nguồn tốt nhất) ─────────────────────────────────
    try:
        records = fetch_tcbs_financial_ratios(ticker, yearly=False, periods=1)
        if records:
            r = records[0]
            m["pe"]             = r.get("priceToEarning")
            m["pb"]             = r.get("priceToBook")
            m["ev_ebitda"]      = r.get("valueBeforeEbitda")
            m["roe"]            = _pct(r.get("roe"))
            m["roa"]            = _pct(r.get("roa"))
            m["eps"]            = r.get("earningPerShare")
            m["bvps"]           = r.get("bookValuePerShare")
            m["dividend_yield"] = _pct(r.get("dividendYield"))
            m["net_margin"]     = _pct(r.get("postTaxMargin"))
            m["gross_margin"]   = _pct(r.get("grossProfitMargin"))
            m["de_ratio"]       = r.get("debtOnEquity")
            m["current_ratio"]  = r.get("currentPayment")
            m["ky"]             = f"Q{r.get('quarter', '?')}/{r.get('year', '?')}"
    except Exception as e:
        print(f"[Ratios] TCBS error: {e}")

    # ── 2. VNDirect ratios (bổ sung) ─────────────────────────────────────
    try:
        records = fetch_vndirect_ratios(ticker, periods=1)
        if records:
            r = records[0]
            if m.get("pe") is None:
                m["pe"] = r.get("pe")
            if m.get("pb") is None:
                m["pb"] = r.get("pb")
            if m.get("roe") is None and r.get("roe") is not None:
                m["roe"] = _pct(r["roe"])
            if m.get("roa") is None and r.get("roa") is not None:
                m["roa"] = _pct(r["roa"])
            if m.get("eps") is None:
                m["eps"] = r.get("eps")
            if m.get("de_ratio") is None:
                m["de_ratio"] = r.get("debtToEquity")
    except Exception as e:
        print(f"[Ratios] VNDirect error: {e}")

    # ── 3. FireAnt fundamental (bổ sung market cap + missing ratios) ─────
    try:
        fa_fund = fetch_fireant_fundamental(ticker)
        if fa_fund:
            if m.get("pe") is None:
                m["pe"] = fa_fund.get("pe") or fa_fund.get("P/E")
            if m.get("pb") is None:
                m["pb"] = fa_fund.get("pb") or fa_fund.get("P/B")
            mc_raw = fa_fund.get("marketCap") or fa_fund.get("market_cap")
            if mc_raw:
                try:
                    fv = float(mc_raw)
                    m["market_cap_ty"] = round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Ratios] FireAnt fundamental error: {e}")

    # ── 4. vnstock VCI/KBS ratio fallback (nếu TCBS + VNDirect + FireAnt đều thiếu P/E) ──
    if not m.get("pe") and not m.get("roe"):
        try:
            from vnstock import Vnstock
            vn = Vnstock()
            for src in ["VCI", "KBS"]:
                try:
                    stock = vn.stock(symbol=ticker, source=src)
                    df_ratio = stock.finance.ratio(period="quarter")
                    if df_ratio is None or df_ratio.empty:
                        continue
                    row = df_ratio.iloc[0].to_dict() if hasattr(df_ratio, "iloc") else {}
                    for col_name, val in row.items():
                        if val is None:
                            continue
                        try:
                            fv = float(val)
                        except (ValueError, TypeError):
                            continue
                        cl = str(col_name).lower()
                        if cl in ("p/e", "pe", "pricetoearning", "price_to_earning"):
                            m.setdefault("pe", round(fv, 2))
                        elif cl in ("p/b", "pb", "pricetobook", "price_to_book"):
                            m.setdefault("pb", round(fv, 2))
                        elif "roe" in cl:
                            m.setdefault("roe", _pct(fv))
                        elif "roa" in cl:
                            m.setdefault("roa", _pct(fv))
                        elif "eps" in cl and "bvps" not in cl and "growth" not in cl:
                            m.setdefault("eps", round(fv, 0))
                        elif "bvps" in cl or "book_value_per" in cl:
                            m.setdefault("bvps", round(fv, 0))
                        elif "dividend" in cl or "co_tuc" in cl:
                            m.setdefault("dividend_yield", _pct(fv))
                        elif "net_margin" in cl or "postTaxMargin" in cl.replace("_",""):
                            m.setdefault("net_margin", _pct(fv))
                        elif "gross_margin" in cl or "grossProfitMargin" in cl.replace("_",""):
                            m.setdefault("gross_margin", _pct(fv))
                        elif "debt_on_equity" in cl or "debttoequity" in cl.replace("_",""):
                            m.setdefault("de_ratio", round(fv, 2))
                        elif "market_cap" in cl or "marketcap" in cl.replace("_",""):
                            if fv > 1e9:
                                m.setdefault("market_cap_ty", round(fv / 1e9, 1))
                            elif fv > 1000:
                                m.setdefault("market_cap_ty", round(fv, 1))
                    if m.get("pe") or m.get("roe"):
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"[Ratios] vnstock VCI/KBS fallback: {e}")

    # ── 5. FireAnt snapshot (market cap nếu chưa có) ─────────────────────
    try:
        fa_snap = fetch_fireant_snapshot(ticker)
        if fa_snap:
            if m.get("pe") is None:
                m["pe"] = fa_snap.get("pe") or fa_snap.get("priceToEarning")
            if m.get("market_cap_ty") is None:
                mc_raw = fa_snap.get("marketCap")
                if mc_raw:
                    try:
                        fv = float(mc_raw)
                        m["market_cap_ty"] = round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[Ratios] FireAnt snapshot error: {e}")

    return {k: v for k, v in m.items() if v is not None}