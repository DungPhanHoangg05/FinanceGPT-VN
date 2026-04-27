import json
import math
import re
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import requests
from langchain_core.tools import tool

# ── Import multi-source data layer ────────────────────────────────────────────
try:
    from data_sources import (
        fetch_tcbs_income_statement,
        fetch_tcbs_cashflow,
        fetch_tcbs_financial_ratios,
        fetch_vndirect_analyst_recs,
        fetch_vndirect_ownership,
        fetch_vndirect_dividends,
        get_multi_source_research_reports,
        get_valuation_snapshot,
    )
    _DATA_SOURCES_AVAILABLE = True
except ImportError:
    _DATA_SOURCES_AVAILABLE = False

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_period(period: str = "3m") -> tuple[str, str]:
    end = datetime.now()
    p = (period or "3m").lower().strip()
    PERIOD_MAP = {
        "1w": 7, "1 tuần": 7, "tuần": 7,
        "1m": 30, "1 tháng": 30, "tháng": 30,
        "3m": 90, "3 tháng": 90,
        "6m": 180, "6 tháng": 180,
        "1y": 365, "1 năm": 365, "năm": 365,
        "2y": 730, "2 năm": 730,
    }
    if p in ("ytd", "đầu năm"):
        start = datetime(end.year, 1, 1)
    else:
        days = PERIOD_MAP.get(p, 90)
        start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _df_to_compressed(df: pd.DataFrame, max_rows: int = 200) -> list[dict]:
    out = []
    step = max(1, len(df) // max_rows)
    for i in range(0, len(df), step):
        row = df.iloc[i]
        o, h, l, c = [float(row[k]) for k in ["Open", "High", "Low", "Close"]]
        rec = {
            "d": str(row["Datetime"])[:10],
            "o": int(round(o * 1000)) if o < 10000 else int(round(o)),
            "h": int(round(h * 1000)) if h < 10000 else int(round(h)),
            "l": int(round(l * 1000)) if l < 10000 else int(round(l)),
            "c": int(round(c * 1000)) if c < 10000 else int(round(c)),
        }
        if "Volume" in df.columns and pd.notna(row.get("Volume")):
            rec["v"] = int(row["Volume"])
        out.append(rec)
    return out


def _auto_unit_billions(val):
    """Tự động đổi sang tỷ đồng nếu giá trị quá lớn (raw VND)."""
    if val is None:
        return None
    try:
        f = float(val)
        if abs(f) > 1e10:          # > 10 tỷ → chắc chắn là raw VND
            return round(f / 1e9, 1)
        elif abs(f) > 1e7:         # > 10 triệu → có thể là tỷ rồi
            return round(f, 1)
        return round(f, 1)
    except Exception:
        return val


def _norm_p(p):
    """Chuẩn hóa giá cổ phiếu và trả về chuỗi định dạng (135.0 -> '135,000')."""
    if p is None: return "0"
    try:
        f = float(p)
        # Cổ phiếu VN: nếu < 10000 (đơn vị nghìn đồng), nhân 1000 và làm tròn
        val = int(round(f * 1000)) if f < 10000 else int(round(f))
        return "{:,}".format(val)
    except:
        return str(p)

def _norm_index(p):
    """Chuẩn hóa chỉ số và trả về chuỗi định dạng (1.28 -> '1,280.00')."""
    if p is None: return "0"
    try:
        f = float(p)
        # Chỉ số VN: nếu < 10 (ví dụ 1.28), nhân 1000. 
        val = f * 1000 if f < 10 else f
        return "{:,.2f}".format(val)
    except:
        return str(p)


# ── VCI / KBS Finance (vnstock API cập nhật theo phiên bản mới) ───────────────

def _fetch_finance(ticker: str, source: str, statement_type: str, yearly: bool) -> Optional[pd.DataFrame]:
    """
    Gọi vnstock Finance API dựa trên cấu trúc chuẩn của vnstock mới.
    """
    period_str = "year" if yearly else "quarter"
    try:
        from vnstock import Finance
        
        # Khởi tạo class Finance chuẩn theo tài liệu mới nhất
        fin = Finance(symbol=ticker.upper(), source=source)
        
        method_map = {
            "income":   getattr(fin, "income_statement", None),
            "balance":  getattr(fin, "balance_sheet", None),
            "cashflow": getattr(fin, "cash_flow", None),
            "ratio":    getattr(fin, "ratio", None),
        }
        
        method = method_map.get(statement_type)
        if method is None:
            return None
            
        # VCI hỗ trợ tham số lang='vi' và dropna=True
        if source == "VCI":
            try:
                df = method(period=period_str, lang='vi', dropna=True)
            except TypeError:
                df = method(period=period_str)
        else: # KBS
            df = method(period=period_str)

        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        return df
    except Exception as e:
        print(f"[Finance/{source}] {ticker} {statement_type}: {e}")
        return None


def _fetch_vndirect_company_meta(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    if not ticker:
        return {}
    try:
        url = (
            f"https://api-finfo.vndirect.com.vn/v4/stocks"
            f"?q=code:{ticker}"
            f"&fields=companyId,companyName,shortName,companyNameEng,"
            f"isin,floor,listedDate,indexCode,taxCode,status"
        )
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=8)
        if resp.status_code != 200:
            return {}
        data = resp.json().get("data") or []
        if not data:
            return {}
        return data[0] if isinstance(data[0], dict) else {}
    except Exception:
        return {}


# ── Industry-based enrichment templates (fallbacks) ─────────────────────────
def _industry_profile_templates() -> dict:
    return {
        "vingroup": {
            "mo_ta": (
                "Vingroup là tập đoàn kinh tế tư nhân đa ngành lớn nhất Việt Nam, "
                "hoạt động trong các lĩnh vực: bất động sản (Vinhomes), sản xuất ô tô điện (VinFast), "
                "công nghệ (VinTech), bán lẻ (VinMart), y tế (Vinmec) và giáo dục (Vinschool)."
            ),
            "linh_vuc_hoat_dong": (
                "Bất động sản, sản xuất ô tô điện (VinFast), công nghệ thông tin (VinTech), "
                "bán lẻ (VinMart), y tế (Vinmec), giáo dục (Vinschool)."
            ),
            "san_pham_dich_vu": (
                "Khu đô thị & căn hộ Vinhomes, xe điện VinFast, "
                "trung tâm thương mại Vincom, bệnh viện Vinmec, trường học Vinschool."
            ),
            "thi_truong": (
                "Thị trường trong nước và quốc tế; VinFast xuất khẩu xe điện sang Mỹ, Canada, châu Âu."
            ),
        },
        "tập đoàn": {
            "mo_ta": (
                "Tập đoàn kinh tế đa ngành hoạt động trong nhiều lĩnh vực: "
                "bất động sản, công nghiệp, thương mại, dịch vụ và đầu tư tài chính."
            ),
            "linh_vuc_hoat_dong": "Đầu tư đa ngành: bất động sản, công nghiệp, thương mại, dịch vụ.",
            "san_pham_dich_vu": "Bất động sản, sản phẩm công nghiệp, dịch vụ thương mại và đầu tư.",
            "thi_truong": "Thị trường trong nước và quốc tế theo từng lĩnh vực kinh doanh.",
        },
        "ngân hàng": {
            "mo_ta": (
                "Ngân hàng thương mại cung cấp dịch vụ tài chính: huy động vốn, cho vay, thanh toán, "
                "dịch vụ thẻ và ngân hàng điện tử cho khách hàng cá nhân và doanh nghiệp."
            ),
            "linh_vuc_hoat_dong": "Ngân hàng thương mại: huy động, cho vay, thanh toán, quản lý tài sản.",
            "san_pham_dich_vu": "Tài khoản thanh toán/tiết kiệm, cho vay cá nhân/doanh nghiệp, thẻ, dịch vụ thanh toán điện tử.",
            "thi_truong": "Hoạt động chủ yếu trên thị trường nội địa; phục vụ khách hàng cá nhân và doanh nghiệp; cạnh tranh cao giữa ngân hàng thương mại."
        },
        "công nghệ": {
            "mo_ta": "Doanh nghiệp hoạt động trong lĩnh vực công nghệ thông tin: phát triển phần mềm, giải pháp CNTT, gia công phần mềm và chuyển đổi số.",
            "linh_vuc_hoat_dong": "Phát triển phần mềm, gia công phần mềm (outsourcing), giải pháp chuyển đổi số.",
            "san_pham_dich_vu": "Phần mềm doanh nghiệp, dịch vụ chuyển đổi số, dịch vụ đám mây, tư vấn CNTT.",
            "thi_truong": "Thị trường trong nước và xuất khẩu dịch vụ phần mềm sang châu Á, châu Âu và Bắc Mỹ."
        },
        "bất động sản": {
            "mo_ta": "Hoạt động chính: phát triển dự án bất động sản, cho thuê, phân phối và quản lý tài sản.",
            "linh_vuc_hoat_dong": "Đầu tư phát triển dự án, kinh doanh bất động sản dân cư và thương mại.",
            "san_pham_dich_vu": "Dự án nhà ở, trung tâm thương mại, cho thuê bất động sản, dịch vụ quản lý tài sản.",
            "thi_truong": "Thị trường bất động sản trong nước, tập trung vào phát triển đô thị và khu công nghiệp."
        },
        "bán lẻ": {
            "mo_ta": "Cung cấp sản phẩm tiêu dùng và dịch vụ bán lẻ qua chuỗi cửa hàng và kênh trực tuyến.",
            "linh_vuc_hoat_dong": "Bán lẻ truyền thống, thương mại điện tử, chuỗi phân phối.",
            "san_pham_dich_vu": "Hàng tiêu dùng, điện tử, thiết bị, logistic cho bán lẻ.",
            "thi_truong": "Thị trường tiêu dùng trong nước, kênh bán lẻ truyền thống và trực tuyến."
        },
        "viễn thông": {
            "mo_ta": "Cung cấp dịch vụ viễn thông: thoại, dữ liệu, internet, giải pháp hạ tầng mạng.",
            "linh_vuc_hoat_dong": "Dịch vụ di động, Internet băng thông rộng, hạ tầng viễn thông.",
            "san_pham_dich_vu": "Dịch vụ di động, Internet, dịch vụ doanh nghiệp như VPN, cloud connectivity.",
            "thi_truong": "Thị trường viễn thông trong nước; cạnh tranh giữa các nhà cung cấp lớn."
        },
        "sản xuất": {
            "mo_ta": "Hoạt động sản xuất công nghiệp: chế tạo, lắp ráp và xuất khẩu sản phẩm công nghiệp.",
            "linh_vuc_hoat_dong": "Sản xuất hàng tiêu dùng, công nghiệp và chế tạo.",
            "san_pham_dich_vu": "Sản phẩm công nghiệp, linh kiện, xuất khẩu gia công.",
            "thi_truong": "Thị trường nội địa và xuất khẩu tùy theo ngành hàng."
        },
        "thực phẩm": {
            "mo_ta": "Sản xuất và phân phối thực phẩm, chế biến nông sản và đồ ăn đóng gói.",
            "linh_vuc_hoat_dong": "Chế biến thực phẩm, sản xuất đồ uống, phân phối.",
            "san_pham_dich_vu": "Thực phẩm đóng gói, đồ uống, chế biến nông sản.",
            "thi_truong": "Thị trường tiêu dùng trong nước và xuất khẩu thực phẩm chế biến."
        }
    }


def _apply_industry_profile(result: dict) -> None:
    """Fill missing descriptive fields from templates based on `nganh` or `ten_cong_ty`."""
    if not isinstance(result, dict):
        return
    try:
        industry_str = (result.get("nganh") or "") + " " + (result.get("ten_cong_ty") or "")
        s = industry_str.lower()
        templates = _industry_profile_templates()

        # Priority: check specific company names first
        priority_keys = ["vingroup", "tập đoàn", "ngân hàng", "công nghệ",
                         "bất động sản", "bán lẻ", "viễn thông", "sản xuất", "thực phẩm"]
        matched = None
        for key in priority_keys:
            if key in s:
                matched = templates.get(key)
                break

        # Heuristic: company name/industry contains "bank" or "ngân hàng"
        if not matched and ("ngân hàng" in s or "bank" in s):
            matched = templates.get("ngân hàng")

        if matched:
            if not result.get("gioi_thieu"):
                result["gioi_thieu"] = matched.get("mo_ta")
            if not result.get("linh_vuc_hoat_dong"):
                result["linh_vuc_hoat_dong"] = matched.get("linh_vuc_hoat_dong")
            if not result.get("san_pham_dich_vu"):
                result["san_pham_dich_vu"] = matched.get("san_pham_dich_vu")
            if not result.get("thi_truong"):
                result["thi_truong"] = matched.get("thi_truong")
        else:
            if not result.get("gioi_thieu"):
                result["gioi_thieu"] = "Doanh nghiệp hoạt động trong lĩnh vực liên quan đến hoạt động kinh doanh đã đăng ký."
            if not result.get("linh_vuc_hoat_dong"):
                result["linh_vuc_hoat_dong"] = "Hoạt động kinh doanh chính: sản xuất/kinh doanh/dịch vụ theo ngành đăng ký."
    except Exception:
        return

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 - Company Info (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

# Helpers to normalize vnstock company data (overview, shareholders, officers, subsidiaries)
def _df_to_records_safe(obj):
    if obj is None:
        return []
    try:
        # pandas DataFrame
        if hasattr(obj, "to_dict"):
            return obj.to_dict(orient="records")
    except Exception:
        pass
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []


def _get_candidate_value(rec: dict, candidates: list):
    # Try direct key matches, then case-insensitive matches
    for c in candidates:
        if c in rec and rec[c] not in (None, "", np.nan):
            return rec[c]
    for k, v in rec.items():
        if str(k).lower() in [c.lower() for c in candidates] and v not in (None, "", np.nan):
            return v
    return None


def _coerce_number(v):
    try:
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        fv = float(v)
        if abs(fv - int(fv)) < 1e-9:
            return int(fv)
        return fv
    except Exception:
        return v


def _coerce_percent(v):
    try:
        fv = float(v)
        # If 0-1 style (e.g., 0.12), convert to percent
        if abs(fv) <= 1.5:
            fv = fv * 100
        return round(fv, 4)
    except Exception:
        return v


def _normalize_records(records: list[dict], mapping: dict) -> list[dict]:
    out = []
    for rec in records:
        nr = {}
        for dest, cands in mapping.items():
            val = _get_candidate_value(rec, cands)
            if val is None:
                continue
            nr[dest] = val
        out.append(nr)
    return out


class Company:
    """Lightweight Company wrapper to fetch/normalize company data from Vnstock (KBS/VCI).

    Usage:
        company = Company(symbol='VCI', source='KBS')
        df = company.overview()
        sh = company.shareholders()
        officers = company.officers()
        subs = company.subsidiaries()  # KBS only
    """

    def __init__(self, symbol: str, source: str = "KBS"):
        self.symbol = (symbol or "").strip().upper()
        self.source = (source or "KBS").strip().upper()
        self._vn = None

    def _ensure_vn(self):
        if self._vn is not None:
            return True
        try:
            from vnstock import Vnstock
            self._vn = Vnstock()
            return True
        except Exception:
            self._vn = None
            return False

    def _safe_stock(self):
        if not self._ensure_vn():
            return None
        try:
            return self._vn.stock(symbol=self.symbol, source=self.source)
        except Exception:
            try:
                return self._vn.stock(symbol=self.symbol, source=self.source.upper())
            except Exception:
                return None

    def _map_records_to_df(self, records: list[dict], cols: list[str], mapping: dict) -> pd.DataFrame:
        rows = []
        for rec in records:
            row = {}
            for col in cols:
                # mapping may have candidate keys per column
                cands = mapping.get(col, [col])
                val = _get_candidate_value(rec, cands)
                row[col] = val if val is not None else None
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    def overview(self) -> pd.DataFrame:
        """Return company overview as DataFrame using column sets per source."""
        src = self.source
        stock = self._safe_stock()
        records = []
        if stock:
            try:
                df = stock.company.overview()
                records = _df_to_records_safe(df)
            except Exception:
                records = []

        # Fallback: try the module-level get_company_info to extract cached standardized block
        if not records:
            try:
                import json
                js = json.loads(get_company_info(self.symbol))
                comp = js.get("company") if isinstance(js, dict) else None
                if comp and comp.get(src) and comp[src].get("overview"):
                    std = comp[src]["overview"].get("standardized") or comp[src]["overview"].get("raw")
                    if std:
                        records = std
            except Exception:
                pass

        if src == "KBS":
            cols = [
                "business_model","symbol","founded_date","charter_capital","number_of_employees",
                "listing_date","par_value","exchange","listing_price","listed_volume",
                "ceo_name","ceo_position","inspector_name","inspector_position","establishment_license",
                "business_code","tax_id","auditor","company_type","address","phone","fax","email",
                "website","branches","history","free_float_percentage","free_float","outstanding_shares","as_of_date",
            ]
            mapping = {
                "business_model": ["business_model","businessModel","mo_hinh_kinh_doanh","mo_ta","company_profile"],
                "symbol": ["symbol","ticker","code"],
                "founded_date": ["founded_date","foundedDate","established_date","establishmentDate","founded"],
                "charter_capital": ["charter_capital","charterCapital","capitalAmount","capital","von_dieu_le"],
                "number_of_employees": ["number_of_employees","employees","employee_count","staff"],
                "listing_date": ["listing_date","listed_date","listedDate","listingDate"],
                "par_value": ["par_value","parValue","face_value"],
                "exchange": ["exchange","floor","listed_floor","san","san_niem_yet"],
                "listing_price": ["listing_price","listed_price","listingPrice"],
                "listed_volume": ["listed_volume","listedShare","outstanding_share","outstandingShare","listed_share"],
                "ceo_name": ["ceo_name","ceo","ceoName","chairman"],
                "ceo_position": ["ceo_position","position","position_en"],
                "inspector_name": ["inspector_name","inspectorName","controller"],
                "inspector_position": ["inspector_position","inspectorPosition"],
                "establishment_license": ["establishment_license","establishmentLicense","license"],
                "business_code": ["business_code","businessCode","industry_code","ma_nganh"],
                "tax_id": ["tax_id","taxId","tax_code","ma_so_thue"],
                "auditor": ["auditor","auditor_name","auditorName"],
                "company_type": ["company_type","companyType","type"],
                "address": ["address","company_address","head_office"],
                "phone": ["phone","telephone","phone_number"],
                "fax": ["fax","fax_number"],
                "email": ["email","contact_email","email_address"],
                "website": ["website","homeUrl","webUrl","web"],
                "branches": ["branches","branch","subsidiaries"],
                "history": ["history","history_dev","company_history"],
                "free_float_percentage": ["free_float_percentage","freeFloatPercent","free_float_pct"],
                "free_float": ["free_float","freeFloat","free_float_shares"],
                "outstanding_shares": ["outstanding_shares","outstandingShare","listed_share","shareAmount"],
                "as_of_date": ["as_of_date","asOfDate","update_date","last_updated"],
            }
        else:
            # VCI
            cols = ["symbol","id","issue_share","history","company_profile","icb_name3","icb_name2","icb_name4","financial_ratio_issue_share","charter_capital"]
            mapping = {
                "symbol": ["symbol","ticker","code"],
                "id": ["id","companyId","company_id"],
                "issue_share": ["issue_share","issueShare","issue_shares","issueShare"],
                "history": ["history","history_dev","company_history"],
                "company_profile": ["company_profile","companyProfile","profile","description"],
                "icb_name3": ["icb_name3","icbName3","icb3","industry_name"],
                "icb_name2": ["icb_name2","icbName2","icb2"],
                "icb_name4": ["icb_name4","icbName4","icb4"],
                "financial_ratio_issue_share": ["financial_ratio_issue_share","financialRatioIssueShare"],
                "charter_capital": ["charter_capital","charterCapital","capitalAmount","capital"],
            }

        return self._map_records_to_df(records, cols, mapping)

    def shareholders(self) -> pd.DataFrame:
        src = self.source
        stock = self._safe_stock()
        records = []
        if stock:
            try:
                df = stock.company.shareholders()
                records = _df_to_records_safe(df)
            except Exception:
                records = []

        if not records:
            try:
                import json
                js = json.loads(get_company_info(self.symbol))
                comp = js.get("company") if isinstance(js, dict) else None
                if comp and comp.get(src) and comp[src].get("shareholders"):
                    std = comp[src]["shareholders"].get("standardized") or comp[src]["shareholders"].get("raw")
                    if std:
                        records = std
            except Exception:
                pass

        if src == "KBS":
            cols = ["name","update_date","shares_owned","ownership_percentage"]
            mapping = {
                "name": ["name","investorName","investor_name","share_holder","shareholder","ownerName"],
                "update_date": ["update_date","updateDate","as_of_date"],
                "shares_owned": ["shares_owned","sharesOwned","shares","quantity"],
                "ownership_percentage": ["ownership_percentage","ownershipPercent","ownedRate","share_own_ratio","ratio"],
            }
        else:
            cols = ["id","share_holder","quantity","share_own_percent","update_date"]
            mapping = {
                "id": ["id","shareholderId"],
                "share_holder": ["share_holder","shareHolder","shareholder","investorName","name"],
                "quantity": ["quantity","qty","shares","shares_owned"],
                "share_own_percent": ["share_own_percent","shareOwnPercent","share_own_ratio","ownedRate","ratio"],
                "update_date": ["update_date","updateDate"],
            }

        df_out = self._map_records_to_df(records, cols, mapping)
        return df_out

    def officers(self) -> pd.DataFrame:
        src = self.source
        stock = self._safe_stock()
        records = []
        if stock:
            try:
                df = stock.company.officers()
                records = _df_to_records_safe(df)
            except Exception:
                records = []

        if not records:
            try:
                import json
                js = json.loads(get_company_info(self.symbol))
                comp = js.get("company") if isinstance(js, dict) else None
                if comp and comp.get(src) and comp[src].get("officers"):
                    std = comp[src]["officers"].get("standardized") or comp[src]["officers"].get("raw")
                    if std:
                        records = std
            except Exception:
                pass

        if src == "KBS":
            cols = ["from_date","position","name","position_en","owner_code"]
            mapping = {
                "from_date": ["from_date","fromDate","startYear","year"],
                "position": ["position","position_vi","position_vn"],
                "name": ["name","fullName","officerName","officer_name"],
                "position_en": ["position_en","positionEn"],
                "owner_code": ["owner_code","ownerCode"],
            }
        else:
            cols = ["id","officer_name","officer_position","officer_own_percent","quantity","update_date","position"]
            mapping = {
                "id": ["id","officerId"],
                "officer_name": ["officer_name","officerName","name"],
                "officer_position": ["officer_position","position","role"],
                "officer_own_percent": ["officer_own_percent","officerOwnPercent","share_own_ratio"],
                "quantity": ["quantity","qty","shares"],
                "update_date": ["update_date","updateDate"],
                "position": ["position","position_field"],
            }

        return self._map_records_to_df(records, cols, mapping)

    def subsidiaries(self) -> pd.DataFrame:
        # Only KBS typically provides subsidiaries
        src = self.source
        if src != "KBS":
            return pd.DataFrame(columns=["update_date","name","charter_capital","ownership_percent","currency","type"])

        stock = self._safe_stock()
        records = []
        if stock and hasattr(stock.company, "subsidiaries"):
            try:
                df = stock.company.subsidiaries()
                records = _df_to_records_safe(df)
            except Exception:
                records = []

        if not records:
            try:
                import json
                js = json.loads(get_company_info(self.symbol))
                comp = js.get("company") if isinstance(js, dict) else None
                if comp and comp.get(src) and comp[src].get("subsidiaries"):
                    std = comp[src]["subsidiaries"].get("standardized") or comp[src]["subsidiaries"].get("raw")
                    if std:
                        records = std
            except Exception:
                pass

        cols = ["update_date","name","charter_capital","ownership_percent","currency","type"]
        mapping = {
            "update_date": ["update_date","updateDate"],
            "name": ["name","companyName","subsidiaryName"],
            "charter_capital": ["charter_capital","charterCapital","capitalAmount"],
            "ownership_percent": ["ownership_percent","ownershipPercent"],
            "currency": ["currency","cur"],
            "type": ["type","relationType"],
        }
        return self._map_records_to_df(records, cols, mapping)


@tool
def get_company_info(ticker: str) -> str:
    """
    Tra cứu thông tin đầy đủ của công ty theo mã chứng khoán.
    Trả về: tên, ngành, sàn, vốn điều lệ, số cổ phiếu, ngày niêm yết,
    website, mô tả, cổ đông lớn, ban lãnh đạo, công ty con, P/E, P/B, ROE, vốn hóa.

    Args:
        ticker: Mã chứng khoán (VD: VNM, HPG, ACB, FPT, VIC)
    """
    ticker = ticker.strip().upper()
    result: dict = {"ticker": ticker}

    # ── 1. Comprehensive company profile (TCBS → VNDirect → FireAnt → SSI) ──
    if _DATA_SOURCES_AVAILABLE:
        try:
            from data_sources import fetch_comprehensive_company_info
            comp = fetch_comprehensive_company_info(ticker)
            result.update(comp)
        except Exception as e:
            print(f"[get_company_info] comprehensive fetch error: {e}")

    # ── 2. Fallback: VNDirect meta if ten_cong_ty still missing ─────────────
    if not result.get("ten_cong_ty") or result["ten_cong_ty"] == ticker:
        try:
            vnd = _fetch_vndirect_company_meta(ticker)
            if vnd:
                result.setdefault("ten_cong_ty",   vnd.get("companyName", ticker))
                result.setdefault("san_niem_yet",   vnd.get("floor", ""))
                result.setdefault("ngay_niem_yet",  vnd.get("listedDate", ""))
                result.setdefault("isin",           vnd.get("isin", ""))
                result.setdefault("ma_so_thue",     vnd.get("taxCode", ""))
        except Exception:
            pass

    # ── 3. Comprehensive financial ratios (TCBS → VNDirect → FireAnt) ───────
    if _DATA_SOURCES_AVAILABLE:
        try:
            from data_sources import fetch_comprehensive_financial_ratios
            ratios = fetch_comprehensive_financial_ratios(ticker)
            if ratios:
                chi_so = {
                    "P/E":             ratios.get("pe"),
                    "P/B":             ratios.get("pb"),
                    "EV/EBITDA":       ratios.get("ev_ebitda"),
                    "ROE_%":           ratios.get("roe"),
                    "ROA_%":           ratios.get("roa"),
                    "EPS_dong":        ratios.get("eps"),
                    "BVPS_dong":       ratios.get("bvps"),
                    "Co_tuc_%":        ratios.get("dividend_yield"),
                    "Bien_LN_rong_%":  ratios.get("net_margin"),
                    "Bien_LN_gop_%":   ratios.get("gross_margin"),
                    "D_E":             ratios.get("de_ratio"),
                    "Current_Ratio":   ratios.get("current_ratio"),
                    "ky_bao_cao":      ratios.get("ky"),
                }
                result["chi_so_dinh_gia"] = {k: v for k, v in chi_so.items() if v is not None}
                if ratios.get("market_cap_ty"):
                    result["market_cap_ty_dong"] = ratios["market_cap_ty"]
        except Exception as e:
            print(f"[get_company_info] ratios error: {e}")

    # ── 4. Giá hiện tại + metadata công ty từ TCBS Realtime ─────────────────
    try:
        from realtime_loader import get_stock_info_realtime, fetch_realtime_ohlcv
        price_raw = get_stock_info_realtime(ticker)

        # ── 4a. Trích xuất metadata công ty từ TCBS stock-info ────────────
        if price_raw:
            result.setdefault("ten_cong_ty",
                price_raw.get("organName") or price_raw.get("organ_name") or "")
            result.setdefault("ten_tieng_anh",
                price_raw.get("enOrganName") or price_raw.get("en_organ_name") or "")
            _ind = (price_raw.get("industryVi") or price_raw.get("industry_vi") or
                    price_raw.get("industryEn") or price_raw.get("industry") or "")
            if _ind:
                result.setdefault("nganh", _ind)
            _exch = price_raw.get("exchange") or price_raw.get("floor") or ""
            if _exch:
                result.setdefault("san_niem_yet", _exch)
            for _k in ("website", "homeUrl", "webUrl"):
                _w = price_raw.get(_k)
                if _w:
                    result.setdefault("website", str(_w))
                    break
            for _k in ("charterCapital", "charter_capital", "capitalAmount"):
                _raw_cc = price_raw.get(_k)
                if _raw_cc is not None:
                    try:
                        fv = float(_raw_cc)
                        result.setdefault("von_dieu_le_ty",
                            round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1))
                    except Exception:
                        pass
                    break
            for _k in ("outstandingShare", "outstanding_share", "listedShare", "shareAmount"):
                _raw_sh = price_raw.get(_k)
                if _raw_sh is not None:
                    try:
                        fv = float(_raw_sh)
                        result.setdefault("co_phieu_luu_hanh_trieu",
                            round(fv / 1e6, 2) if fv > 1e6 else round(fv, 2))
                    except Exception:
                        pass
                    break
            for _k in ("marketCap", "market_cap", "marketCapitalization"):
                _raw_mc = price_raw.get(_k)
                if _raw_mc is not None:
                    try:
                        fv = float(_raw_mc)
                        result.setdefault("market_cap_ty_dong",
                            round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1))
                    except Exception:
                        pass
                    break

        close_val = None
        if price_raw:
            close_val = (
                price_raw.get("close") or price_raw.get("price") or
                price_raw.get("lastPrice") or price_raw.get("last")
            )

        if close_val:
            fclose = float(close_val)
            change_raw = price_raw.get("change") or price_raw.get("priceChange") or 0
            pct_raw    = price_raw.get("pctChange") or price_raw.get("changePercent") or price_raw.get("percent") or 0
            try:
                fpct = float(pct_raw)
                fpct_display = round(fpct * 100, 2) if abs(fpct) < 1 else round(fpct, 2)
            except Exception:
                fpct_display = 0
            result["gia_hien_tai"] = {
                "gia":        _norm_p(fclose),
                "don_vi":     "VND",
                "thay_doi":   _norm_p(float(change_raw)) if change_raw else None,
                "thay_doi_%": fpct_display,
                "ngay":       datetime.now().strftime("%d/%m/%Y"),
                "trang_thai": "Realtime",
                "nguon":      "TCBS Realtime",
            }
        else:
            # Fallback: lấy giá phiên gần nhất từ OHLCV (kể cả thị trường đóng cửa)
            df_fb = None
            for _src in ("KBS", "VCI"):
                df_fb = _fetch_ohlcv_vnstock(ticker, _src, "1D", None, None, "10")
                if df_fb is not None and not df_fb.empty:
                    break
            if df_fb is None or df_fb.empty:
                df_fb2, err2 = fetch_realtime_ohlcv(symbol=ticker, interval="1d", lookback_days=15, tail=10)
                if not err2 and df_fb2 is not None and not df_fb2.empty:
                    df_fb = df_fb2

            if df_fb is not None and not df_fb.empty:
                latest = df_fb.iloc[-1]
                prev   = df_fb.iloc[-2] if len(df_fb) > 1 else latest
                fclose  = float(latest["Close"])
                fprev   = float(prev["Close"])
                fchg    = fclose - fprev
                phien_ngay = str(latest["Datetime"])[:10]
                hom_nay    = datetime.now().strftime("%Y-%m-%d")
                trang_thai = "Realtime" if phien_ngay == hom_nay else f"Giá đóng cửa phiên {phien_ngay} (thị trường đang đóng)"
                result["gia_hien_tai"] = {
                    "gia":        _norm_p(fclose),
                    "don_vi":     "VND",
                    "thay_doi":   _norm_p(fchg),
                    "thay_doi_%": round(fchg / fprev * 100, 2) if fprev else 0,
                    "ngay_phien": phien_ngay,
                    "trang_thai": trang_thai,
                    "nguon":      "OHLCV history",
                }
    except Exception as e:
        print(f"[get_company_info] price error: {e}")

    # ── 5. Cổ đông lớn ───────────────────────────────────────────────────────
    if _DATA_SOURCES_AVAILABLE:
        try:
            owners = fetch_vndirect_ownership(ticker)
            if owners:
                result["co_dong_lon"] = [
                    {
                        "ten":     o.get("investorName") or o.get("ownerName", ""),
                        "ty_le_%": round(float(o.get("ownedRate", 0)) * 100, 2),
                        "loai":    o.get("investorType", ""),
                    }
                    for o in owners[:5]
                ]
        except Exception:
            pass

    # ── 6. vnstock company.overview() + company.profile() fallback ───────────
    try:
        from vnstock import Vnstock
        vn = Vnstock()
        result.setdefault("company", {})

        # mappings for normalization (candidate keys likely returned by vnstock)
        overview_kbs_map = {
            "company_name": ["company_name", "companyName", "short_name", "shortName", "name"],
            "business_model": ["business_model", "businessModel", "business_model_vi", "mo_hinh_kinh_doanh"],
            "symbol": ["symbol", "ticker", "code"],
            "founded_date": ["founded_date", "foundedDate", "established_date", "establishmentDate"],
            "charter_capital": ["charter_capital", "charterCapital", "capitalAmount", "capital"],
            "number_of_employees": ["number_of_employees", "employees", "employee_count"],
            "listing_date": ["listing_date", "listed_date", "listedDate"],
            "par_value": ["par_value", "parValue", "face_value"],
            "exchange": ["exchange", "floor", "listed_floor"],
            "listing_price": ["listing_price", "listed_price", "listingPrice"],
            "listed_volume": ["listed_volume", "listedShare", "outstanding_share", "outstandingShare"],
            "ceo_name": ["ceo_name", "ceo", "chairman"],
            "ceo_position": ["ceo_position", "position", "position_en"],
            "inspector_name": ["inspector_name", "inspectorName", "controller"],
            "inspector_position": ["inspector_position", "inspectorPosition"],
            "establishment_license": ["establishment_license", "establishmentLicense", "license"],
            "business_code": ["business_code", "businessCode", "industry_code", "ma_nganh"],
            "tax_id": ["tax_id", "taxId", "tax_code", "ma_so_thue"],
            "auditor": ["auditor", "auditor_name", "auditorName"],
            "company_type": ["company_type", "companyType", "type"],
            "address": ["address", "company_address", "head_office"],
            "phone": ["phone", "telephone", "phone_number"],
            "fax": ["fax", "fax_number"],
            "email": ["email", "contact_email"],
            "website": ["website", "homeUrl", "webUrl", "web"],
            "branches": ["branches", "branch", "subsidiaries"],
            "history": ["history", "history_dev", "company_history"],
            "free_float_percentage": ["free_float_percentage", "freeFloatPercent", "free_float_pct"],
            "free_float": ["free_float", "freeFloat", "free_float_shares"],
            "outstanding_shares": ["outstanding_shares", "outstandingShare", "listed_share"],
            "as_of_date": ["as_of_date", "asOfDate", "update_date"],
        }

        overview_vci_map = {
            "symbol": ["symbol", "ticker", "code"],
            "id": ["id", "companyId", "company_id"],
            "issue_share": ["issue_share", "issueShare", "issue_shares"],
            "history": ["history", "history_dev", "company_history"],
            "company_profile": ["company_profile", "companyProfile", "profile"],
            "icb_name3": ["icb_name3", "icbName3", "icb3"],
            "icb_name2": ["icb_name2", "icbName2", "icb2"],
            "icb_name4": ["icb_name4", "icbName4", "icb4"],
            "financial_ratio_issue_share": ["financial_ratio_issue_share", "financialRatioIssueShare"],
            "charter_capital": ["charter_capital", "charterCapital", "capitalAmount", "capital"],
            "company_name": ["company_name", "companyName", "short_name", "name"],
        }

        shareholders_kbs_map = {
            "name": ["name", "investorName", "investor_name"],
            "update_date": ["update_date", "updateDate"],
            "shares_owned": ["shares_owned", "sharesOwned", "shares", "quantity"],
            "ownership_percentage": ["ownership_percentage", "ownershipPercent", "ownedRate", "owned_rate"],
        }

        shareholders_vci_map = {
            "id": ["id", "shareholderId"],
            "share_holder": ["share_holder", "shareHolder", "shareholder", "investorName"],
            "quantity": ["quantity", "qty", "shares"],
            "share_own_percent": ["share_own_percent", "shareOwnPercent", "share_own_ratio"],
            "update_date": ["update_date", "updateDate"],
        }

        officers_kbs_map = {
            "from_date": ["from_date", "fromDate", "startYear"],
            "position": ["position", "position_vi", "position_vn"],
            "name": ["name", "fullName", "officerName"],
            "position_en": ["position_en", "positionEn"],
            "owner_code": ["owner_code", "ownerCode"],
        }

        officers_vci_map = {
            "id": ["id", "officerId"],
            "officer_name": ["officer_name", "officerName", "officer"],
            "officer_position": ["officer_position", "position", "role"],
            "officer_own_percent": ["officer_own_percent", "officerOwnPercent"],
            "quantity": ["quantity", "qty", "shares"],
            "update_date": ["update_date", "updateDate"],
            "position": ["position", "position_field"],
        }

        subsidiaries_kbs_map = {
            "update_date": ["update_date", "updateDate"],
            "name": ["name", "companyName", "subsidiaryName"],
            "charter_capital": ["charter_capital", "charterCapital", "capitalAmount"],
            "ownership_percent": ["ownership_percent", "ownershipPercent"],
            "currency": ["currency", "cur"],
            "type": ["type", "relationType"],
        }

        for _src in ("KBS", "VCI"):
            try:
                try:
                    _stock = vn.stock(symbol=ticker, source=_src)
                except Exception:
                    continue

                src_block = {
                    "overview": {"raw": [], "standardized": []},
                    "profile": {"raw": []},
                    "shareholders": {"raw": [], "standardized": []},
                    "officers": {"raw": [], "standardized": []},
                    "subsidiaries": {"raw": [], "standardized": []},
                }

                # Overview
                try:
                    df_ov = _stock.company.overview()
                    raw_ov = _df_to_records_safe(df_ov)
                    src_block["overview"]["raw"] = raw_ov
                    if raw_ov:
                        mapping = overview_kbs_map if _src == "KBS" else overview_vci_map
                        std_ov = _normalize_records(raw_ov, mapping)
                        # post-process numeric / percent fields
                        for r in std_ov:
                            if r.get("charter_capital") is not None:
                                r["charter_capital"] = _coerce_number(r["charter_capital"])
                            if r.get("free_float_percentage") is not None:
                                r["free_float_percentage"] = _coerce_percent(r["free_float_percentage"])
                            if r.get("free_float") is not None:
                                r["free_float"] = _coerce_number(r["free_float"])
                            if r.get("outstanding_shares") is not None:
                                r["outstanding_shares"] = _coerce_number(r["outstanding_shares"])
                            if r.get("issue_share") is not None:
                                r["issue_share"] = _coerce_number(r["issue_share"])
                        src_block["overview"]["standardized"] = std_ov

                        # promote some top-level fields (if not present)
                        first = std_ov[0]
                        if not result.get("ten_cong_ty"):
                            for cand in ("company_name", "name", "short_name", "symbol"):
                                if first.get(cand):
                                    result.setdefault("ten_cong_ty", str(first.get(cand)))
                                    break
                        if not result.get("nganh"):
                            for cand in ("icb_name3", "icb_name2", "industry_name"):
                                if first.get(cand):
                                    result.setdefault("nganh", first.get(cand))
                                    break
                        if not result.get("website") and first.get("website"):
                            result.setdefault("website", first.get("website"))
                        if not result.get("von_dieu_le_ty") and first.get("charter_capital") is not None:
                            try:
                                fv = float(first.get("charter_capital"))
                                result.setdefault("von_dieu_le_ty", round(fv / 1e9, 1) if fv > 1e9 else round(fv, 1))
                            except Exception:
                                pass
                        if not result.get("co_phieu_luu_hanh_trieu") and first.get("outstanding_shares") is not None:
                            try:
                                fv = float(first.get("outstanding_shares"))
                                result.setdefault("co_phieu_luu_hanh_trieu", round(fv / 1e6, 2) if fv > 1e6 else round(fv, 2))
                            except Exception:
                                pass
                except Exception:
                    pass

                # Profile / description
                try:
                    df_pr = _stock.company.profile()
                    raw_pr = _df_to_records_safe(df_pr)
                    src_block["profile"]["raw"] = raw_pr
                    if raw_pr and not result.get("gioi_thieu"):
                        r = raw_pr[0]
                        for cand in ("company_profile", "history_dev", "business_strategies", "description", "companyProfile"):
                            if r.get(cand):
                                result["gioi_thieu"] = str(r.get(cand))[:600]
                                break
                except Exception:
                    pass

                # Shareholders
                try:
                    df_sh = _stock.company.shareholders()
                    raw_sh = _df_to_records_safe(df_sh)
                    src_block["shareholders"]["raw"] = raw_sh
                    if raw_sh:
                        map_sh = shareholders_kbs_map if _src == "KBS" else shareholders_vci_map
                        std_sh = _normalize_records(raw_sh, map_sh)
                        for r in std_sh:
                            if r.get("shares_owned") is not None:
                                r["shares_owned"] = _coerce_number(r["shares_owned"])
                            if r.get("ownership_percentage") is not None:
                                r["ownership_percentage"] = _coerce_percent(r["ownership_percentage"])
                            if r.get("share_own_percent") is not None:
                                r["share_own_percent"] = _coerce_percent(r["share_own_percent"])
                            if r.get("quantity") is not None:
                                r["quantity"] = _coerce_number(r["quantity"])
                        src_block["shareholders"]["standardized"] = std_sh
                        if not result.get("co_dong_lon"):
                            top5 = []
                            for r in std_sh[:5]:
                                if _src == "KBS":
                                    top5.append({
                                        "ten": r.get("name") or r.get("share_holder") or "",
                                        "ty_le_%": r.get("ownership_percentage") or 0,
                                        "so_co_phieu": r.get("shares_owned") or r.get("quantity") or None,
                                    })
                                else:
                                    top5.append({
                                        "ten": r.get("share_holder") or r.get("name") or "",
                                        "ty_le_%": r.get("share_own_percent") or 0,
                                        "so_co_phieu": r.get("quantity") or r.get("shares_owned") or None,
                                    })
                            result["co_dong_lon"] = top5
                except Exception:
                    pass

                # Officers / leadership
                try:
                    df_off = _stock.company.officers()
                    raw_off = _df_to_records_safe(df_off)
                    src_block["officers"]["raw"] = raw_off
                    if raw_off:
                        map_off = officers_kbs_map if _src == "KBS" else officers_vci_map
                        std_off = _normalize_records(raw_off, map_off)
                        for r in std_off:
                            if r.get("from_date") is not None:
                                try:
                                    r["from_date"] = int(r["from_date"])
                                except Exception:
                                    pass
                            if r.get("officer_own_percent") is not None:
                                r["officer_own_percent"] = _coerce_percent(r["officer_own_percent"])
                        src_block["officers"]["standardized"] = std_off
                        if not result.get("ban_lanh_dao"):
                            result["ban_lanh_dao"] = std_off[:10]
                except Exception:
                    pass

                # Subsidiaries (usually provided by KBS)
                try:
                    if hasattr(_stock.company, "subsidiaries"):
                        df_sub = _stock.company.subsidiaries()
                        raw_sub = _df_to_records_safe(df_sub)
                        src_block["subsidiaries"]["raw"] = raw_sub
                        if raw_sub:
                            std_sub = _normalize_records(raw_sub, subsidiaries_kbs_map)
                            for r in std_sub:
                                if r.get("charter_capital") is not None:
                                    r["charter_capital"] = _coerce_number(r["charter_capital"])
                                if r.get("ownership_percent") is not None:
                                    r["ownership_percent"] = _coerce_percent(r["ownership_percent"])
                            src_block["subsidiaries"]["standardized"] = std_sub
                            if not result.get("cong_ty_con"):
                                result["cong_ty_con"] = std_sub[:10]
                except Exception:
                    pass

                result["company"].setdefault(_src, src_block)
            except Exception:
                continue
    except Exception as e:
        print(f"[get_company_info] vnstock overview fallback: {e}")

    # ── 7. Apply industry profile if description still missing ───────────────
    if not result.get("gioi_thieu"):
        try:
            _apply_industry_profile(result)
        except Exception:
            pass

    # ── 8. Build structured company_data block (dễ đọc cho LLM) ────────────
    # Lấy dữ liệu chuẩn từ company block (ưu tiên KBS, fallback VCI)
    try:
        company_block = result.get("company", {})

        def _first_std(src, section):
            blk = company_block.get(src, {}).get(section, {})
            std = blk.get("standardized") or blk.get("raw") or []
            return std

        def _safe_val(v):
            """Trả về giá trị nếu không rỗng, else None."""
            if v is None:
                return None
            if isinstance(v, float) and pd.isna(v):
                return None
            if isinstance(v, str) and v.strip() == "":
                return None
            return v

        company_data = {}

        # --- KBS Overview (30 trường) ---
        kbs_ov = _first_std("KBS", "overview")
        if kbs_ov:
            r0 = kbs_ov[0]
            kbs_overview = {}
            _KBS_OV_FIELDS = [
                "company_name", "business_model", "symbol", "founded_date", "charter_capital",
                "number_of_employees", "listing_date", "par_value", "exchange",
                "listing_price", "listed_volume", "ceo_name", "ceo_position",
                "inspector_name", "inspector_position", "establishment_license",
                "business_code", "tax_id", "auditor", "company_type",
                "address", "phone", "fax", "email", "website", "branches",
                "history", "free_float_percentage", "free_float",
                "outstanding_shares", "as_of_date",
            ]
            for f in _KBS_OV_FIELDS:
                v = _safe_val(r0.get(f))
                if v is not None:
                    kbs_overview[f] = v
            if kbs_overview:
                company_data["kbs_overview"] = kbs_overview

        # --- VCI Overview (10 trường) ---
        vci_ov = _first_std("VCI", "overview")
        if vci_ov:
            r0 = vci_ov[0]
            vci_overview = {}
            _VCI_OV_FIELDS = [
                "company_name", "symbol", "id", "issue_share", "history", "company_profile",
                "icb_name3", "icb_name2", "icb_name4",
                "financial_ratio_issue_share", "charter_capital",
            ]
            for f in _VCI_OV_FIELDS:
                v = _safe_val(r0.get(f))
                if v is not None:
                    vci_overview[f] = v
            if vci_overview:
                company_data["vci_overview"] = vci_overview

        if company_data:
            result["company_data"] = company_data
    except Exception as e:
        print(f"[get_company_info] company_data build error: {e}")

    # Remove empty fields (None, empty string, empty list/dict, NaN) recursively
    def _is_empty_value(v):
        try:
            if v is None:
                return True
            # pandas / numpy NaN
            if isinstance(v, float) and pd.isna(v):
                return True
            if isinstance(v, (list, tuple, set)) and len(v) == 0:
                return True
            if isinstance(v, dict) and len(v) == 0:
                return True
            if isinstance(v, str) and v.strip() == "":
                return True
        except Exception:
            return False
        return False

    def _prune_empty(obj):
        # Recursively prune empty values from dicts and lists
        try:
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    pv = _prune_empty(v)
                    if _is_empty_value(pv):
                        continue
                    out[k] = pv
                return out

            if isinstance(obj, list):
                out_list = []
                for item in obj:
                    pi = _prune_empty(item)
                    if _is_empty_value(pi):
                        continue
                    out_list.append(pi)
                return out_list

            # pandas DataFrame / Series -> convert to native types then prune
            try:
                import pandas as _pd
                if _pd and (_pd.api.types.is_scalar(obj) is False) and hasattr(obj, "empty"):
                    if getattr(obj, "empty", False):
                        return None
                    # DataFrame -> list of records
                    if hasattr(obj, "to_dict"):
                        try:
                            recs = obj.to_dict(orient="records")
                            return _prune_empty(recs)
                        except Exception:
                            pass
            except Exception:
                pass

            # numpy scalar -> native python
            try:
                import numpy as _np
                if isinstance(obj, (_np.floating, _np.integer)):
                    if _np.isnan(obj) if isinstance(obj, _np.floating) else False:
                        return None
                    return obj.item()
            except Exception:
                pass

            # strings: trim
            if isinstance(obj, str):
                s = obj.strip()
                return s if s != "" else None

            return obj
        except Exception:
            return obj

    pruned = _prune_empty(result)
    return json.dumps(pruned, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 - Price History (vnstock quote.history + realtime_loader fallback)
# ─────────────────────────────────────────────────────────────────────────────

# Mapping interval người dùng → vnstock format
_INTERVAL_MAP = {
    "1m": "1m", "1min": "1m",
    "5m": "5m", "5min": "5m",
    "15m": "15m", "15min": "15m",
    "30m": "30m", "30min": "30m",
    "1h": "1H", "1H": "1H",
    "1d": "1D", "1D": "1D", "D": "1D",
    "1w": "1W", "1W": "1W", "W": "1W",
    "1mo": "1M", "1M": "1M", "M": "1M",
}

# Mapping period string → length cho vnstock
_PERIOD_TO_LENGTH = {
    "1w": "7",   "1 tuần": "7",   "tuần": "7",
    "1m": "1M",  "1 tháng": "1M", "tháng": "1M",
    "3m": "3M",  "3 tháng": "3M",
    "6m": "6M",  "6 tháng": "6M",
    "1y": "1Y",  "1 năm": "1Y",   "năm": "1Y",
    "2y": "2Y",  "2 năm": "2Y",
    "ytd": None, "đầu năm": None,  # ytd → dùng start_date
}


def _fetch_ohlcv_vnstock(ticker: str, source: str, interval_vn: str,
                          start: Optional[str], end: Optional[str],
                          length: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Gọi vnstock quote.history() và chuẩn hóa kết quả thành DataFrame
    với cột: Datetime, Open, High, Low, Close, Volume (giá đơn vị đồng VND).
    Trả về None nếu lỗi.
    """
    try:
        from vnstock import Vnstock
        vn = Vnstock()
        stock = vn.stock(symbol=ticker, source=source)
        quote = stock.quote

        # Xây dựng kwargs cho quote.history()
        kwargs: dict = {"interval": interval_vn}
        if start:
            kwargs["start"] = start
            kwargs["end"] = end or datetime.now().strftime("%Y-%m-%d")
        elif length:
            kwargs["length"] = length
        else:
            kwargs["length"] = "3M"  # default

        # KBS hỗ trợ get_all
        if source == "KBS":
            try:
                df = quote.history(**kwargs, get_all=False)
            except TypeError:
                df = quote.history(**kwargs)
        else:
            df = quote.history(**kwargs)

        if df is None or (hasattr(df, "empty") and df.empty):
            return None

        # Chuẩn hóa cột
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("time", "date", "datetime", "tradingdate", "trading_date"):
                col_map[c] = "Datetime"
            elif cl == "open":
                col_map[c] = "Open"
            elif cl == "high":
                col_map[c] = "High"
            elif cl == "low":
                col_map[c] = "Low"
            elif cl in ("close", "price"):
                col_map[c] = "Close"
            elif cl in ("volume", "vol"):
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        needed = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if "Datetime" not in df.columns or "Close" not in df.columns:
            return None

        df = df[needed].copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Chuẩn hóa giá về đơn vị đồng VND
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float).apply(
                    lambda x: round(x * 1000) if x < 10_000 else round(x)
                )
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

        df = df.sort_values("Datetime").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"[_fetch_ohlcv_vnstock] {ticker}/{source}: {e}")
        return None


def _fetch_ohlcv_realtime(ticker: str, interval_vn: str,
                           start: Optional[str], end: Optional[str],
                           length: Optional[str]) -> Optional[pd.DataFrame]:
    """Fallback dùng realtime_loader.fetch_realtime_ohlcv."""
    try:
        from realtime_loader import fetch_realtime_ohlcv

        # Tính lookback_days
        if start:
            try:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                lookback = (datetime.now() - start_dt).days + 30
            except Exception:
                lookback = 400
        else:
            # Đổi length → ngày xấp xỉ
            try:
                lg = str(length or "3M").upper()
                if lg.endswith("Y"):
                    lookback = int(lg[:-1]) * 365 + 30
                elif lg.endswith("M"):
                    lookback = int(lg[:-1]) * 31 + 30
                elif lg.endswith("B"):
                    lookback = int(lg[:-1]) * 2 + 30
                else:
                    lookback = int(lg) + 30
            except Exception:
                lookback = 120

        # Map interval vnstock → realtime_loader format
        rl_interval = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1H": "1h", "1D": "1d", "1W": "1w", "1M": "1M",
        }.get(interval_vn, "1d")

        df, err = fetch_realtime_ohlcv(
            symbol=ticker,
            interval=rl_interval,
            start_date=start,
            end_date=end,
            lookback_days=lookback,
            tail=lookback,
        )
        if err or df is None or df.empty:
            return None

        # Chuẩn hóa cột
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("time", "date", "datetime"):
                col_map[c] = "Datetime"
            elif cl == "open":  col_map[c] = "Open"
            elif cl == "high":  col_map[c] = "High"
            elif cl == "low":   col_map[c] = "Low"
            elif cl == "close": col_map[c] = "Close"
            elif cl in ("volume", "vol"): col_map[c] = "Volume"
        df = df.rename(columns=col_map)
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Chuẩn hóa giá
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float).apply(
                    lambda x: round(x * 1000) if x < 10_000 else round(x)
                )

        df = df.sort_values("Datetime").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[_fetch_ohlcv_realtime] {ticker}: {e}")
        return None


@tool
def get_price_history(
    ticker: str,
    period: str = "3m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1D",
) -> str:
    """
    Lấy dữ liệu giá lịch sử OHLCV của cổ phiếu.
    Hỗ trợ lọc theo ngày cụ thể hoặc khung thời gian linh hoạt.
    Giá trả về đơn vị đồng (VND).

    Args:
        ticker:     Mã chứng khoán (VD: VNM, HPG, ACB, FPT, VIC)
        period:     Khoảng thời gian: "1m","3m","6m","1y","2y","ytd"
                    hoặc định dạng vnstock: "1M","3M","1Y","150","100b"
        start_date: Ngày bắt đầu YYYY-MM-DD (ưu tiên hơn period)
        end_date:   Ngày kết thúc YYYY-MM-DD (mặc định hôm nay)
        interval:   Khung nến: "1m","5m","15m","30m","1H","1D","1W","1M"
    """
    ticker = ticker.strip().upper()
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    # Chuẩn hóa interval
    interval_vn = _INTERVAL_MAP.get(interval, _INTERVAL_MAP.get(interval.upper(), "1D"))

    # Xác định start_date / length
    length: Optional[str] = None
    s: Optional[str] = None
    e: Optional[str] = None

    if start_date:
        s = start_date
        e = end_date
    else:
        p = (period or "3m").lower().strip()
        mapped = _PERIOD_TO_LENGTH.get(p)

        if mapped is None and p in ("ytd", "đầu năm"):
            s = datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d")
            e = end_date
        elif mapped:
            length = mapped
        else:
            # Thử dùng trực tiếp làm length vnstock ("150", "100b", "3M")
            length = period

    # ── Thử vnstock KBS → VCI → realtime_loader ──────────────────────────────
    df: Optional[pd.DataFrame] = None
    source_used = ""

    for src in ("KBS", "VCI"):
        df = _fetch_ohlcv_vnstock(ticker, src, interval_vn, s, e, length)
        if df is not None and not df.empty:
            source_used = src
            break

    if df is None or df.empty:
        df = _fetch_ohlcv_realtime(ticker, interval_vn, s, e, length)
        source_used = "realtime_loader"

    if df is None or df.empty:
        return json.dumps({"error": f"Không lấy được dữ liệu {ticker}. Vui lòng thử lại."}, ensure_ascii=False)

    # ── Lọc theo start/end nếu có ─────────────────────────────────────────────
    thi_truong_dong_cua = False       # flag thị trường đóng cửa
    ngay_yeu_cau        = e or ""     # ngày user hỏi (dùng để so sánh)

    if s and e:
        mask = (df["Datetime"].dt.strftime("%Y-%m-%d") >= s) & \
               (df["Datetime"].dt.strftime("%Y-%m-%d") <= e)
        df_filtered = df[mask].reset_index(drop=True)

        if df_filtered.empty:
            # ── Thị trường đóng cửa (cuối tuần / nghỉ lễ): lấy phiên gần nhất ──
            # Lấy tất cả phiên ≤ ngày yêu cầu (hoặc toàn bộ nếu không có)
            mask_past = df["Datetime"].dt.strftime("%Y-%m-%d") <= e
            df_past   = df[mask_past].reset_index(drop=True)
            df_use    = df_past if not df_past.empty else df
            # Chỉ giữ phiên giao dịch gần nhất
            df        = df_use.iloc[[-1]].reset_index(drop=True)
            thi_truong_dong_cua = True
        else:
            df = df_filtered

    if df.empty:
        return json.dumps({"error": f"Không có dữ liệu {ticker} trong khoảng thời gian yêu cầu."}, ensure_ascii=False)

    # ── Thống kê tổng hợp ─────────────────────────────────────────────────────
    closes      = df["Close"].astype(float)
    first_close = float(closes.iloc[0])
    last_close  = float(closes.iloc[-1])
    chg         = last_close - first_close
    chg_pct     = round(chg / first_close * 100, 2) if first_close else 0

    high_idx  = df["High"].astype(float).idxmax() if "High" in df.columns else None
    low_idx   = df["Low"].astype(float).idxmin()  if "Low"  in df.columns else None
    high_date = str(df.loc[high_idx, "Datetime"])[:10] if high_idx is not None else ""
    low_date  = str(df.loc[low_idx,  "Datetime"])[:10] if low_idx  is not None else ""

    avg_vol = int(df["Volume"].mean()) if ("Volume" in df.columns and df["Volume"].sum() > 0) else 0

    tu_ngay  = str(df["Datetime"].iloc[0])[:10]
    den_ngay = str(df["Datetime"].iloc[-1])[:10]

    result = {
        "ticker":           ticker,
        "nguon":            source_used,
        "interval":         interval_vn,
        "tu_ngay":          tu_ngay,
        "den_ngay":         den_ngay,
        "so_phien":         len(df),
        "don_vi_gia":       "đồng (VND)",
        "gia_mo_dau_ky":    int(round(float(df["Open"].iloc[0]))) if "Open" in df.columns else None,
        "gia_dong_dau_ky":  int(round(first_close)),
        "gia_dong_cuoi_ky": int(round(last_close)),
        "thay_doi_vnd":     int(round(chg)),
        "thay_doi_pct":     chg_pct,
        "gia_cao_nhat":     int(round(float(df["High"].max()))) if "High" in df.columns else None,
        "ngay_cao_nhat":    high_date,
        "gia_thap_nhat":    int(round(float(df["Low"].min()))) if "Low"  in df.columns else None,
        "ngay_thap_nhat":   low_date,
        "klgd_tb_phien":    avg_vol,
        "data":             _df_to_compressed(df, max_rows=200),
    }

    # ── Gắn thông tin thị trường đóng cửa ────────────────────────────────────
    if thi_truong_dong_cua:
        result["thi_truong_dong_cua"] = True
        result["ngay_phien_gan_nhat"] = den_ngay
        result["ghi_chu"] = (
            f"⚠️ Thị trường đóng cửa ngày {ngay_yeu_cau}. "
            f"Giá hiển thị là giá đóng cửa phiên giao dịch gần nhất: {den_ngay}."
        )
    else:
        # Kiểm tra thêm: nếu phiên cuối khác hôm nay thì cũng ghi chú
        hom_nay = datetime.now().strftime("%Y-%m-%d")
        if den_ngay < hom_nay and not s:
            result["ghi_chu"] = (
                f"Dữ liệu tới phiên {den_ngay}. "
                "Thị trường có thể đã đóng cửa hoặc chưa cập nhật phiên hôm nay."
            )

    result = {k: v for k, v in result.items() if v is not None}
    return json.dumps(result, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 - Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculate_technical_indicators(
    ticker: str,
    indicators: str = "SMA,RSI",
    sma_windows: str = "20,50,200",
    rsi_window: int = 14,
    period: str = "1y",
) -> str:
    """
    Tính các chỉ báo kỹ thuật: SMA, EMA, RSI, MACD, Bollinger Bands, Stoch, ATR.
    Giá tính bằng đồng (VND).

    Args:
        ticker: Mã chứng khoán
        indicators: Chỉ báo cách nhau bằng dấu phẩy (SMA, EMA, RSI, MACD, BB, STOCH, ATR)
        sma_windows: Chu kỳ SMA/EMA (VD: "20,50,200")
        rsi_window: Chu kỳ RSI (thường 14)
        period: Khoảng lấy dữ liệu ("6m","1y")
    """
    ticker = ticker.strip().upper()
    ind_list = [i.strip().upper() for i in indicators.split(",")]
    try:
        from realtime_loader import fetch_realtime_ohlcv
        df, err = fetch_realtime_ohlcv(symbol=ticker, interval="1d", lookback_days=400, tail=400)
        if err or df.empty:
            return json.dumps({"error": f"Không lấy được dữ liệu {ticker}: {err}"})

        # Normalize OHLC data to VND units first
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float).apply(lambda x: x*1000 if x < 10000 else x)

        closes = df["Close"].astype(float)
        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)
        latest_close = float(closes.iloc[-1])
        latest_date  = str(df["Datetime"].iloc[-1])[:10]

        result = {
            "ticker":      ticker,
            "ngay":        latest_date,
            "gia_dong_cua": _norm_p(latest_close),
            "don_vi_gia":  "đồng (VND)",
            "indicators":  {},
            "signals":     [],
        }

        if "SMA" in ind_list:
            windows_list = [int(w.strip()) for w in sma_windows.split(",") if w.strip().isdigit()] or [20, 50, 200]
            sma_data = {}
            for w in windows_list:
                sma = closes.rolling(window=w, min_periods=w).mean()
                val = sma.iloc[-1]
                if pd.isna(val):
                    sma_data[f"SMA{w}"] = {"value_dong": None, "note": f"Cần {w} nến"}
                    continue
                v_norm = _norm_p(float(val))
                diff_pct = round((latest_close - float(val)) / float(val) * 100, 2)
                sma_data[f"SMA{w}"] = {
                    "value_dong": v_norm,
                    "chenh_lech_pct": diff_pct,
                    "signal": "GIÁ TRÊN SMA" if diff_pct > 0 else "GIÁ DƯỚI SMA",
                }
                cur_p_str = _norm_p(latest_close)
                result["signals"].append(
                    f"✅ Giá ({cur_p_str}đ) trên SMA{w} ({v_norm}đ) +{diff_pct}%"
                    if diff_pct > 0
                    else f"❌ Giá ({cur_p_str}đ) dưới SMA{w} ({v_norm}đ) {diff_pct}%"
                )
            # Golden/Death Cross detection
            if 20 in windows_list and 50 in windows_list:
                sma20 = closes.rolling(20).mean()
                sma50 = closes.rolling(50).mean()
                if len(sma20) >= 2 and not pd.isna(sma20.iloc[-1]) and not pd.isna(sma50.iloc[-1]):
                    if sma20.iloc[-2] < sma50.iloc[-2] and sma20.iloc[-1] >= sma50.iloc[-1]:
                        result["signals"].append("🟡 GOLDEN CROSS: SMA20 vừa cắt lên SMA50")
                    elif sma20.iloc[-2] > sma50.iloc[-2] and sma20.iloc[-1] <= sma50.iloc[-1]:
                        result["signals"].append("🔴 DEATH CROSS: SMA20 vừa cắt xuống SMA50")
            result["indicators"]["SMA"] = sma_data

        if "EMA" in ind_list:
            windows_list = [int(w.strip()) for w in sma_windows.split(",") if w.strip().isdigit()] or [12, 26]
            ema_data = {}
            for w in windows_list:
                ema = closes.ewm(span=w, adjust=False).mean()
                val = ema.iloc[-1]
                v_norm = _norm_p(float(val))
                diff_pct = round((latest_close - float(val)) / float(val) * 100, 2)
                ema_data[f"EMA{w}"] = {
                    "value_dong": v_norm,
                    "chenh_lech_pct": diff_pct,
                    "signal": "GIÁ TRÊN EMA" if diff_pct > 0 else "GIÁ DƯỚI EMA",
                }
            result["indicators"]["EMA"] = ema_data

        if "RSI" in ind_list:
            delta    = closes.diff()
            gain     = delta.clip(lower=0)
            loss     = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=rsi_window - 1, min_periods=rsi_window).mean()
            avg_loss = loss.ewm(com=rsi_window - 1, min_periods=rsi_window).mean()
            rsi      = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-9))
            rsi_val  = round(float(rsi.iloc[-1]), 2)

            if rsi_val >= 70:
                rsi_sig = "⚠️ QUÁ MUA (≥70) - cân nhắc chốt lời"
            elif rsi_val <= 30:
                rsi_sig = "✅ QUÁ BÁN (≤30) - cơ hội mua vào"
            elif rsi_val >= 55:
                rsi_sig = "📈 Vùng tăng (55–70)"
            elif rsi_val <= 45:
                rsi_sig = "📉 Vùng giảm (30–45)"
            else:
                rsi_sig = "➡️ Trung tính (45–55)"

            result["indicators"]["RSI"] = {
                "chu_ky": rsi_window,
                "gia_tri": rsi_val,
                "signal": rsi_sig,
                "lich_su_5_phien": [round(float(v), 1) for v in rsi.tail(5).tolist()],
            }
            result["signals"].append(f"RSI({rsi_window}) = {rsi_val} → {rsi_sig}")

        if "MACD" in ind_list:
            ema12       = closes.ewm(span=12, adjust=False).mean()
            ema26       = closes.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram   = macd_line - signal_line
            h_val = round(float(histogram.iloc[-1]), 4)
            if len(histogram) >= 2:
                prev_h = float(histogram.iloc[-2])
                if h_val > 0 and prev_h < 0:
                    macd_sig = "🟢 GOLDEN CROSS - Tín hiệu MUA mạnh"
                elif h_val < 0 and prev_h > 0:
                    macd_sig = "🔴 DEATH CROSS - Tín hiệu BÁN mạnh"
                elif h_val > 0 and h_val > prev_h:
                    macd_sig = "📈 MACD dương, momentum tăng"
                elif h_val > 0:
                    macd_sig = "⚠️ MACD dương nhưng yếu dần"
                elif h_val < 0 and h_val < prev_h:
                    macd_sig = "📉 MACD âm, momentum giảm"
                else:
                    macd_sig = "⚠️ MACD âm đang phục hồi"
            else:
                macd_sig = "MACD dương" if h_val > 0 else "MACD âm"
            result["indicators"]["MACD"] = {
                "macd":      round(float(macd_line.iloc[-1]), 4),
                "signal":    round(float(signal_line.iloc[-1]), 4),
                "histogram": h_val,
                "interpretation": macd_sig,
            }
            result["signals"].append(f"MACD histogram={h_val:.4f} → {macd_sig}")

        if "BB" in ind_list or "BOLLINGER" in ind_list:
            sma20 = closes.rolling(20).mean()
            std20 = closes.rolling(20).std()
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            u  = float(upper.iloc[-1])
            m  = float(sma20.iloc[-1])
            lo = float(lower.iloc[-1])
            pct_b = round((latest_close - lo) / (u - lo + 1e-9), 4)
            bw = round(float(((upper - lower) / sma20).iloc[-1] * 100), 2)
            if latest_close > u:
                bb_sig = "⚠️ Phá vỡ dải trên - Quá mua"
            elif latest_close < lo:
                bb_sig = "✅ Phá vỡ dải dưới - Cơ hội"
            elif pct_b > 0.8:
                bb_sig = "📈 Gần dải trên - Thận trọng"
            elif pct_b < 0.2:
                bb_sig = "📉 Gần dải dưới - Cơ hội mua"
            else:
                bb_sig = "➡️ Trong dải BB"
            result["indicators"]["BB"] = {
                "dai_tren_dong": _norm_p(u), "trung_binh_dong": _norm_p(m), "dai_duoi_dong": _norm_p(lo),
                "bang_rong_pct": bw, "pct_b": pct_b, "signal": bb_sig,
            }

        if "STOCH" in ind_list:
            lo14   = lows.rolling(14).min()
            hi14   = highs.rolling(14).max()
            stoch_k = 100 * (closes - lo14) / (hi14 - lo14 + 1e-9)
            stoch_d = stoch_k.rolling(3).mean()
            k = round(float(stoch_k.iloc[-1]), 2)
            d = round(float(stoch_d.iloc[-1]), 2)
            st_sig = "⚠️ Quá mua" if k >= 80 else ("✅ Quá bán" if k <= 20 else "➡️ Trung tính")
            result["indicators"]["Stochastic"] = {"K": k, "D": d, "signal": st_sig}

        if "ATR" in ind_list:
            tr  = pd.concat([
                highs - lows,
                (highs - closes.shift(1)).abs(),
                (lows  - closes.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            atr_val = float(atr.iloc[-1])
            atr_pct = round(atr_val / latest_close * 100, 2)
            result["indicators"]["ATR"] = {
                "value_dong": _norm_p(atr_val),
                "atr_pct": atr_pct,
                "ghi_chu": f"Biến động TB ngày ≈ {atr_pct}% ({_norm_p(atr_val)} đồng)",
            }

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Lỗi tính chỉ báo: {str(e)}"})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 - News & Sentiment
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_news_and_sentiment(ticker: str, max_articles: int = 15) -> str:
    """
    Lấy tin tức gần nhất và đánh giá sentiment từ CafeF dùng ViSoBERT.

    Args:
        ticker: Mã chứng khoán (VD: VNM, HPG, ACB)
        max_articles: Số bài báo tối đa (5–30)
    """
    ticker = ticker.strip().upper()
    max_articles = max(5, min(30, max_articles))
    try:
        from sentiment_agent import (
            _collect_articles, _score_text, _aggregate_sentiment,
            _get_company_keywords, _is_article_relevant
        )
        articles = _collect_articles(ticker, max_articles=max_articles)
        if not articles:
            return json.dumps({
                "ticker": ticker, "status": "no_articles",
                "message": f"Không tìm thấy bài viết cho {ticker}",
                "sentiment": "neutral", "score": 0.0,
            })
        kws      = _get_company_keywords(ticker)
        relevant = [a for a in articles if _is_article_relevant(a, ticker, kws)] or articles
        scored   = []
        for art in relevant[:max_articles]:
            text = (art.get("title", "") + " " + art.get("snippet", "")).strip()
            if not text:
                continue
            s = _score_text(text)
            scored.append({
                "title": art.get("title", "")[:100],
                "date":  art.get("date", ""),
                "url":   art.get("url", ""),
                "label": s["label"],
                "score": round(s["numeric_score"], 3),
            })
        if not scored:
            return json.dumps({"ticker": ticker, "error": "Không score được bài nào"})
        agg = _aggregate_sentiment([{"numeric_score": a["score"], "label": a["label"]} for a in scored])
        label_vi = {"positive": "Tích cực 🟢", "negative": "Tiêu cực 🔴", "neutral": "Trung tính ⚪"}
        return json.dumps({
            "ticker": ticker,
            "tong_bai": len(scored),
            "sentiment": label_vi.get(agg["label"], agg["label"]),
            "diem_tb": round(agg.get("avg_score", 0.0), 4),
            "tich_cuc": agg.get("positive", 0),
            "tieu_cuc": agg.get("negative", 0),
            "trung_tinh": agg.get("neutral_count", 0),
            "articles": scored[:8],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": f"Lỗi: {str(e)}"})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 - Financial Statements
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_financial_statements(
    ticker: str, 
    statement_type: str = "income_statement", 
    period: str = "quarter", 
    limit: int = 4
) -> str:
    """
    Lấy dữ liệu Báo cáo tài chính của doanh nghiệp.
    Args:
        ticker: Mã cổ phiếu (VD: FPT, HPG).
        statement_type: Loại báo cáo ('income_statement' - KQKD, 'balance_sheet' - CDKT, 'cash_flow' - LCTT, 'ratio' - Chỉ số).
        period: Kỳ báo cáo ('quarter' - Quý hoặc 'year' - Năm).
        limit: Số lượng kỳ (quý/năm) gần nhất cần lấy. Mặc định là 4.
    """
    from vnstock import Finance
    try:
        # 1. Ánh xạ parameter đầu vào
        st_type_map = {
            "income": "income_statement",
            "income_statement": "income_statement",
            "kqkd": "income_statement",
            "balance": "balance_sheet",
            "balance_sheet": "balance_sheet",
            "cdkt": "balance_sheet",
            "cashflow": "cash_flow",
            "cash_flow": "cash_flow",
            "lctt": "cash_flow",
            "ratio": "ratio",
            "chi_so": "ratio"
        }
        
        rep_type = st_type_map.get(statement_type.lower(), "income_statement")
        per = "quarter" if "quarter" in period.lower() or "quy" in period.lower() else "year"
        
        # 2. Cơ chế Fallback với bộ thư viện Finance độc lập mới
        sources = ["KBS", "VCI", "TCBS", "SSI"]
        df = None
        used_source = ""
        error_logs = []

        for src in sources:
            try:
                # Khởi tạo class Finance chuẩn
                finance = Finance(symbol=ticker.upper(), source=src)
                
                # Gọi hàm trực tiếp từ object finance
                if rep_type == "income_statement":
                    df_temp = finance.income_statement(period=per)
                elif rep_type == "balance_sheet":
                    df_temp = finance.balance_sheet(period=per)
                elif rep_type == "cash_flow":
                    df_temp = finance.cash_flow(period=per)
                elif rep_type == "ratio":
                    df_temp = finance.ratio(period=per)
                else:
                    df_temp = finance.income_statement(period=per)
                
                if df_temp is not None and not df_temp.empty:
                    df = df_temp
                    used_source = src
                    break
            except Exception as e:
                error_logs.append(f"[{src}]: {str(e)}")
                continue
        
        # 3. DIRECT API BYPASS (Phương án dự phòng tối thượng khi vnstock bị lỗi chặn)
        if df is None or df.empty:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json"
                }
                is_yearly = "1" if per == "year" else "0"
                tcbs_ep = ""
                
                if rep_type == "income_statement": tcbs_ep = "incomestatement"
                elif rep_type == "balance_sheet": tcbs_ep = "balancesheet"
                elif rep_type == "cash_flow": tcbs_ep = "cashflow"
                elif rep_type == "ratio": tcbs_ep = "ratio"
                
                if tcbs_ep:
                    url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{ticker.upper()}/{tcbs_ep}?yearly={is_yearly}&isAll=true"
                    resp = requests.get(url, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        df = pd.DataFrame(resp.json())
                        used_source = "TCBS_Direct_API"
            except Exception as e:
                error_logs.append(f"[Direct_API]: {str(e)}")

        # Kiểm tra lần cuối
        if df is None or df.empty:
            return json.dumps({
                "error": f"Hệ thống không thể tải dữ liệu {rep_type} cho mã {ticker.upper()}.",
                "chi_tiet_loi": " | ".join(error_logs)
            }, ensure_ascii=False)
        
        # 4. Xử lý thuật toán Cấu trúc Dữ liệu Động (Dynamic DataFrame Parser)
        # Nhận diện cột theo định dạng năm/quý (VD: '2023', '2023-Q4')
        period_cols = [c for c in df.columns if re.search(r'^\d{4}', str(c)) or re.search(r'Q\d-\d{4}', str(c))]
        
        if len(period_cols) > 0 and len(period_cols) < len(df.columns) - 1:
            # Cấu trúc CỘT (vd: KBS): Cắt lấy 'limit' cột kỳ báo cáo đầu tiên
            base_cols = [c for c in df.columns if c not in period_cols]
            selected_periods = period_cols[:limit]
            df_final = df[base_cols + selected_periods].copy()
        else:
            # Cấu trúc DÒNG (vd: VCI, TCBS): Lấy 'limit' dòng đầu tiên
            df_final = df.head(limit).copy()
        
        # 5. Làm sạch dữ liệu rác (NaN, NaT) để serialize JSON không bị lỗi
        records = df_final.to_dict(orient="records")
        cleaned_records = []
        for row in records:
            clean_row = {}
            for k, v in row.items():
                if pd.isna(v):
                    clean_row[k] = None
                else:
                    clean_row[k] = v
            cleaned_records.append(clean_row)
        
        # 6. Build JSON output
        result = {
            "ticker": ticker.upper(),
            "statement_type": rep_type,
            "period": per,
            "data": cleaned_records,
            "ghi_chu": f"Dữ liệu BCTC được lấy qua vnstock.api.finance (Nguồn: {used_source})."
        }
        
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": f"Lỗi không xác định khi parse BCTC: {str(e)}"
        }, ensure_ascii=False)
    
# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 - Compare Stocks
# ─────────────────────────────────────────────────────────────────────────────

@tool
def compare_stocks(tickers: str, period: str = "3m") -> str:
    """
    So sánh hiệu suất nhiều cổ phiếu trong cùng khoảng thời gian.
    Giá tính bằng đồng (VND).

    Args:
        tickers: Danh sách mã cách nhau bằng dấu phẩy (VD: "VNM,HPG,ACB,FPT")
        period: Khoảng thời gian ("1m","3m","6m","1y")
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")][:6]
    s, e = _parse_period(period)
    lookback = (datetime.now() - datetime.strptime(s, "%Y-%m-%d")).days + 60
    results = []
    errors  = []
    try:
        from realtime_loader import fetch_realtime_ohlcv
        for ticker in ticker_list:
            try:
                df, err = fetch_realtime_ohlcv(symbol=ticker, interval="1d",
                                               lookback_days=lookback, tail=400)
                if err or df.empty:
                    errors.append(f"{ticker}: {err}"); continue
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                mask = df["Datetime"].dt.strftime("%Y-%m-%d").between(s, e)
                df = df[mask].reset_index(drop=True)
                if df.empty or len(df) < 2:
                    errors.append(f"{ticker}: không có dữ liệu"); continue
                closes  = df["Close"].astype(float)
                first_p = float(closes.iloc[0])
                last_p  = float(closes.iloc[-1])
                chg_pct = (last_p - first_p) / first_p * 100
                vol = float(closes.pct_change().std() * 100 * math.sqrt(252))
                delta   = closes.diff()
                gain    = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
                loss    = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
                rsi     = 100 - 100 / (1 + gain / (loss + 1e-9))
                results.append({
                    "ticker":     ticker,
                    "gia_dau":    round(first_p, 0),
                    "gia_cuoi":   round(last_p, 0),
                    "don_vi_gia": "đồng",
                    "thay_doi_%": round(chg_pct, 2),
                    "cao_nhat":   round(float(df["High"].max()), 0),
                    "thap_nhat":  round(float(df["Low"].min()), 0),
                    "bien_dong_%_nam": round(vol, 1),
                    "rsi":        round(float(rsi.iloc[-1]), 1),
                    "so_phien":   len(df),
                })
            except Exception as ex:
                errors.append(f"{ticker}: {str(ex)[:60]}")
        results.sort(key=lambda x: x.get("thay_doi_%", 0), reverse=True)
        return json.dumps({
            "ky": f"{s} → {e}", "stocks": results, "loi": errors
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 - Market Overview 
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_market_overview(period: str = "1m") -> str:
    """
    Tổng quan thị trường chứng khoán VN: VN-Index, HNX-Index, VN30.
    CHÚ Ý: Chỉ số thị trường tính bằng ĐIỂM (points), không phải đồng.

    Args:
        period: Khoảng thời gian ("1w","1m","3m")
    """
    indices  = ["VNINDEX", "VN30", "HNXIndex"]
    s, e     = _parse_period(period)
    lookback = (datetime.now() - datetime.strptime(s, "%Y-%m-%d")).days + 30
    result   = {
        "period":   f"{s} → {e}",
        "don_vi":   "điểm (points) - KHÔNG phải đồng VND",
        "ghi_chu":  "VN-Index, HNX-Index, VN30 đều tính bằng điểm",
        "indices":  {},
    }
    try:
        from realtime_loader import fetch_realtime_ohlcv
        for idx in indices:
            try:
                df, err = fetch_realtime_ohlcv(symbol=idx, interval="1d",
                                               lookback_days=lookback, tail=100)
                if err or df.empty:
                    continue
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                latest = df.iloc[-1]
                prev   = df.iloc[-2] if len(df) > 1 else latest
                first  = df.iloc[0]
                close_latest = float(latest["Close"])
                close_prev   = float(prev["Close"])
                close_first  = float(first["Close"])
                chg_day      = close_latest - close_prev
                chg_day_pct  = chg_day / close_prev * 100 if close_prev else 0
                chg_period   = (close_latest - close_first) / close_first * 100 if close_first else 0
                result["indices"][idx] = {
                    "gia_tri_diem":     _norm_index(close_latest),
                    "don_vi":           "điểm",
                    "thay_doi_ngay":    _norm_index(chg_day),
                    "thay_doi_ngay_%":  round(chg_day_pct, 2),
                    "thay_doi_ky_%":    round(chg_period, 2),
                    "cao_nhat_ky":      _norm_index(float(df["High"].max())),
                    "thap_nhat_ky":     _norm_index(float(df["Low"].min())),
                    "xu_huong_5_phien": [_norm_index(float(r["Close"])) for _, r in df.tail(5).iterrows()],
                    "ngay_cap_nhat":    str(latest["Datetime"])[:10],
                }
            except Exception as ex:
                result["indices"][idx] = {"error": str(ex)[:80]}
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Lỗi: {str(e)}"})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 8 - Screen Stocks
# ─────────────────────────────────────────────────────────────────────────────

@tool
def screen_stocks(
    exchange: str = "HOSE",
    min_change_pct: float = 3.0,
    period: str = "1m",
    top_n: int = 10,
) -> str:
    """
    Lọc cổ phiếu nổi bật theo hiệu suất giá. Giá tính bằng đồng.

    Args:
        exchange: Sàn ("HOSE","HNX","UPCOM","ALL")
        min_change_pct: Biến động tối thiểu (%), âm để lọc giảm
        period: Khoảng thời gian ("1w","1m","3m")
        top_n: Số cổ phiếu trả về (tối đa 20)
    """
    top_n = min(20, max(3, top_n))
    s, e  = _parse_period(period)
    lookback = (datetime.now() - datetime.strptime(s, "%Y-%m-%d")).days + 30
    UNIVERSE = {
        "HOSE": [
            "VNM","VIC","VHM","VCB","BID","CTG","ACB","MBB","TCB","VPB",
            "HPG","MSN","MWG","FPT","SSI","VND","HCM","GAS","SAB","PLX",
            "NVL","PDR","DXG","KDH","VRE","DGC","HSG","NKG","PNJ","HDB",
            "SHB","LPB","VIB","EIB","STB","OCB","TPB","MSB","VCI","BCM",
        ],
        "HNX":   ["PVS","SHS","NVB","VCS","BCC","TNG","NTP","OIL","SSB"],
        "UPCOM": ["ACV","BSR","VEA","BAF","GSM","MCH"],
    }
    if exchange.upper() == "ALL":
        tickers = UNIVERSE["HOSE"] + UNIVERSE["HNX"] + UNIVERSE["UPCOM"]
    else:
        tickers = UNIVERSE.get(exchange.upper(), UNIVERSE["HOSE"])
    results = []
    try:
        from realtime_loader import fetch_realtime_ohlcv
        for ticker in tickers:
            try:
                df, err = fetch_realtime_ohlcv(symbol=ticker, interval="1d",
                                               lookback_days=lookback, tail=60)
                if err or df.empty or len(df) < 2:
                    continue
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df2 = df[df["Datetime"].dt.strftime("%Y-%m-%d").between(s, e)]
                if df2.empty or len(df2) < 2:
                    continue
                closes  = df2["Close"].astype(float)
                chg_pct = (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0]) * 100
                if (min_change_pct >= 0 and chg_pct >= min_change_pct) or \
                   (min_change_pct < 0 and chg_pct <= min_change_pct):
                    results.append({
                        "ticker":     ticker,
                        "thay_doi_%": round(chg_pct, 2),
                        "gia_dong":   round(float(closes.iloc[-1]), 0),
                        "don_vi_gia": "đồng",
                    })
            except Exception:
                continue
        results.sort(key=lambda x: abs(x["thay_doi_%"]), reverse=True)
        return json.dumps({
            "san": exchange, "ky": f"{s} → {e}",
            "loc": f"biến động {'≥' if min_change_pct >= 0 else '≤'} {min_change_pct}%",
            "tim_thay": len(results),
            "top": results[:top_n],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 9 - Brokerage Research Reports
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_brokerage_research_reports(
    ticker: str,
    max_articles: int = 10,
    sources: str = "all",
) -> str:
    """
    Tổng hợp báo cáo phân tích từ nhiều công ty chứng khoán.
    Hiển thị: khuyến nghị (MUA/BÁN/GIỮ), giá mục tiêu, công ty phân tích.

    Args:
        ticker: Mã cổ phiếu (VD: VNM, ACB, HPG)
        max_articles: Số báo cáo tối đa (5–20)
        sources: Nguồn: "all" | "cafef" | "vndirect" | "ssi"
    """
    ticker = ticker.strip().upper()
    max_articles = max(5, min(20, max_articles))
    all_reports = []
    source_status = {}

    if _DATA_SOURCES_AVAILABLE:
        try:
            reports = get_multi_source_research_reports(ticker, max_per_source=max_articles)
            all_reports.extend(reports)
            source_status["tong_hop"] = f"{len(reports)} báo cáo từ đa nguồn"
        except Exception as e:
            source_status["loi"] = str(e)[:80]

        try:
            recs = fetch_vndirect_analyst_recs(ticker)
            for rec in recs[:5]:
                all_reports.insert(0, {
                    "title":          f"[{rec.get('brokerName','VNDIRECT')}] {rec.get('title', ticker)} - {rec.get('recommendation','')}",
                    "date":           rec.get("publishedDate", ""),
                    "nguon":          rec.get("brokerName", "VNDIRECT"),
                    "khuyen_nghi":    rec.get("recommendation", ""),
                    "gia_muc_tieu":   str(rec.get("targetPrice", "")),
                    "phan_tich_vien": rec.get("analystName", ""),
                    "url":            rec.get("reportUrl", ""),
                })
            if recs:
                source_status["vndirect"] = f"{len(recs)} khuyến nghị"
        except Exception as e:
            source_status["vndirect_loi"] = str(e)[:60]

    # Deduplicate
    seen = set()
    unique_reports = []
    for rp in all_reports:
        key = rp.get("title", "")[:60].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique_reports.append(rp)

    # Summary recommendations
    rec_summary = {"MUA": 0, "BÁN": 0, "GIỮ": 0}
    target_prices = []
    for rp in unique_reports:
        rec = str(rp.get("khuyen_nghi") or rp.get("recommendation") or "")
        if any(k in rec.upper() for k in ["MUA", "BUY", "OUTPERFORM", "ADD"]):
            rec_summary["MUA"] += 1
        elif any(k in rec.upper() for k in ["BÁN", "SELL", "UNDERPERFORM", "REDUCE"]):
            rec_summary["BÁN"] += 1
        elif any(k in rec.upper() for k in ["GIỮ", "HOLD", "NEUTRAL", "NẮM GIỮ"]):
            rec_summary["GIỮ"] += 1
        tp = rp.get("gia_muc_tieu") or rp.get("target_price") or ""
        if tp:
            target_prices.append(tp)

    return json.dumps({
        "ticker":           ticker,
        "tong_bao_cao":     len(unique_reports),
        "dong_thuan":       max(rec_summary, key=rec_summary.get) if sum(rec_summary.values()) > 0 else "Chưa đủ dữ liệu",
        "phan_lo":          {k: v for k, v in rec_summary.items() if v > 0},
        "gia_muc_tieu":     target_prices[:4],
        "nguon_status":     source_status,
        "bao_cao":          unique_reports[:max_articles],
        "ghi_chu":          "Tổng hợp CafeF, VNDirect, SSI, DNSE, MBS Research",
    }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 10 - Valuation Metrics
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_valuation_metrics(ticker: str) -> str:
    """
    Lấy chỉ số định giá: P/E, P/B, EV/EBITDA, ROE, ROA, EPS, BVPS,
    biên lợi nhuận, tỷ suất cổ tức, cơ cấu nợ, vốn hóa.

    Args:
        ticker: Mã chứng khoán (VD: VNM, HPG, ACB, FPT)
    """
    ticker = ticker.strip().upper()

    if not _DATA_SOURCES_AVAILABLE:
        return json.dumps({"ticker": ticker, "error": "data_sources.py chưa được cài đặt"})

    try:
        snap = get_valuation_snapshot(ticker)
        metrics_raw = snap.get("metrics", {})

        # Chuẩn hóa đơn vị
        metrics_fmt = {}
        for k, v in metrics_raw.items():
            if v is None:
                continue
            try:
                fv = float(v)
                # ROE, ROA: TCBS trả về dạng 0.25 → phải nhân 100 để thành %
                if k in ("ROE (%)", "ROA (%)", "Biên LN gộp (%)", "Biên LN ròng (%)", "Cổ tức (%)"):
                    if abs(fv) <= 1.5:  # dạng 0-1 → nhân 100
                        metrics_fmt[k] = f"{round(fv * 100, 2)}%"
                    else:               # đã là %
                        metrics_fmt[k] = f"{round(fv, 2)}%"
                else:
                    metrics_fmt[k] = round(fv, 4)
            except (ValueError, TypeError):
                metrics_fmt[k] = v

        # Lịch sử 4 quý
        history = []
        try:
            raw_ratios = fetch_tcbs_financial_ratios(ticker, yearly=False, periods=4)
            for r in raw_ratios:
                qp = f"Q{r.get('quarter','?')}/{r.get('year','?')}"
                roe = r.get("roe")
                roa = r.get("roa")
                history.append({
                    "ky":               qp,
                    "P/E":              r.get("priceToEarning"),
                    "P/B":              r.get("priceToBook"),
                    "ROE_%":            round(roe * 100, 2) if roe else None,
                    "ROA_%":            round(roa * 100, 2) if roa else None,
                    "EPS_dong":         r.get("earningPerShare"),
                    "bien_LN_rong_%":   round(r.get("postTaxMargin", 0) * 100, 2) if r.get("postTaxMargin") else None,
                    "D/E":              r.get("debtOnEquity"),
                    "ICR_EBIT/lai_vay": r.get("ebitOnInterest"),
                })
        except Exception:
            pass

        # Cổ tức lịch sử
        dividends = []
        try:
            raw_divs = fetch_vndirect_dividends(ticker, periods=5)
            for d in raw_divs[:5]:
                dividends.append({
                    "ngay": d.get("exerciseDate") or d.get("paymentDate") or "",
                    "loai": d.get("eventCode") or d.get("dividendType") or "",
                    "gia_tri": d.get("eventValue") or d.get("cashDividend") or "",
                })
        except Exception:
            pass

        # Khuyến nghị analyst
        analyst_targets = []
        try:
            recs = fetch_vndirect_analyst_recs(ticker)
            for rec in recs[:4]:
                tp = rec.get("targetPrice")
                if tp:
                    analyst_targets.append({
                        "broker":      rec.get("brokerName", ""),
                        "khuyen_nghi": rec.get("recommendation", ""),
                        "gia_muc_tieu_dong": tp,
                        "ngay":        rec.get("publishedDate", ""),
                    })
        except Exception:
            pass

        # Quick signals
        signals = []
        pe_val = metrics_fmt.get("P/E")
        pb_val = metrics_fmt.get("P/B")
        roe_str = metrics_fmt.get("ROE (%)", "")
        if pe_val:
            try:
                pe = float(pe_val)
                if pe < 10:
                    signals.append("✅ P/E thấp (<10) - định giá hấp dẫn")
                elif pe > 25:
                    signals.append("⚠️ P/E cao (>25) - định giá đắt")
                else:
                    signals.append(f"➡️ P/E = {pe:.1f} - vùng hợp lý")
            except Exception:
                pass
        if roe_str:
            try:
                roe = float(str(roe_str).replace("%", ""))
                if roe >= 20:
                    signals.append("✅ ROE ≥20% - hiệu quả sử dụng vốn tốt")
                elif roe < 10:
                    signals.append("⚠️ ROE <10% - hiệu quả vốn kém")
            except Exception:
                pass

        return json.dumps({
            "ticker":              ticker,
            "ky_bao_cao":          snap.get("period", ""),
            "nguon":               snap.get("sources", []),
            "chi_so_hien_tai":     metrics_fmt,
            "nhan_xet_nhanh":      signals,
            "lich_su_4_quy":       history,
            "co_tuc":              dividends,
            "analyst_muc_tieu":    analyst_targets,
            "don_vi_note":         "ROE/ROA/Biên LN: %, P/E/P/B: lần, EPS/BVPS: đồng",
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"ticker": ticker, "error": f"Lỗi: {str(e)}"})


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 11 - Comprehensive Analysis
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_comprehensive_analysis(ticker: str) -> str:
    """
    Phân tích toàn diện: tài chính + định giá + broker research.
    Dùng khi hỏi 'phân tích toàn diện' hoặc 'có nên đầu tư vào X?'.

    Args:
        ticker: Mã chứng khoán (VD: VNM, ACB, FPT, HPG)
    """
    ticker = ticker.strip().upper()
    result = {"ticker": ticker, "sections": {}, "don_vi": "tỷ đồng cho tiền tệ, % cho tỷ lệ"}

    if _DATA_SOURCES_AVAILABLE:
        # Kết quả kinh doanh gần nhất
        try:
            income_raw = fetch_tcbs_income_statement(ticker, yearly=False, periods=2)
            if income_raw:
                l = income_raw[0]
                p = income_raw[1] if len(income_raw) > 1 else {}
                result["sections"]["ket_qua_kinh_doanh"] = {
                    "ky": f"Q{l.get('quarter','?')}/{l.get('year','?')}",
                    "doanh_thu_ty":   _auto_unit_billions(l.get("revenue", 0)),
                    "loi_nhuan_ty":   _auto_unit_billions(l.get("shareHolderIncome", 0)),
                    "bien_gop_%":     f"{round(l.get('grossProfitMargin', 0)*100, 1)}%",
                    "bien_rong_%":    f"{round(l.get('postTaxMargin', 0)*100, 1)}%",
                    "tang_truong_dt": f"{round(l.get('yearRevenueGrowth', 0)*100, 1)}%",
                    "tang_truong_ln": f"{round(l.get('yearShareHolderIncomeGrowth', 0)*100, 1)}%",
                }
        except Exception:
            pass

        # Chỉ số định giá
        try:
            ratios = fetch_tcbs_financial_ratios(ticker, yearly=False, periods=1)
            if ratios:
                r = ratios[0]
                roe = r.get("roe") or 0
                roa = r.get("roa") or 0
                div = r.get("dividendYield") or 0
                result["sections"]["dinh_gia"] = {
                    "P/E":      r.get("priceToEarning"),
                    "P/B":      r.get("priceToBook"),
                    "ROE_%":    round(roe * 100, 1),
                    "ROA_%":    round(roa * 100, 1),
                    "EPS_dong": r.get("earningPerShare"),
                    "D/E":      r.get("debtOnEquity"),
                    "co_tuc_%": round(div * 100, 2),
                }
        except Exception:
            pass

        # Dòng tiền
        try:
            cf = fetch_tcbs_cashflow(ticker, yearly=False, periods=1)
            if cf:
                result["sections"]["dong_tien"] = {
                    "ky":             f"Q{cf[0].get('quarter','?')}/{cf[0].get('year','?')}",
                    "dong_tien_hdkd": _auto_unit_billions(cf[0].get("fromSale", 0)),
                    "FCF_ty":         _auto_unit_billions(cf[0].get("freeCashFlow", 0)),
                }
        except Exception:
            pass

        # Broker consensus
        try:
            recs = fetch_vndirect_analyst_recs(ticker)
            if recs:
                result["sections"]["broker_consensus"] = {
                    "so_khuyen_nghi": len(recs),
                    "moi_nhat": {
                        "broker":      recs[0].get("brokerName", ""),
                        "khuyen_nghi": recs[0].get("recommendation", ""),
                        "gia_muc_tieu": recs[0].get("targetPrice", ""),
                        "ngay":        recs[0].get("publishedDate", ""),
                    },
                }
        except Exception:
            pass

        # Cổ đông lớn
        try:
            owners = fetch_vndirect_ownership(ticker)
            if owners:
                result["sections"]["co_dong_lon"] = [
                    {"ten": o.get("investorName",""), "ty_le": f"{round(float(o.get('ownedRate',0))*100,2)}%"}
                    for o in owners[:4]
                ]
        except Exception:
            pass

    result["ghi_chu"] = "Dữ liệu từ TCBS, VNDirect. Chỉ mang tính tham khảo."
    return json.dumps(result, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    get_company_info,
    get_price_history,
    calculate_technical_indicators,
    get_news_and_sentiment,
    get_financial_statements,
    get_brokerage_research_reports,
    compare_stocks,
    get_market_overview,
    screen_stocks,
    get_valuation_metrics,
    get_comprehensive_analysis,
]