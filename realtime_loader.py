import time
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK = 365
DEFAULT_TAIL     = 200
# Nguồn dữ liệu lịch sử ổn định nhất theo thứ tự
DATA_SOURCES     = ["VCI", "TCBS", "KBS", "FMP"]

_cache: dict = {}
CACHE_TTL_SECONDS = 300 

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_vnstock_available() -> bool:
    try:
        from vnstock import Vnstock
        return True
    except ImportError:
        return False

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    rename: dict = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("time", "date", "datetime", "tradingdate", "trading_date", "ngay", "time_stamp"):
            rename[col] = "Datetime"
        elif lc in ("open", "mo_cua", "open_price"): rename[col] = "Open"
        elif lc in ("high", "cao_nhat", "high_price"): rename[col] = "High"
        elif lc in ("low", "thap_nhat", "low_price"):  rename[col] = "Low"
        elif lc in ("close", "dong_cua", "close_price"): rename[col] = "Close"
        elif lc in ("volume", "vol", "total_volume", "totalvolume", "khoi_luong", "nm_volume"):
            rename[col] = "Volume"
    return df.rename(columns=rename)

# ── Core Fetching ─────────────────────────────────────────────────────────────

def _fetch_from_vnstock(symbol: str, start: str, end: str, interval: str, source: str) -> pd.DataFrame:
    try:
        from vnstock import Vnstock
        # Khởi tạo Vnstock
        vn = Vnstock()
        
        # Thử lấy qua giao diện mới nhất
        try:
            stock = vn.stock(symbol=symbol.upper(), source=source)
            df = stock.quote.history(start=start, end=end, interval=interval)
            if df is not None and not df.empty and not isinstance(df, dict):
                return _normalise_columns(df)
        except:
            pass

        # Fallback 1: Thử phương thức truyền thống của vnstock (nếu có)
        try:
            # Đối với một số phiên bản vnstock cũ hơn hoặc dùng API trực tiếp
            from vnstock import stock_historical_data
            df = stock_historical_data(symbol=symbol.upper(), start_date=start, end_date=end, resolution=interval, type='stock', source=source)
            if df is not None and not df.empty:
                return _normalise_columns(df)
        except:
            pass

        return pd.DataFrame()
    except Exception as e:
        print(f"[RealtimeLoader] Error with source {source}: {e}")
        return pd.DataFrame()

def fetch_realtime_ohlcv(
    symbol: str,
    interval: str = "1d",
    lookback_days: int = None,
    start_date: str = None,
    end_date: str = None,
    tail: int = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Truy xuất dữ liệu OHLCV + Volume.
    Tối ưu hóa theo tài liệu vnstock mới nhất.
    """
    symbol = symbol.upper()
    if lookback_days is None: lookback_days = DEFAULT_LOOKBACK
    if tail is None: tail = DEFAULT_TAIL

    cache_key = (symbol, interval, start_date, end_date)
    if use_cache and cache_key in _cache:
        ts, cached_df = _cache[cache_key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return cached_df.tail(tail).reset_index(drop=True), ""

    # Date calculation
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        except:
            return pd.DataFrame(), "Sai định dạng ngày YYYY-MM-DD"
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # Mapping interval
    ivl_map = {"1m":"1", "5m":"5", "15m":"15", "30m":"30", "1h":"60", "1d":"1D", "1w":"1W", "1mo":"1M"}
    interval_str = ivl_map.get(interval.lower(), "1D")

    df_raw = pd.DataFrame()
    success_source = None

    if check_vnstock_available():
        # Thử xoay vòng các nguồn ổn định nhất
        for source in DATA_SOURCES:
            df_raw = _fetch_from_vnstock(symbol, start_str, end_str, interval_str, source)
            if not df_raw.empty:
                success_source = source
                break
    else:
        return pd.DataFrame(), "Thư viện vnstock chưa được cài đặt."

    if df_raw.empty:
        return pd.DataFrame(), f"Không thể lấy dữ liệu lịch sử cho {symbol} từ bất kỳ nguồn nào (VCI, TCBS, KBS)."

    # Post-processing
    df_raw = _normalise_columns(df_raw)
    
    # Kiểm tra cột tối thiểu
    if "Close" not in df_raw.columns:
        return pd.DataFrame(), f"Dữ liệu từ {success_source} thiếu cột giá đóng cửa."
        
    if "Volume" not in df_raw.columns:
        df_raw["Volume"] = 0

    # Chuyển đổi Datetime
    df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    
    # Chỉ giữ các cột cần thiết
    final_cols = [c for c in REQUIRED_COLS if c in df_raw.columns]
    df_clean = df_raw[final_cols].copy()

    if use_cache:
        _cache[cache_key] = (time.time(), df_clean)

    return df_clean.tail(tail).reset_index(drop=True), ""

# ── Symbol Helpers ───────────────────────────────────────────────────────────

def get_all_symbols_realtime() -> List[dict]:
    try:
        url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/listing-all"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get("data", [])
    except:
        pass
    return []

def get_stock_info_realtime(code: str) -> dict:
    """Lấy thông tin giá hiện tại từ TCBS. Trả về dict với field names chuẩn hóa."""
    try:
        url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/{code.upper()}/stock-info"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json() or {}
            # Normalize: TCBS trả về camelCase, map về snake_case consistent
            normalized = {**data}  # giữ tất cả key gốc
            # Price fields — handle both camelCase and snake_case
            for src, dst in [
                ("organName",      "organ_name"),
                ("enOrganName",    "en_organ_name"),
                ("pctChange",      "percent"),
                ("priceChange",    "change"),
                ("lastPrice",      "close"),
                ("matchedPrice",   "close"),
                ("referencePrice", "ref_price"),
                ("industryEn",     "industry"),
                ("industryVi",     "industry_vi"),
                ("exchange",       "exchange"),
            ]:
                if src in data and dst not in normalized:
                    normalized[dst] = data[src]
            return normalized
    except Exception:
        pass
    return {}