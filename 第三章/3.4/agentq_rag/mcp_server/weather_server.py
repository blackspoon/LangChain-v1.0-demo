"""
Weather MCP 服务：
- 封装 Open-Meteo 的地理编码与天气查询 API
- 通过 streamable-http 传输协议提供城市天气检索工具
"""

import sys
from mcp.server.fastmcp import FastMCP
import httpx
from typing import Dict, Any, Optional

mcp = FastMCP("Weather")

# ---------------------------------------------------------------------
# 定义 Open-Meteo 公共 API 地址（无需 API Key）
#   1. WEATHER_URL —— 根据经纬度查询当前天气
#   2. GEOCODE_URL —— 根据城市名查询经纬度
# ---------------------------------------------------------------------
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# 工具一：geocode_city()
# 将城市名解析为经纬度信息
@mcp.tool()
def geocode_city(name: str, country: Optional[str] = None, language: str = "zh") -> Dict[str, Any]:
    """将城市名解析为经纬度，返回首个匹配结果的基础信息。"""
    params = {"name": name, "count": 1, "language": language, "format": "json"}
    if country:
        params["country"] = country
    with httpx.Client(timeout=10) as client:
        r = client.get(OPEN_METEO_GEOCODE_URL, params=params)
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if not results:
        return {"error": f"未找到城市：{name}"}
    top = results[0]
    # 使用 stderr 输出调试信息，避免干扰 HTTP 响应
    print(f"-----> [Weather Server] Geocoding {name} to {top.get('latitude')}, {top.get('longitude')}", file=sys.stderr)
    return {
        "name": top.get("name"),
        "lat": top.get("latitude"),
        "lon": top.get("longitude"),
        "country": top.get("country"),
    }

# 工具二：get_current_weather()
# 根据经纬度查询当前天气
@mcp.tool()
def get_current_weather(lat: float, lon: float) -> Dict[str, Any]:
    """根据经纬度查询当前天气，返回温度、风速等核心指标。"""
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    with httpx.Client(timeout=10) as client:
        r = client.get(OPEN_METEO_WEATHER_URL, params=params)
        r.raise_for_status()
        payload = r.json()
    cw = payload.get("current_weather") or {}
    print(f"-----> [Weather Server] Getting current weather for {lat}, {lon}: {cw}", file=sys.stderr)
    return {
        "latitude": lat,
        "longitude": lon,
        "temperature": cw.get("temperature"),
        "windspeed": cw.get("windspeed"),
        "weathercode": cw.get("weathercode"),
        "time": cw.get("time"),
    }

# 工具三：get_current_weather_by_city()
# 组合前两个工具，实现「城市名 → 当前天气」的完整流程
@mcp.tool()
def get_current_weather_by_city(name: str, country: Optional[str] = None, language: str = "zh") -> Dict[str, Any]:
    """城市名 -> 当前天气（内部先地理编码再查询天气）。"""
    g = geocode_city(name=name, country=country, language=language)
    if "error" in g:
        return g
    w = get_current_weather(lat=g["lat"], lon=g["lon"])
    print(f"-----> [Weather Server] Getting current weather by city {name}: {w}", file=sys.stderr)
    return {**g, **w}

# transport="streamable-http" 表示使用 HTTP 协议暴露服务
# 端点默认路径是 /mcp，端口通常为 8000
if __name__ == "__main__":
    print("Starting Weather MCP Server (streamable-http) on http://localhost:8000/mcp ...")
    mcp.run(transport="streamable-http")