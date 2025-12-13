"""
MCP 工具装配模块：
- 通过 MultiServerMCPClient 同时连接本地 math 与 weather 服务
- 返回 LangChain Agent 可直接使用的 Tool 列表
"""

import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

def build_tools():
    """
    封装 math 与 weather 两个 MCP 服务：
    - math：通过 stdio 启动本地 math_server.py
    - weather：通过 streamable-http 连接到固定端口（http://localhost:8000/mcp）
    """
    # 获取项目根目录（agent/mcp_tools.py 的父目录的父目录）
    project_root = Path(__file__).parent.parent
    math_server_path = project_root / "mcp_server" / "math_server.py"
    
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [str(math_server_path)],
        },
        "weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",       # Weather MCP Server 固定地址
        },
    })

    # asyncio.run 会自动创建/关闭事件循环，确保在同步上下文中安全获取工具
    return asyncio.run(client.get_tools())

