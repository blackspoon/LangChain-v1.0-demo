"""
Math MCP 服务：
- 提供最基础的算术工具（加法、乘法）
- 通过 stdio 传输协议供 LangChain MCP 客户端调用
"""

import sys
from mcp.server.fastmcp import FastMCP


# 创建一个名为 "Math" 的 MCP 服务
mcp = FastMCP("Math")

# 定义第一个工具：加法
# 使用 @mcp.tool() 装饰器即可将函数注册为 MCP 工具。
@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两数之和，同时输出调试信息便于排查。"""
    # 注意：stdio 模式下，print 输出会被 MCP 协议吞掉
    # 使用 stderr 输出调试信息，这样可以在主进程日志中看到
    print(f"-----> [Math Server] Adding {a} and {b}", file=sys.stderr)
    result = a + b
    print(f"-----> [Math Server] Result: {result}", file=sys.stderr)
    return result

# 定义第二个工具：乘法
# 使用 @mcp.tool() 装饰器即可将函数注册为 MCP 工具。
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """计算两数之积，同时输出调试信息便于排查。"""
    # 使用 stderr 输出调试信息
    print(f"-----> [Math Server] Multiplying {a} and {b}", file=sys.stderr)
    result = a * b
    print(f"-----> [Math Server] Result: {result}", file=sys.stderr)
    return result

# 运行 MCP 服务，使用 stdio 传输协议
# transport="stdio" 表示通过标准输入输出进行通信。
if __name__ == "__main__":
    mcp.run(transport="stdio")