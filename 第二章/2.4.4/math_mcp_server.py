from mcp.server.fastmcp import FastMCP

# 创建一个名为 "Math" 的 MCP 服务
mcp = FastMCP("Math")

# 定义第一个工具：加法
# 使用 @mcp.tool() 装饰器即可将函数注册为 MCP 工具。
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# 定义第二个工具：乘法
# 使用 @mcp.tool() 装饰器即可将函数注册为 MCP 工具。
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# 运行 MCP 服务，使用 stdio 传输协议
# transport="stdio" 表示通过标准输入输出进行通信。
if __name__ == "__main__":
    mcp.run(transport="stdio")