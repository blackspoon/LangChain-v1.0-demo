"""
对话记忆模块：
- 提供轻量级滑动窗口记忆，用于保存最近若干轮会话
"""

from typing import List
from langchain_core.messages import BaseMessage


class ConversationWindow:
    """
    对话短期记忆窗口。
    - 基于 InMemorySaver 封装（仅进程内存储，不落库）
    - 按 window_size 控制保留最近若干轮对话消息
    """

    def __init__(self, window_size: int = 5):
        # 控制窗口长度，例如保留最近 5 轮会话
        self.window_size = window_size
        self.buffer: List[BaseMessage] = []

    def add(self, msgs: List[BaseMessage]):
        """新增消息并自动裁剪窗口"""
        self.buffer.extend(msgs)
        print(f"-----> Memory: Adding messages: {self.buffer}")
        if len(self.buffer) > self.window_size * 2:
            self.buffer = self.buffer[-self.window_size * 2 :]

    def get(self) -> List[BaseMessage]:
        """返回当前会话窗口中的消息"""
        print(f"-----> Memory: Getting messages: {self.buffer}")
        return list(self.buffer)

    def clear(self):
        """清空对话缓存"""
        print(f"-----> Memory: Clearing messages: {self.buffer}")
        self.buffer = []
        print(f"-----> Memory: Cleared messages: {self.buffer}")

def build_memory(window_size: int = 5) -> ConversationWindow:
    """工厂方法：创建对话记忆对象"""
    return ConversationWindow(window_size=window_size)
