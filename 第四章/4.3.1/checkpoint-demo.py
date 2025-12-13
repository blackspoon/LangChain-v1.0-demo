from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from operator import add

# ========== 1. 状态定义 ==========
class ProcessState(TypedDict):
    """工作流状态"""
    current_task: str  # 当前任务
    completed_steps: Annotated[list[str], add]  # 已完成步骤

# ========== 2. 节点定义 ==========
def process_data_node(state: ProcessState) -> Dict[str, Any]:
    """数据处理节点"""
    task_name = "数据处理"
    step_info = f"{task_name}: 加载并验证数据"
    return {
        "current_task": task_name,
        "completed_steps": [step_info]
    }

def analyze_data_node(state: ProcessState) -> Dict[str, Any]:
    """数据分析节点"""
    task_name = "数据分析"
    step_info = f"{task_name}: 生成分析报告"
    return {
        "current_task": task_name,
        "completed_steps": [step_info]
    }

# ========== 3. 构建图 ==========
workflow = StateGraph(ProcessState)
workflow.add_node("process_data", process_data_node)
workflow.add_node("analyze_data", analyze_data_node)
workflow.add_edge(START, "process_data")
workflow.add_edge("process_data", "analyze_data")
workflow.add_edge("analyze_data", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# ========== 4. 检查点演示 ==========
def print_state(title: str, state: Any):
    """打印状态信息"""
    print(f"\n{title}")
    print("-" * 50)
    if state:
        if hasattr(state, "values"):
            print(f"当前任务: {state.values.get('current_task', 'N/A')}")
            print(f"已完成步骤: {state.values.get('completed_steps', [])}")
        else:
            print(state)
    else:
        print("状态为空")

if __name__ == "__main__":
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    
    # 获取初始状态
    initial_state = graph.get_state(config)
    print_state("初始状态", initial_state)
    
    # 执行工作流
    graph.invoke({"current_task": ""}, config)
    final_state = graph.get_state(config)
    print_state("执行后状态", final_state)
    
    # 查看状态历史（按时间顺序从旧到新）
    state_history = list(graph.get_state_history(config))
    print("\n状态历史")
    print("-" * 50)
    print(f"共 {len(state_history)} 个检查点（按时间顺序）")
    # 反转列表以从旧到新显示
    for i, state in enumerate(reversed(state_history), 1):
        print(f"\n检查点 {i}:")
        if hasattr(state, "values"):
            task = state.values.get('current_task', 'N/A')
            steps = state.values.get('completed_steps', [])
            print(f"  任务: {task if task else '(空)'}")
            print(f"  步骤: {steps if steps else '[]'}")
        else:
            print(f"  {state}")
    
    # 更新状态
    before_update_state = graph.get_state(config)
    print_state("更新前状态", before_update_state)
    
    graph.update_state(
        config, 
        {"current_task": "生成图表", "completed_steps": ["数据展示: 生成图表"]}
    )
    
    updated_state = graph.get_state(config)
    print_state("更新后状态", updated_state)
