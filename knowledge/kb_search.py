"""
统一检索接口
- 知识库检索: 从Chroma向量库检索相关文档
- 记忆检索: 从SQLite读取历史对话 + 向量库召回相关摘要
- Prompt组装: 知识库资料 + 长期记忆 + 历史对话 → 标准化输出
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from .knowledge_base import KnowledgeBase
from .chat_memory import ChatMemory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===================== 统一检索 =====================
class MaterialAgent:
    """材料科研Agent: 知识库 + 记忆 + Prompt组装"""

    def __init__(self):
        print("初始化知识库...")
        self.kb = KnowledgeBase()
        print("初始化对话记忆...")
        self.memory = ChatMemory()

    def search_knowledge(self, query, top_n=3):
        """从知识库检索相关资料"""
        return self.kb.search(query, top_n=top_n)

    def search_memory(self, session_id, query=None, limit=20):
        """检索对话记忆"""
        history = self.memory.get_history(session_id, limit=limit)
        summaries = self.memory.get_summaries(session_id)
        return {
            "history": history,
            "summaries": summaries,
        }

    def build_prompt(self, session_id, user_query, top_n=3):
        """
        组装标准化Prompt
        结构: 知识库资料 + 长期记忆摘要 + 历史对话 + 用户提问
        """
        parts = []

        # 1. 知识库检索
        kb_results = self.search_knowledge(user_query, top_n=top_n)
        if kb_results:
            kb_text = "\n".join([
                f"[{r['collection']}] (相似度{r['similarity']:.2f}) {r['document']}"
                for r in kb_results
            ])
            parts.append(f"【本地私有知识库检索参考资料】\n{kb_text}")

        # 2. 长期记忆摘要
        mem_data = self.search_memory(session_id, user_query)
        if mem_data["summaries"]:
            summary_text = "\n".join([
                f"- {s['summary']}" for s in mem_data["summaries"]
            ])
            parts.append(f"【长期历史实验讨论摘要】\n{summary_text}")

        # 3. 历史对话
        if mem_data["history"]:
            history_text = "\n".join([
                f"{m['role']}: {m['content']}" for m in mem_data["history"]
            ])
            parts.append(f"【本工程完整历史聊天记录】\n{history_text}")

        # 4. 身份约束 + 用户提问
        parts.append(
            "身份约束：你是材料科学科研助手。回答严格依托上方本地知识库与历史对话，"
            "针对激光熔覆、金相分析、焊接机器人、材料机器学习输出可复现方案、"
            "带注释代码、标准化工艺参数；无资料支撑需明确说明，禁止编造实验数据。\n"
            f"用户当前提问：{user_query}"
        )

        return "\n\n".join(parts)

    def chat(self, session_id, user_query, top_n=3):
        """
        完整对话流程:
        1. 读取历史
        2. 检索知识库
        3. 组装Prompt
        4. (预留)调用LLM
        5. 保存对话
        """
        # 确保会话存在
        sessions = self.memory.get_session_list()
        if not any(s["session_id"] == session_id for s in sessions):
            self.memory.new_session(session_id, session_id)

        # 保存用户消息
        self.memory.add_message(session_id, "user", user_query)

        # 组装Prompt
        prompt = self.build_prompt(session_id, user_query, top_n)

        # (预留) 这里调用LLM生成回答
        # response = call_llm(prompt)
        response = f"[Agent] 已检索知识库和历史记忆，等待LLM接入后生成回答。\n\nPrompt长度: {len(prompt)} 字符"

        # 保存助手回复
        self.memory.add_message(session_id, "assistant", response)

        # 检查是否需要摘要
        if self.memory.should_summarize(session_id):
            print(f"  [INFO] 会话 {session_id} 已达摘要阈值，建议生成摘要")

        return response, prompt

    def stats(self):
        """获取系统统计"""
        kb_stats = self.kb.get_stats()
        sessions = self.memory.get_session_list()
        return {
            "knowledge_base": kb_stats,
            "total_documents": sum(kb_stats.values()),
            "sessions": len(sessions),
            "session_list": sessions,
        }


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("材料科研Agent - 检索与Prompt组装测试")
    print("=" * 60)

    agent = MaterialAgent()

    print("\n[1] 系统统计")
    stats = agent.stats()
    print(f"  知识库文档: {stats['total_documents']} 条")
    for name, count in stats["knowledge_base"].items():
        print(f"    {name}: {count}")
    print(f"  会话数: {stats['sessions']}")

    print("\n[2] 测试对话流程")
    session_id = "JG1_laser_cladding"

    test_queries = [
        "JG-1铁基合金在不同激光功率下的显微硬度变化趋势是什么？",
        "如何优化激光熔覆工艺参数以减少气孔缺陷？",
        "XGBoost模型在材料性能预测中的应用方法",
    ]

    for q in test_queries:
        print(f"\n  用户: {q}")
        response, prompt = agent.chat(session_id, q)
        print(f"  Agent: {response[:100]}...")
        print(f"  Prompt长度: {len(prompt)} 字符")

    print("\n[3] 导出会话")
    agent.memory.export_session(session_id)

    print("\n[4] 会话列表")
    for s in agent.memory.get_session_list():
        print(f"  {s['session_id']}: {s['message_count']}条消息")

    print("\n" + "=" * 60)
    print("Agent测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
