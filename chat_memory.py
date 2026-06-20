"""
SQLite对话记忆系统
- 按session_id隔离课题
- 实时同步: 每轮问答写入数据库
- 历史召回: 读取指定session全部对话
- 会话管理: 新建/清空/导出
- 长期向量归档: 对话超阈值自动摘要
"""

import os
import sys
import json
import sqlite3
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "kb_config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MEM_CONFIG = CONFIG["chat_memory"]
SQLITE_PATH = os.path.join(BASE_DIR, MEM_CONFIG["sqlite_path"])
MAX_SHORT_ROUND = MEM_CONFIG["max_short_round"]
SUMMARY_THRESHOLD = MEM_CONFIG["summary_threshold"]


# ===================== 数据库管理 =====================
class ChatMemory:
    """SQLite持久化对话记忆"""

    def __init__(self, db_path=SQLITE_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                project_name TEXT,
                created_at TEXT,
                updated_at TEXT,
                message_count INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                summary_text TEXT,
                message_range TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        self.conn.commit()

    def new_session(self, session_id, project_name=""):
        """新建会话"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT OR REPLACE INTO sessions (session_id, project_name, created_at, updated_at, message_count) "
            "VALUES (?, ?, ?, ?, 0)",
            (session_id, project_name, now, now)
        )
        self.conn.commit()
        print(f"  [OK] 新建会话: {session_id}")
        return session_id

    def add_message(self, session_id, role, content):
        """添加一条消息"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        cursor.execute(
            "UPDATE sessions SET updated_at=?, message_count=message_count+1 WHERE session_id=?",
            (now, session_id)
        )
        self.conn.commit()

        cursor.execute("SELECT message_count FROM sessions WHERE session_id=?", (session_id,))
        count = cursor.fetchone()[0]
        return count

    def get_history(self, session_id, limit=None):
        """获取会话历史"""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id=? ORDER BY id",
                (session_id,)
            )
        rows = cursor.fetchall()
        if limit:
            rows = rows[::-1]
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def get_session_list(self):
        """获取所有会话列表"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT session_id, project_name, created_at, message_count FROM sessions ORDER BY updated_at DESC"
        )
        return [{"session_id": r[0], "project_name": r[1],
                 "created_at": r[2], "message_count": r[3]} for r in cursor.fetchall()]

    def clear_session(self, session_id):
        """清空单个会话"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
        cursor.execute("UPDATE sessions SET message_count=0 WHERE session_id=?", (session_id,))
        self.conn.commit()
        print(f"  [OK] 已清空会话: {session_id}")

    def delete_session(self, session_id):
        """删除会话"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        self.conn.commit()
        print(f"  [OK] 已删除会话: {session_id}")

    def export_session(self, session_id, output_dir=None):
        """导出会话为JSON"""
        if output_dir is None:
            output_dir = os.path.join(BASE_DIR, "chat_memory", "session_backup_json")
        os.makedirs(output_dir, exist_ok=True)

        history = self.get_history(session_id)
        cursor = self.conn.cursor()
        cursor.execute("SELECT project_name FROM sessions WHERE session_id=?", (session_id,))
        row = cursor.fetchone()
        project_name = row[0] if row else ""

        data = {
            "session_id": session_id,
            "project_name": project_name,
            "message_count": len(history),
            "exported_at": datetime.now().isoformat(),
            "messages": history,
        }

        fpath = os.path.join(output_dir, f"{session_id}_backup.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [OK] 导出: {fpath}")
        return fpath

    def add_summary(self, session_id, summary_text, message_range=""):
        """添加摘要"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO summaries (session_id, summary_text, message_range, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, summary_text, message_range, now)
        )
        self.conn.commit()

    def get_summaries(self, session_id):
        """获取会话摘要"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT summary_text, message_range, created_at FROM summaries "
            "WHERE session_id=? ORDER BY id",
            (session_id,)
        )
        return [{"summary": r[0], "range": r[1], "created_at": r[2]} for r in cursor.fetchall()]

    def should_summarize(self, session_id):
        """判断是否需要生成摘要"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT message_count FROM sessions WHERE session_id=?", (session_id,))
        row = cursor.fetchone()
        if not row:
            return False
        return row[0] >= SUMMARY_THRESHOLD

    def close(self):
        self.conn.close()


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("对话记忆系统 - 功能测试")
    print("=" * 60)

    mem = ChatMemory()

    print("\n[1] 新建会话")
    mem.new_session("JG1_laser_cladding", "JG-1激光熔覆工艺优化")
    mem.new_session("metallography_analysis", "金相图像定量分析")

    print("\n[2] 写入对话")
    test_messages = [
        ("user", "JG-1铁基合金在1800W下的显微硬度是多少？"),
        ("assistant", "根据实测数据，JG-1铁基合金在1800W激光功率下的显微硬度为385.2±111.7 HV，范围186.0-493.2 HV。"),
        ("user", "和900W相比有什么变化？"),
        ("assistant", "900W下硬度为224.2±62.7 HV。从900W到1800W，硬度提升了约72%，主要机制可能是相变强化而非Hall-Petch强化。"),
    ]
    for role, content in test_messages:
        count = mem.add_message("JG1_laser_cladding", role, content)
        print(f"  [{role}] {content[:50]}... (total: {count})")

    print("\n[3] 读取历史")
    history = mem.get_history("JG1_laser_cladding")
    print(f"  共 {len(history)} 条消息")
    for msg in history:
        print(f"  [{msg['role']}] {msg['content'][:60]}...")

    print("\n[4] 会话列表")
    sessions = mem.get_session_list()
    for s in sessions:
        print(f"  {s['session_id']}: {s['project_name']} ({s['message_count']}条)")

    print("\n[5] 导出会话")
    mem.export_session("JG1_laser_cladding")

    print("\n[6] 判断是否需要摘要")
    for s in sessions:
        needed = mem.should_summarize(s["session_id"])
        print(f"  {s['session_id']}: {'需要摘要' if needed else '不需要'}")

    mem.close()
    print("\n" + "=" * 60)
    print("记忆系统测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
