"""
本地离线材料科学知识库
- 文档导入: PDF/TXT/MD/CSV/Word
- 文本分块: chunk_size=512, overlap=100
- 本地向量化: paraphrase-multilingual-MiniLM-L12-v2 (不上传第三方)
- 4个专业向量集合
- 智能检索: 相似度阈值0.3, Top3
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

import yaml
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "kb_config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

KB_CONFIG = CONFIG["knowledge_base"]
CHROMA_PATH = os.path.join(BASE_DIR, KB_CONFIG["chroma_path"])
CHUNK_SIZE = KB_CONFIG["chunk_size"]
CHUNK_OVERLAP = KB_CONFIG["chunk_overlap"]
SIMILARITY_THRESHOLD = KB_CONFIG["similarity_threshold"]
RETRIEVE_TOP_N = KB_CONFIG["retrieve_top_n"]
EMBED_MODEL = CONFIG["embed_model"]


class SentenceTransformerEmbedding(EmbeddingFunction):
    """基于sentence-transformers的本地嵌入函数"""
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


# ===================== 文档读取 =====================
def read_file(filepath):
    """读取单个文件，返回纯文本"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".md", ".txt"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".csv":
        import pandas as pd
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        return df.to_string(index=False)

    elif ext == ".docx":
        from docx import Document
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(filepath)
            return "\n".join([page.get_text() for page in doc])
        except ImportError:
            print(f"  [WARN] PyMuPDF not installed, skipping {filepath}")
            return ""

    return ""


# ===================== 文本分块 =====================
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """将文本按指定大小分块，带重叠"""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        last_period = chunk.rfind("。")
        last_newline = chunk.rfind("\n")
        split_point = max(last_period, last_newline)
        if split_point > chunk_size * 0.5:
            chunk = text[start:start + split_point + 1]
            end = start + split_point + 1

        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap

    return chunks


# ===================== 知识库管理 =====================
class KnowledgeBase:
    """本地离线材料科学知识库"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collections = {}
        for col_cfg in KB_CONFIG["collections"]:
            name = col_cfg["name"]
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"description": col_cfg["description"]}
            )
        print(f"知识库初始化完成: {len(self.collections)} 个集合")
        for name, col in self.collections.items():
            print(f"  {name}: {col.count()} 条文档")

    def import_directory(self, collection_name, dir_path):
        """导入目录下所有支持的文档"""
        if collection_name not in self.collections:
            print(f"[ERROR] 集合 {collection_name} 不存在")
            return 0

        col = self.collections[collection_name]
        count = 0

        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  [INFO] 创建目录: {dir_path}")
            return 0

        for root, dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in [".md", ".txt", ".csv", ".docx", ".pdf"]:
                    continue

                fpath = os.path.join(root, fname)
                text = read_file(fpath)
                if not text.strip():
                    continue

                chunks = chunk_text(text)
                rel_path = os.path.relpath(fpath, BASE_DIR)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{rel_path}::chunk_{i}"
                    metadata = {
                        "source": rel_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "collection": collection_name,
                    }
                    col.upsert(ids=[doc_id], documents=[chunk], metadatas=[metadata])
                    count += 1

                print(f"  [OK] {rel_path} → {len(chunks)} chunks")

        print(f"  集合 {collection_name}: 导入 {count} 条文档")
        return count

    def search(self, query, collection_name=None, top_n=RETRIEVE_TOP_N):
        """检索知识库"""
        results = []

        if collection_name and collection_name in self.collections:
            cols = {collection_name: self.collections[collection_name]}
        else:
            cols = self.collections

        for name, col in cols.items():
            if col.count() == 0:
                continue
            try:
                res = col.query(query_texts=[query], n_results=min(top_n, col.count()))
                if res["documents"] and res["documents"][0]:
                    for doc, meta, dist in zip(
                        res["documents"][0],
                        res["metadatas"][0],
                        res["distances"][0]
                    ):
                        similarity = 1 - dist
                        if similarity >= SIMILARITY_THRESHOLD:
                            results.append({
                                "collection": name,
                                "document": doc,
                                "metadata": meta,
                                "similarity": similarity,
                            })
            except Exception as e:
                print(f"  [WARN] 检索 {name} 失败: {e}")

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_n]

    def get_stats(self):
        """获取知识库统计信息"""
        stats = {}
        for name, col in self.collections.items():
            stats[name] = col.count()
        return stats

    def delete_collection(self, name):
        """删除指定集合"""
        if name in self.collections:
            self.client.delete_collection(name)
            del self.collections[name]
            print(f"  已删除集合: {name}")

    def reset(self):
        """清空所有集合"""
        for name in list(self.collections.keys()):
            self.delete_collection(name)
        self.collections = {}
        for col_cfg in KB_CONFIG["collections"]:
            name = col_cfg["name"]
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"description": col_cfg["description"]}
            )
        print("  所有集合已清空并重建")


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("材料科学知识库 - 文档导入与构建")
    print("=" * 60)

    kb = KnowledgeBase()

    print("\n[导入文档]")
    for col_cfg in KB_CONFIG["collections"]:
        name = col_cfg["name"]
        doc_path = os.path.join(BASE_DIR, col_cfg["doc_path"])
        print(f"\n  集合: {name} ({col_cfg['description']})")
        kb.import_directory(name, doc_path)

    print("\n[统计]")
    stats = kb.get_stats()
    total = sum(stats.values())
    for name, count in stats.items():
        print(f"  {name}: {count} 条")
    print(f"  总计: {total} 条")

    print("\n[测试检索]")
    test_queries = [
        "JG-1铁基合金显微硬度",
        "激光熔覆工艺参数优化",
        "XGBoost材料性能预测",
    ]
    for q in test_queries:
        print(f"\n  查询: {q}")
        results = kb.search(q)
        if results:
            for r in results:
                print(f"    [{r['collection']}] sim={r['similarity']:.3f} | {r['document'][:80]}...")
        else:
            print("    无匹配结果")

    print("\n" + "=" * 60)
    print("知识库构建完成!")
    print("=" * 60)


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
