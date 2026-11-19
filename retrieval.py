#!/usr/bin/env python3
"""轻量检索：Markdown 按 H2/H3 分块 + BM25 关键词评分（纯标准库）。

没有向量数据库、没有 embedding 模型。分词用中文双字（bigram）+
英文/数字词，标题命中加权。索引在启动时从 knowledge/ 全量重建，
新增文档后调 POST /admin/reload 即可重建，不用重启容器。
"""
import math
import os
import re
from collections import Counter, defaultdict

K1 = 1.5
B = 0.75
TITLE_BOOST = 2.0

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
# 通用章节标题（是干什么的 / 怎么工作 / 动手例子 / 常见坑）切出的双字，
# 没有区分度：不滤掉的话，“吃什么火锅”会因为“什么”命中所有
# “是干什么的”章节。查询和索引都过同一道过滤。
STOP_TOKENS = frozenset({
    "是干", "干什", "什么", "么的", "是什",
    "怎么", "么工", "工作", "动手", "手例", "例子",
    "常见", "见坑", "概述",
})


def tokenize(text):
    """分词：英文/数字词小写 + 中文双字 bigram。

    例："Docker 网络模式" -> ["docker", "网络", "络模", "模式"]
    单个汉字不成词，不单独计分，避免 "的/是" 这类高频字淹没结果。
    """
    if not text:
        return []
    tokens = []
    for w in _WORD_RE.findall(text):
        tokens.append(w.lower())
    cjk = "".join(ch for ch in text if _CJK_RE.match(ch))
    for i in range(len(cjk) - 1):
        bigram = cjk[i:i + 2]
        if bigram not in STOP_TOKENS:
            tokens.append(bigram)
    return tokens


def parse_markdown_chunks(path, source):
    """把一个 Markdown 文件按 H2/H3 切块。

    返回 [{"source": 文件名, "title": 文档标题, "heading": 章节标题,
            "text": 标题 + 正文, "tokens": [...], "title_tokens": [...]}]
    H1 只当文档标题，不单独成块；无标题的正文归到"概述"块。
    空文件返回 []。
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    doc_title = source
    chunks = []
    cur_heading = "概述"
    cur_lines = []

    def flush():
        body = "\n".join(cur_lines).strip()
        if not body and cur_heading == "概述":
            return
        text = f"{cur_heading}\n{body}" if body else cur_heading
        chunks.append({
            "source": source,
            "title": doc_title,
            "heading": cur_heading,
            "text": text,
            "tokens": tokenize(text),
            "title_tokens": tokenize(f"{doc_title} {cur_heading}"),
        })

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if level == 1:
                doc_title = heading
                continue
            flush()
            cur_heading = heading
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()
    # 修正第一块的 title（H1 可能出现在文件中间之前已 flush 的块里）
    for c in chunks:
        c["title"] = doc_title
        c["title_tokens"] = tokenize(f"{doc_title} {c['heading']}")
    return [c for c in chunks if c["tokens"]]


class KnowledgeIndex:
    """内存索引：BM25 + 标题加权。"""

    def __init__(self, knowledge_dir):
        self.knowledge_dir = knowledge_dir
        self.chunks = []
        self.doc_count = 0
        self.avgdl = 0.0
        self._df = Counter()
        self._lengths = []
        self.load()

    def load(self):
        """从 knowledge_dir 重建索引。返回 (文档数, 块数)。"""
        chunks = []
        if os.path.isdir(self.knowledge_dir):
            for name in sorted(os.listdir(self.knowledge_dir)):
                if not name.lower().endswith(".md"):
                    continue
                path = os.path.join(self.knowledge_dir, name)
                if not os.path.isfile(path):
                    continue
                chunks.extend(parse_markdown_chunks(path, name))
        self.chunks = chunks
        self.doc_count = len({c["source"] for c in chunks})
        df = Counter()
        lengths = []
        for c in chunks:
            lengths.append(len(c["tokens"]))
            for tok in set(c["tokens"]):
                df[tok] += 1
        self._df = df
        self._lengths = lengths
        self.avgdl = sum(lengths) / len(lengths) if lengths else 0.0
        return self.doc_count, len(self.chunks)

    def reload(self):
        return self.load()

    def _idf(self, tok):
        n = len(self.chunks)
        df = self._df.get(tok, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25(self, chunk, query_counts):
        score = 0.0
        tf = Counter(chunk["tokens"])
        dl = len(chunk["tokens"])
        for tok, qf in query_counts.items():
            if tok not in tf:
                continue
            idf = self._idf(tok)
            denom = tf[tok] + K1 * (1 - B + B * dl / self.avgdl) if self.avgdl else tf[tok] + K1
            score += idf * (tf[tok] * (K1 + 1) / denom)
        # 标题命中加权：标题里出现查询词的块优先
        title_tf = Counter(chunk["title_tokens"])
        for tok in query_counts:
            if tok in title_tf:
                score += TITLE_BOOST * self._idf(tok)
        return score

    def search(self, query, top_n=3):
        """返回 top_n 个块，每块含 source/title/heading/text/score。

        无匹配（全 0 分）返回 []：v1.1 里这只是“没有参考摘录”，
        调用方仍会让大模型用通用知识回答，不会因此拦住回答。
        """
        if not query or not query.strip() or not self.chunks:
            return []
        query_counts = Counter(tokenize(query.strip()))
        if not query_counts:
            return []
        scored = []
        for chunk in self.chunks:
            # 多词查询至少命中 2 个不同词：单个通用词命中不算相关，
            # 例如“火锅”只命中“什么”这种已被过滤的词时不再误报。
            # 单个词的查询（例如只问“Docker”）不受此限。
            chunk_tokens = set(chunk["tokens"])
            matched = set(query_counts) & chunk_tokens
            if len(query_counts) >= 2 and len(matched) < 2:
                continue
            score = self._bm25(chunk, query_counts)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, chunk in scored[:top_n]:
            out.append({
                "source": chunk["source"],
                "title": chunk["title"],
                "heading": chunk["heading"],
                "text": chunk["text"],
                "score": round(score, 3),
            })
        return out

    def stats(self):
        return {"docs": self.doc_count, "chunks": len(self.chunks)}
