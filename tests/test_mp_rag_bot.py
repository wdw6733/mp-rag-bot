# -*- coding: utf-8 -*-
"""公众号 RAG 机器人 v1.1 测试。

只测本地逻辑，不连微信、不连真实 LLM、不读真实 NAS：
LLM 的 HTTP 调用全部 mock，知识库用仓库里真实的 7 篇种子文档。
v1.1 回答模式：大模型是主力，知识库只是参考；检索不到也照常让
大模型用通用知识回答，只有大模型失败时才兜底。
运行：python -m pytest tests/ -v
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mp_rag_bot as mpb
from retrieval import KnowledgeIndex, parse_markdown_chunks, tokenize

TOKEN = "test_mp_token_123"
KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")


def _sig(token, timestamp, nonce):
    return hashlib.sha1("".join(sorted([token, timestamp, nonce])).encode("utf-8")).hexdigest()


def _text_xml(to_user="gh_test", from_user="user_openid_abc", content="Docker 网络模式有哪些"):
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        "<CreateTime>1700000000</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "<MsgId>1234567890123456</MsgId>"
        "</xml>"
    )


@pytest.fixture()
def index():
    return KnowledgeIndex(KNOWLEDGE_DIR)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(mpb, "MP_TOKEN", TOKEN)
    monkeypatch.setattr(mpb, "ADMIN_TOKEN", "admin-secret-test")
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    return mpb.app.test_client()


# ---------------- 分词 ----------------

def test_tokenize_chinese_bigram():
    toks = tokenize("Docker 网络模式")
    assert "docker" in toks
    assert "网络" in toks
    assert "模式" in toks


def test_tokenize_single_char_not_a_token():
    # 单个汉字不成词，避免 "的/是" 淹没评分
    assert tokenize("的") == []
    assert tokenize("") == []


def test_tokenize_ascii_lowercased():
    assert "kubernetes" in tokenize("Kubernetes K8s")
    assert "k8s" in tokenize("K8s")


# ---------------- Markdown 分块 ----------------

def test_parse_chunks_splits_by_h2(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n\n## 第一节\n内容一\n\n## 第二节\n内容二\n", encoding="utf-8")
    chunks = parse_markdown_chunks(str(p), "a.md")
    assert len(chunks) == 2
    assert chunks[0]["heading"] == "第一节"
    assert chunks[1]["heading"] == "第二节"
    assert chunks[0]["title"] == "标题"


def test_parse_chunks_empty_file(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    assert parse_markdown_chunks(str(p), "empty.md") == []


def test_parse_chunks_missing_file():
    assert parse_markdown_chunks("/nonexistent/nope.md", "nope.md") == []


# ---------------- 检索 ----------------

def test_index_loads_seed_docs(index):
    stats = index.stats()
    assert stats["docs"] == 7
    assert stats["chunks"] >= 28  # 每篇至少 4 个 H2 块


def test_search_docker_network(index):
    res = index.search("Docker 网络模式有哪些", top_n=3)
    assert res
    assert res[0]["source"] == "04-docker-network.md"


def test_search_k8s_service(index):
    res = index.search("K8s Service 是什么", top_n=3)
    assert res
    assert res[0]["source"] == "06-kubernetes-network-service.md"


def test_search_fiber(index):
    res = index.search("光纤熔接要注意什么", top_n=3)
    assert res
    assert res[0]["source"] == "02-communication-fiber-cabling.md"


def test_search_tcp(index):
    res = index.search("TCP 三次握手是怎么回事", top_n=3)
    assert res
    assert res[0]["source"] == "01-network-tcpip.md"


def test_search_cloud_native(index):
    res = index.search("云原生都包括什么", top_n=3)
    assert res
    assert res[0]["source"] == "07-cloud-native-overview.md"


def test_search_returns_top3_with_scores(index):
    res = index.search("Docker", top_n=3)
    assert len(res) == 3
    for r in res:
        assert r["score"] > 0
        assert r["source"] and r["heading"] and r["text"]


def test_search_no_match_returns_empty(index):
    assert index.search("今天晚上吃什么火锅", top_n=3) == []


def test_search_empty_query(index):
    assert index.search("", top_n=3) == []
    assert index.search("   ", top_n=3) == []


def test_reload_rebuilds(index, tmp_path):
    # 新目录只放一篇文档，reload 后只剩它
    (tmp_path / "x.md").write_text("# 测\n\n## 甲\nhello world\n", encoding="utf-8")
    idx = KnowledgeIndex(str(tmp_path))
    assert idx.stats() == {"docs": 1, "chunks": 1}
    (tmp_path / "y.md").write_text("# 测二\n\n## 乙\nfoo bar\n", encoding="utf-8")
    docs, chunks = idx.reload()
    assert (docs, chunks) == (2, 2)


# ---------------- 签名 ----------------

def test_verify_signature_pass():
    ts, nonce = "1700000000", "abc123"
    assert mpb.verify_signature(TOKEN, ts, nonce, _sig(TOKEN, ts, nonce)) is True


def test_verify_signature_wrong_token():
    ts, nonce = "1700000000", "abc123"
    assert mpb.verify_signature(TOKEN, ts, nonce, _sig("other", ts, nonce)) is False


def test_verify_signature_missing_fields():
    assert mpb.verify_signature(TOKEN, "", "abc", "xxx") is False
    assert mpb.verify_signature("", "1", "2", "3") is False


# ---------------- XML 解析与回复 ----------------

def test_parse_text_message():
    msg = mpb.parse_mp_message(_text_xml(content="你好"))
    assert msg["msg_type"] == "text"
    assert msg["content"] == "你好"
    assert msg["from_user"] == "user_openid_abc"
    assert msg["to_user"] == "gh_test"


def test_parse_invalid_xml_raises():
    with pytest.raises(ValueError):
        mpb.parse_mp_message("<xml><broken>")


def test_build_text_reply_swaps_to_from():
    xml = mpb.build_text_reply("user_openid_abc", "gh_test", "答案")
    assert "<ToUserName><![CDATA[user_openid_abc]]></ToUserName>" in xml
    assert "<FromUserName><![CDATA[gh_test]]></FromUserName>" in xml
    assert "<Content><![CDATA[答案]]></Content>" in xml
    assert "<MsgType><![CDATA[text]]></MsgType>" in xml


def test_is_help_command():
    for cmd in ("帮助", "help", "菜单", "menu", "/help", "你好"):
        assert mpb.is_help_command(cmd) is True
    assert mpb.is_help_command("Docker 网络模式") is False
    assert mpb.is_help_command("") is False


# ---------------- LLM 与兜底 ----------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_call_llm_success(monkeypatch, index):
    chunks = index.search("Docker 网络模式", top_n=3)
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        assert headers["Authorization"] == "Bearer test-key"
        assert timeout == 4
        return _FakeResp({"choices": [{"message": {"content": "bridge 是默认模式。"}}]})

    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    ans = mpb.call_llm("Docker 网络模式有哪些", chunks)
    assert "bridge 是默认模式" in ans
    assert "04-docker-network.md" in ans  # 自动附来源
    assert captured["url"].endswith("/chat/completions")
    assert captured["model"] == "deepseek-chat"


def test_call_llm_no_key_raises(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "")
    with pytest.raises(RuntimeError):
        mpb.call_llm("Docker", chunks)


def test_call_llm_empty_content_raises(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb.requests, "post",
                        lambda *a, **k: _FakeResp({"choices": [{"message": {"content": "  "}}]}))
    with pytest.raises(RuntimeError):
        mpb.call_llm("Docker", chunks)


def test_extractive_fallback_contains_source(index):
    chunks = index.search("Docker 网络模式", top_n=3)
    text = mpb.extractive_fallback(chunks)
    assert "大模型暂时不可用" in text
    assert "04-docker-network.md" in text
    assert len(text) <= mpb.MAX_ANSWER_CHARS


def test_answer_question_llm_failure_falls_back(monkeypatch, index):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb.requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ans = mpb.answer_question("Docker 网络模式有哪些", index)
    assert "大模型暂时不可用" in ans
    assert "04-docker-network.md" in ans


def test_answer_question_no_match_still_asks_llm(monkeypatch, index):
    # v1.1：检索不到不是门槛，大模型照常用通用知识回答
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "call_llm",
                        lambda q, c: "火锅一般用牛油锅底。" if not c else "不应有摘录")
    ans = mpb.answer_question("今天晚上吃什么火锅", index)
    assert "火锅一般用牛油锅底" in ans
    assert "暂时没有" not in ans


def test_answer_question_llm_failure_no_match_short_fallback(monkeypatch, index):
    # v1.1：大模型失败 + 知识库也没相关内容 -> 一句短提示，不编摘录
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb.requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ans = mpb.answer_question("今天晚上吃什么火锅", index)
    assert ans == mpb.LLM_UNAVAILABLE_TEXT
    assert "请稍后再问" in ans


def test_answer_question_llm_failure_no_key_no_match(monkeypatch, index):
    # 未配 Key + 无匹配 -> 同样是短提示
    monkeypatch.setattr(mpb, "LLM_API_KEY", "")
    ans = mpb.answer_question("今天晚上吃什么火锅", index)
    assert ans == mpb.LLM_UNAVAILABLE_TEXT


def test_extractive_fallback_empty_chunks():
    assert mpb.extractive_fallback([]) == mpb.LLM_UNAVAILABLE_TEXT


def test_call_llm_no_chunks_still_answers(monkeypatch):
    # 无摘录时 user 消息里不带知识库摘录，也不附来源
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["user"] = json["messages"][1]["content"]
        captured["system"] = json["messages"][0]["content"]
        return _FakeResp({"choices": [{"message": {"content": "通用知识答案。"}}]})

    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    ans = mpb.call_llm("今天晚上吃什么火锅", [])
    assert ans == "通用知识答案。"
    assert "知识库摘录" not in captured["user"]
    assert "来源：" not in ans


def test_call_llm_configurable_model_and_base(monkeypatch, index):
    # 模型/地址/超时/token 上限全部来自环境变量，调用时原样透传
    chunks = index.search("Docker 网络模式", top_n=1)
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        captured["timeout"] = timeout
        captured["max_tokens"] = json["max_tokens"]
        captured["system"] = json["messages"][0]["content"]
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(mpb, "LLM_API_KEY", "k2")
    monkeypatch.setattr(mpb, "LLM_API_BASE", "https://example.com/v9/")
    monkeypatch.setattr(mpb, "LLM_MODEL", "my-model-xyz")
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 7)
    monkeypatch.setattr(mpb, "LLM_MAX_TOKENS", 123)
    monkeypatch.setattr(mpb, "SYSTEM_PROMPT", "自定义提示词")
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    mpb.call_llm("Docker 网络模式", chunks)
    assert captured["url"] == "https://example.com/v9/chat/completions"
    assert captured["model"] == "my-model-xyz"
    assert captured["timeout"] == 7
    assert captured["max_tokens"] == 123
    assert captured["system"] == "自定义提示词"


def test_call_llm_truncates_to_1000(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb.requests, "post",
                        lambda *a, **k: _FakeResp({"choices": [{"message": {"content": "答" * 3000}}]}))
    ans = mpb.call_llm("Docker", chunks)
    assert len(ans) <= mpb.MAX_ANSWER_CHARS == 1000


def test_help_text_mentions_llm_primary():
    assert "大模型" in mpb.HELP_TEXT
    assert "自由配置" in mpb.HELP_TEXT


def test_version_is_1_1():
    assert mpb.VERSION == "1.1.0"
    assert mpb.MAX_ANSWER_CHARS == 1000


# ---------------- HTTP 端点 ----------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["version"] == "1.1.0"
    assert data["docs"] == 7
    assert data["chunks"] >= 28


def test_get_verify_success(client):
    ts, nonce = "1700000000", "abc123"
    r = client.get("/mp/callback", query_string={
        "signature": _sig(TOKEN, ts, nonce), "timestamp": ts,
        "nonce": nonce, "echostr": "hello-echo"})
    assert r.status_code == 200
    assert r.get_data(as_text=True) == "hello-echo"


def test_get_verify_fail(client):
    r = client.get("/mp/callback", query_string={
        "signature": "bad", "timestamp": "1700000000",
        "nonce": "abc123", "echostr": "hello-echo"})
    assert r.status_code == 403


def test_post_text_with_mocked_llm(client, monkeypatch):
    monkeypatch.setattr(mpb, "call_llm",
                        lambda q, c: "bridge 是默认模式。\n\n来源：04-docker-network.md")
    r = client.post("/mp/callback", data=_text_xml(),
                    content_type="application/xml")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "bridge 是默认模式" in body
    assert "<ToUserName><![CDATA[user_openid_abc]]></ToUserName>" in body


def test_post_help_command(client):
    r = client.post("/mp/callback", data=_text_xml(content="帮助"),
                    content_type="application/xml")
    assert r.status_code == 200
    assert "TCP 三次握手" in r.get_data(as_text=True)


def test_post_non_text_ignored(client):
    xml = _text_xml().replace("text", "image", 1)
    r = client.post("/mp/callback", data=xml, content_type="application/xml")
    assert r.status_code == 200
    assert r.get_data(as_text=True) == ""


def test_post_bad_xml_returns_empty(client):
    r = client.post("/mp/callback", data="<xml><broken>",
                    content_type="application/xml")
    assert r.status_code == 200
    assert r.get_data(as_text=True) == ""


def test_post_no_match_still_answers_via_llm(client, monkeypatch):
    # v1.1：无匹配也要调 LLM，用通用知识回答，不能回"暂时没有"
    monkeypatch.setattr(mpb, "call_llm",
                        lambda q, c: "火锅一般用牛油锅底。")
    r = client.post("/mp/callback", data=_text_xml(content="今天晚上吃什么火锅"),
                    content_type="application/xml")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "火锅一般用牛油锅底" in body
    assert "暂时没有" not in body


def test_post_with_bad_signature_rejected(client):
    r = client.post("/mp/callback?signature=bad&timestamp=1&nonce=2",
                    data=_text_xml(), content_type="application/xml")
    assert r.status_code == 403


def test_admin_reload_unauthorized(client):
    r = client.post("/admin/reload")
    assert r.status_code == 401
    r = client.post("/admin/reload", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401


def test_admin_reload_success(client):
    r = client.post("/admin/reload", headers={"X-Admin-Token": "admin-secret-test"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["docs"] == 7
