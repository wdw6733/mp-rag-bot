# -*- coding: utf-8 -*-
"""公众号 RAG 机器人 v1.4.1 测试。

只测本地逻辑，不连微信、不连真实 LLM、不读真实 NAS：
LLM 和微信接口的 HTTP 调用全部 mock，知识库用仓库里真实的 7 篇种子文档。
v1.4 回答模式：大多数问题 4 秒内直接回答案；大模型慢时先回占位提示，
后台继续跑到总预算（默认 20 秒），推送开启（配 MP_APPID/MP_APPSECRET）
时后台查完直接推答案，没开启时发“继续”取结果。
测试里绝不真睡 20 秒：都用很小的 LLM_TIMEOUT_SEC / LLM_FIRST_WAIT_SEC。
运行：python -m pytest tests/ -v
"""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time

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


@pytest.fixture(autouse=True)
def _reset_model_state():
    # 每个测试用独立的用量文件，不碰仓库 data/usage.json、不碰 /data
    _orig_usage_file = mpb.USAGE_FILE
    tmp_usage = os.path.join(tempfile.gettempdir(), "mp-rag-bot-test-usage.json")
    if os.path.exists(tmp_usage):
        os.remove(tmp_usage)
    mpb.USAGE_FILE = tmp_usage
    mpb._MODEL_CACHE["models"] = []
    mpb._MODEL_CACHE["fetched_at"] = 0.0
    mpb._AUTO_MODEL = None
    mpb.MP_APPID = ""
    mpb.MP_APPSECRET = ""
    mpb._reset_token_cache()
    mpb._reset_jobs()
    yield
    mpb._reset_jobs()
    mpb._MODEL_CACHE["models"] = []
    mpb._MODEL_CACHE["fetched_at"] = 0.0
    mpb._AUTO_MODEL = None
    mpb.MP_APPID = ""
    mpb.MP_APPSECRET = ""
    mpb._reset_token_cache()
    if os.path.exists(tmp_usage):
        os.remove(tmp_usage)
    mpb.USAGE_FILE = _orig_usage_file


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
        assert timeout == pytest.approx(20, abs=0.01)  # v1.3 默认总预算 20 秒
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
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
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
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
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
    # v1.2：大模型失败 + 知识库也没相关内容 -> 一句短提示，不编摘录
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
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
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
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
    assert captured["timeout"] == pytest.approx(7, abs=0.01)
    assert captured["max_tokens"] == 123
    assert captured["system"] == "自定义提示词"


def test_call_llm_truncates_to_1000(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(mpb.requests, "post",
                        lambda *a, **k: _FakeResp({"choices": [{"message": {"content": "答" * 3000}}]}))
    ans = mpb.call_llm("Docker", chunks)
    assert len(ans) <= mpb.MAX_ANSWER_CHARS == 1000


def test_help_text_mentions_llm_primary():
    assert "大模型" in mpb.HELP_TEXT
    assert "自由配置" in mpb.HELP_TEXT


def test_version_is_1_4():
    assert mpb.VERSION == "1.4.1"
    assert mpb.MAX_ANSWER_CHARS == 1000
    assert mpb.LLM_TIMEOUT_SEC == 20
    assert mpb.LLM_FIRST_WAIT_SEC == 4
    assert mpb.LLM_MAX_TOKENS == 400  # v1.4：默认 400（原来 800），回答更短更快


# ---------------- HTTP 端点 ----------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["version"] == "1.4.1"
    assert data["docs"] == 7
    assert data["chunks"] >= 28
    assert data["llm_mode"] == "auto"
    assert data["llm_model"] == "auto:unknown"
    assert data["llm_timeout_sec"] == 20
    assert data["llm_first_wait_sec"] == 4
    assert data["llm_max_tokens"] == 400
    assert data["push_enabled"] is False  # 默认没配 MP_APPID/MP_APPSECRET
    assert data["async_jobs"] == 0


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

# ---------------- v1.2 模型自动选择 ----------------

class _FakeModelResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeErrorResp:
    def __init__(self, body, status_code=400):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)

    def raise_for_status(self):
        raise mpb.requests.exceptions.HTTPError(
            f"{self.status_code} Client Error", response=self)

    def json(self):
        return json.loads(self.text)


def _mock_models(monkeypatch, ids):
    def fake_get(url, headers=None, timeout=None):
        assert url.endswith("/models")
        assert headers["Authorization"] == "Bearer test-key"
        assert timeout == 3
        return _FakeModelResp({"data": [{"id": i} for i in ids]})
    monkeypatch.setattr(mpb.requests, "get", fake_get)


def test_discover_models_parses_data_list(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    _mock_models(monkeypatch, ["muse-spark", "qwen-turbo"])
    assert mpb.discover_models() == ["muse-spark", "qwen-turbo"]
    assert mpb._AUTO_MODEL == "qwen-turbo"  # 优先名单里 qwen-turbo 在前


def test_discover_models_plain_list(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")

    def fake_get(url, headers=None, timeout=None):
        return _FakeModelResp([{"id": "gpt-4o-mini"}, {"id": "x-embed-1"}])
    monkeypatch.setattr(mpb.requests, "get", fake_get)
    assert mpb.discover_models() == ["gpt-4o-mini", "x-embed-1"]


def test_discover_models_failure_returns_empty_and_no_crash(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    assert mpb.discover_models() == []
    assert mpb.pick_auto_model([]) is None


def test_discover_models_caches_10min(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _FakeModelResp({"data": [{"id": "qwen-turbo"}]})
    monkeypatch.setattr(mpb.requests, "get", fake_get)
    assert mpb.discover_models() == ["qwen-turbo"]
    assert mpb.discover_models() == ["qwen-turbo"]
    assert calls["n"] == 1
    assert mpb.discover_models(force=True) == ["qwen-turbo"]
    assert calls["n"] == 2


def test_pick_auto_prefers_deepseek_chat():
    assert mpb.pick_auto_model(["muse-spark", "deepseek-chat", "qwen-turbo"]) == "deepseek-chat"
    assert mpb.pick_auto_model(["Qwen-Turbo", "gpt-4o"]) == "Qwen-Turbo"


def test_pick_auto_skips_embedding_when_no_preferred():
    assert mpb.pick_auto_model(["text-embedding-3-small", "my-chat-1"]) == "my-chat-1"
    assert mpb.pick_auto_model(["text-embedding-3-small", "whisper-1"]) == "text-embedding-3-small"


def test_ordered_auto_models_preferred_first():
    ordered = mpb._ordered_auto_models(["muse-spark", "gpt-4o", "text-embedding-1", "custom-chat"])
    assert ordered[0] == "gpt-4o"
    assert ordered[1] == "muse-spark"
    assert "custom-chat" in ordered
    assert ordered[-1] == "text-embedding-1"


def test_model_not_found_text():
    assert mpb._model_not_found_text('model [deepseek-chat] service info not found') is True
    assert mpb._model_not_found_text('{"code":11102,"msg":"model unavailable"}') is True
    assert mpb._model_not_found_text('Model unavailable. Please switch models') is True
    assert mpb._model_not_found_text('invalid request: bad json') is False
    assert mpb._model_not_found_text('rate limit exceeded') is False
    assert mpb._model_not_found_text('') is False


def test_call_llm_auto_picks_discovered_model(monkeypatch, index):
    chunks = index.search("Docker 网络模式", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "")
    _mock_models(monkeypatch, ["muse-spark", "qwen-turbo"])
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["model"] = json["model"]
        captured["timeout"] = timeout
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    assert "ok" in mpb.call_llm("Docker 网络模式", chunks)
    assert captured["model"] == "qwen-turbo"
    assert captured["timeout"] <= 20  # v1.3 默认总预算 20 秒


def test_call_llm_configured_400_retries_next_model(monkeypatch, index):
    chunks = index.search("Docker 网络模式", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    _mock_models(monkeypatch, ["gpt-4o-mini", "qwen-turbo"])
    tried = []

    def fake_post(url, headers=None, json=None, timeout=None):
        tried.append(json["model"])
        if json["model"] == "deepseek-chat":
            return _FakeErrorResp(
                {"code": 11102, "msg": "model [deepseek-chat] service info not found"})
        return _FakeResp({"choices": [{"message": {"content": "换模型成功"}}]})
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    ans = mpb.call_llm("Docker 网络模式", chunks)
    assert "换模型成功" in ans
    assert tried[0] == "deepseek-chat"
    assert tried[1] == "qwen-turbo"  # 优先名单里 qwen-turbo 在 gpt-4o-mini 前


def test_call_llm_400_non_model_error_no_retry(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _FakeErrorResp({"message": "invalid request: bad json"})
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    with pytest.raises(mpb.requests.exceptions.HTTPError):
        mpb.call_llm("Docker", chunks)
    assert calls["n"] == 1


def test_call_llm_timeout_no_retry(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 4)
    calls = {"n": 0, "timeouts": []}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        calls["timeouts"].append(timeout)
        raise mpb.requests.exceptions.ReadTimeout("timed out")
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    with pytest.raises(mpb.requests.exceptions.ReadTimeout):
        mpb.call_llm("Docker", chunks)
    assert calls["n"] == 1
    assert calls["timeouts"][0] == pytest.approx(4, abs=0.01)


def test_call_llm_discovery_failure_uses_preferred_fallback(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "")
    monkeypatch.setattr(mpb.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["model"] = json["model"]
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    assert "ok" in mpb.call_llm("Docker", chunks)
    assert captured["model"] == "deepseek-chat"  # 发现失败时盲试优先名单首位


def test_call_llm_400_retry_timeouts_within_total_budget(monkeypatch, index):
    chunks = index.search("Docker", top_n=1)
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 4)
    _mock_models(monkeypatch, ["qwen-turbo"])
    timeouts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            return _FakeErrorResp({"msg": "model [deepseek-chat] service info not found"})
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    assert "ok" in mpb.call_llm("Docker", chunks)
    assert len(timeouts) == 2
    assert all(t <= 4 for t in timeouts)
    assert timeouts[1] <= timeouts[0]


def test_health_fixed_mode(monkeypatch, client):
    monkeypatch.setattr(mpb, "LLM_MODEL", "muse-spark")
    r = client.get("/health")
    data = r.get_json()
    assert data["llm_mode"] == "fixed"
    assert data["llm_model"] == "muse-spark"


def test_health_auto_after_discovery(monkeypatch, client):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "")
    _mock_models(monkeypatch, ["gpt-4o-mini"])
    mpb.discover_models()
    r = client.get("/health")
    data = r.get_json()
    assert data["llm_mode"] == "auto"
    assert data["llm_model"] == "gpt-4o-mini"


def test_startup_info_auto_discovery_failed(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "")
    monkeypatch.setattr(mpb.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    mode, model, info = mpb._llm_startup_info()
    assert (mode, model, info) == ("auto", "auto:unknown", "models=discovery failed")


def test_startup_info_auto_success(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "")
    _mock_models(monkeypatch, ["muse-spark", "deepseek-chat"])
    mode, model, info = mpb._llm_startup_info()
    assert mode == "auto"
    assert model == "deepseek-chat"
    assert info == "models=2 discovered"


def test_startup_info_fixed(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_MODEL", "my-model-xyz")
    mode, model, info = mpb._llm_startup_info()
    assert (mode, model, info) == ("fixed", "my-model-xyz", None)


def test_answer_question_auto_400_falls_back_to_extractive(monkeypatch, index):
    # 自动模式下所有候选都 400 模型不存在 -> 摘录兜底，不崩
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "")
    _mock_models(monkeypatch, ["qwen-turbo"])

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeErrorResp({"msg": "model service info not found"})
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    ans = mpb.answer_question("Docker 网络模式有哪些", index)
    assert "大模型暂时不可用" in ans
    assert "04-docker-network.md" in ans

# ---------------- v1.3 异步查询 ----------------
# 注意：这些测试绝不真睡 20 秒，都用很小的 LLM_TIMEOUT_SEC /
# LLM_FIRST_WAIT_SEC（0.1~1 秒级），再用 time.sleep 等后台线程跑完。

def _mock_slow_llm(monkeypatch, delay, answer="慢模型答案。", calls=None):
    """把 call_llm 换成睡 delay 秒再回答案的假实现。"""
    def fake_call(q, c):
        if calls is not None:
            calls["n"] += 1
        time.sleep(delay)
        return answer
    monkeypatch.setattr(mpb, "call_llm", fake_call)


def _post_text(client, content, from_user="user_openid_abc"):
    return client.post("/mp/callback", data=_text_xml(from_user=from_user, content=content),
                       content_type="application/xml")


def test_placeholder_text():
    # v1.4.1：占位提示永远只有一行，不写第二行、不提“继续”
    text = mpb._placeholder_text()
    assert text == "请稍等，查询中。。。"
    assert "继续" not in text
    assert "\n" not in text


def test_placeholder_does_not_depend_on_timeout(monkeypatch, index):
    # v1.3.1：占位提示固定文案，不再按总预算写秒数，改超时也不变
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 30)
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    _mock_slow_llm(monkeypatch, 0.5)
    ans = mpb.answer_question("Docker 网络模式有哪些", index, user_hash="u-ph")
    assert ans.startswith("请稍等，查询中。。。")
    assert "30秒" not in ans
    time.sleep(0.6)  # 等后台线程跑完再结束，避免残留线程


def test_is_continue_command():
    for cmd in ("继续", "继续查询", "好了吗", "好了么", "结果", "查看结果", "继续查", "继续。"):
        assert mpb._is_continue_command(cmd) is True
    assert mpb._is_continue_command("Docker 网络模式") is False
    assert mpb._is_continue_command("") is False
    assert mpb._is_continue_command("继续努力") is False


def test_normalize_question():
    assert mpb._normalize_question("  Docker   网络模式 ") == "docker 网络模式"
    assert mpb._normalize_question("K8s Service") == "k8s service"
    assert mpb._normalize_question("") == ""


def test_first_wait_sec_clamped(monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 10)
    assert mpb._first_wait_sec() == 4.5  # 超过 4.5 秒截断，保住 5 秒红线
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.2)
    assert mpb._first_wait_sec() == 0.2
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", "bad")
    assert mpb._first_wait_sec() == 4.0


def test_fast_path_returns_answer_directly(client, monkeypatch):
    # 快路径：大模型立刻回，直接就是答案，不出现占位提示
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "bridge 是默认模式。")
    r = _post_text(client, "Docker 网络模式有哪些")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "bridge 是默认模式" in body
    assert "请稍等，查询中。。。" not in body
    # 答案已取走并清除，再发“继续”只回帮助，不会重复发答案
    r2 = _post_text(client, "继续")
    assert "TCP 三次握手" in r2.get_data(as_text=True)


def test_slow_path_placeholder_then_continue(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.2)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.6, "慢模型答案：bridge 是默认模式。")
    r = _post_text(client, "Docker 网络模式有哪些")
    assert r.status_code == 200
    assert "请稍等，查询中。。。" in r.get_data(as_text=True)
    time.sleep(0.7)  # 等后台跑完
    r2 = _post_text(client, "继续")
    assert "慢模型答案" in r2.get_data(as_text=True)
    # 取过一次就清除，第二次“继续”回帮助
    r3 = _post_text(client, "继续")
    assert "TCP 三次握手" in r3.get_data(as_text=True)


def test_continue_while_running_returns_still_waiting(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 1.0, "慢模型答案。")
    r = _post_text(client, "Docker 网络模式有哪些")
    assert "请稍等，查询中。。。" in r.get_data(as_text=True)
    r2 = _post_text(client, "继续")
    assert "还在查询" in r2.get_data(as_text=True)
    time.sleep(1.1)
    r3 = _post_text(client, "继续")
    assert "慢模型答案" in r3.get_data(as_text=True)


def test_repeat_same_question_while_running_no_duplicate(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    calls = {"n": 0}
    _mock_slow_llm(monkeypatch, 0.8, "慢模型答案。", calls)
    r = _post_text(client, "Docker 网络模式有哪些")
    assert "请稍等，查询中。。。" in r.get_data(as_text=True)
    r2 = _post_text(client, "Docker 网络模式有哪些")
    assert "还在查询" in r2.get_data(as_text=True)
    assert calls["n"] == 1  # 没有重复开任务
    time.sleep(0.9)
    r3 = _post_text(client, "继续")
    assert "慢模型答案" in r3.get_data(as_text=True)
    assert calls["n"] == 1


def test_repeat_same_question_after_done_returns_cached(client, monkeypatch):
    # 门禁不影响本测试：把用户设为已关注 8 天（已解锁），否则重发原问题会被拦住
    _seed_usage(mpb._user_hash("user_openid_abc"), subscribe_days_ago=8, question_count=5)
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    calls = {"n": 0}
    _mock_slow_llm(monkeypatch, 0.5, "慢模型答案。", calls)
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    time.sleep(0.6)
    # 完成后重发原问题，直接拿到缓存答案，不再开新任务
    assert "慢模型答案" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    assert calls["n"] == 1
    # 缓存已取走清除，再重发原问题会开新任务（占位提示），调用次数 +1
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    time.sleep(0.6)
    assert "慢模型答案" in _post_text(client, "继续").get_data(as_text=True)
    assert calls["n"] == 2


def test_background_timeout_caches_fallback(client, monkeypatch):
    # 后台跑到总预算还没回：缓存摘录兜底，“继续”拿到的是兜底
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.2)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 1)
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")

    def fake_post(url, headers=None, json=None, timeout=None):
        time.sleep(0.6)  # 超过 1 秒总预算的模拟慢调用
        raise mpb.requests.exceptions.ReadTimeout("timed out")
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    r = _post_text(client, "Docker 网络模式有哪些")
    assert "请稍等，查询中。。。" in r.get_data(as_text=True)
    time.sleep(0.8)
    r2 = _post_text(client, "继续")
    body = r2.get_data(as_text=True)
    assert "大模型暂时不可用" in body
    assert "04-docker-network.md" in body


def test_continue_with_no_job_returns_help(client):
    r = _post_text(client, "继续")
    assert r.status_code == 200
    assert "TCP 三次握手" in r.get_data(as_text=True)


def test_help_bypasses_running_job(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 1.0, "慢模型答案。")
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    # “帮助”直接回帮助，不打断后台任务
    assert "TCP 三次握手" in _post_text(client, "帮助").get_data(as_text=True)
    time.sleep(1.1)
    assert "慢模型答案" in _post_text(client, "继续").get_data(as_text=True)


def test_job_ttl_expiry():
    job = mpb._start_job("u-ttl", "Docker 网络模式", "docker 网络模式")
    assert mpb._get_job("u-ttl") is job
    job["created_at"] = time.monotonic() - 601  # 10 分钟 TTL 已过
    assert mpb._get_job("u-ttl") is None


def test_new_question_supersedes_old(client, monkeypatch):
    # 门禁不影响本测试：把用户设为已关注 8 天（已解锁），否则新问题会被拦住
    _seed_usage(mpb._user_hash("user_openid_abc"), subscribe_days_ago=8, question_count=5)
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)

    def fake_call(q, c):
        if "Docker" in q:
            time.sleep(0.8)
            return "旧答案。"
        return "新答案：Service 是抽象。"
    monkeypatch.setattr(mpb, "call_llm", fake_call)
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    # 问新问题：直接拿到新答案，旧任务作废
    assert "新答案" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)
    # “继续”只回帮助，旧答案不会再冒出来
    assert "旧答案" not in _post_text(client, "继续").get_data(as_text=True)
    time.sleep(1.0)  # 等旧线程跑完再结束测试


def test_concurrent_users_get_own_answers(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)

    def fake_call(q, c):
        time.sleep(0.5)
        return f"答案：{q}"
    monkeypatch.setattr(mpb, "call_llm", fake_call)
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些", "user_a").get_data(as_text=True)
    assert "请稍等，查询中。。。" in _post_text(client, "K8s Service 是什么", "user_b").get_data(as_text=True)
    time.sleep(0.6)
    assert "答案：Docker 网络模式有哪些" in _post_text(client, "继续", "user_a").get_data(as_text=True)
    assert "答案：K8s Service 是什么" in _post_text(client, "继续", "user_b").get_data(as_text=True)


def test_answer_question_direct_slow_path(monkeypatch, index):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.5, "直接调用慢答案。")
    ans = mpb.answer_question("Docker 网络模式有哪些", index, user_hash="u-direct")
    assert "请稍等，查询中。。。" in ans
    time.sleep(0.6)
    ans2 = mpb.answer_question("继续", index, user_hash="u-direct")
    assert "直接调用慢答案" in ans2


def test_health_async_jobs_count(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 1.0, "慢模型答案。")
    _post_text(client, "Docker 网络模式有哪些")
    data = client.get("/health").get_json()
    assert data["async_jobs"] == 1
    time.sleep(1.1)
    assert client.get("/health").get_json()["async_jobs"] == 1  # 完成后仍缓存着，等“继续”取
    _post_text(client, "继续")
    assert client.get("/health").get_json()["async_jobs"] == 0


def test_slow_llm_threads_are_daemon(client, monkeypatch):
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.5, "慢模型答案。")
    _post_text(client, "Docker 网络模式有哪些")
    job = mpb._get_job(mpb._user_hash("user_openid_abc"))
    assert job is not None
    assert job["thread"] is not None
    assert job["thread"].daemon is True
    time.sleep(0.6)

# ---------------- v1.4 不用“继续”：直接回复加速 + 客服推送 ----------------
# 推送默认关闭（MP_APPID/MP_APPSECRET 为空）；下面用 fake appid/secret，
# token 和推送接口全部 mock，绝不连真实微信。

def _enable_push(monkeypatch):
    monkeypatch.setattr(mpb, "MP_APPID", "wx_test_appid")
    monkeypatch.setattr(mpb, "MP_APPSECRET", "test-secret")


def _mock_token(monkeypatch, calls=None, token="test-access-token"):
    def fake_get(url, params=None, headers=None, timeout=None):
        if calls is not None:
            calls["n"] += 1
        assert "cgi-bin/token" in url
        assert params["appid"] == "wx_test_appid"
        assert params["secret"] == "test-secret"
        assert params["grant_type"] == "client_credential"
        return _FakeModelResp({"access_token": token, "expires_in": 7200})
    monkeypatch.setattr(mpb.requests, "get", fake_get)


def _mock_push_post(monkeypatch, calls, errcode=0, errmsg="ok"):
    def fake_post(url, params=None, headers=None, json=None, timeout=None):
        calls["n"] += 1
        calls["url"] = url
        calls["params"] = params
        calls["payload"] = json
        assert "cgi-bin/message/custom/send" in url
        assert params["access_token"] == "test-access-token"
        return _FakeResp({"errcode": errcode, "errmsg": errmsg})
    monkeypatch.setattr(mpb.requests, "post", fake_post)


def test_push_enabled_requires_both(monkeypatch):
    assert mpb.push_enabled() is False
    monkeypatch.setattr(mpb, "MP_APPID", "wx_test_appid")
    assert mpb.push_enabled() is False  # 只有 AppID 不够
    monkeypatch.setattr(mpb, "MP_APPSECRET", "test-secret")
    assert mpb.push_enabled() is True
    monkeypatch.setattr(mpb, "MP_APPID", "")
    assert mpb.push_enabled() is False  # 只有 Secret 也不够


def test_placeholder_push_mode_no_continue_hint(monkeypatch):
    _enable_push(monkeypatch)
    text = mpb._placeholder_text()
    assert text == "请稍等，查询中。。。"
    assert "继续" not in text
    assert "\n" not in text  # 推送开启时只有一行，粉丝只管等


def test_placeholder_no_push_single_line_too():
    # v1.4.1：没开启推送时占位也只有一行，不再写“稍后发‘继续’查看结果”
    assert mpb.push_enabled() is False
    assert mpb._placeholder_text() == "请稍等，查询中。。。"


def test_placeholder_never_mentions_continue(monkeypatch):
    # v1.4.1：两种模式下占位里都没有“继续”二字
    assert "继续" not in mpb._placeholder_text()
    _enable_push(monkeypatch)
    assert "继续" not in mpb._placeholder_text()
    assert mpb._placeholder_text() == "请稍等，查询中。。。"


def test_get_access_token_caches(monkeypatch):
    _enable_push(monkeypatch)
    calls = {"n": 0}
    _mock_token(monkeypatch, calls)
    assert mpb.get_access_token() == "test-access-token"
    assert mpb.get_access_token() == "test-access-token"
    assert calls["n"] == 1  # 第二次走缓存，不再发请求
    mpb._reset_token_cache()
    assert mpb.get_access_token() == "test-access-token"
    assert calls["n"] == 2


def test_get_access_token_disabled_no_request(monkeypatch):
    # 没配 AppID/Secret：不发任何请求，直接 None
    monkeypatch.setattr(mpb.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    assert mpb.get_access_token() is None
    assert mpb.push_custom_message("user_openid_abc", "hi") is False


def test_get_access_token_failure_returns_none(monkeypatch):
    _enable_push(monkeypatch)
    monkeypatch.setattr(mpb.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    assert mpb.get_access_token() is None


def test_push_custom_message_success(monkeypatch):
    _enable_push(monkeypatch)
    _mock_token(monkeypatch)
    calls = {"n": 0}
    _mock_push_post(monkeypatch, calls)
    assert mpb.push_custom_message("user_openid_abc", "答案在这里。") is True
    assert calls["n"] == 1
    assert calls["payload"]["touser"] == "user_openid_abc"
    assert calls["payload"]["msgtype"] == "text"
    assert calls["payload"]["text"]["content"] == "答案在这里。"


def test_push_custom_message_48001_failure(monkeypatch):
    # 未认证的号调客服接口会报 48001：不重试、不抛异常，返回 False
    _enable_push(monkeypatch)
    _mock_token(monkeypatch)
    calls = {"n": 0}
    _mock_push_post(monkeypatch, calls, errcode=48001, errmsg="api unauthorized")
    assert mpb.push_custom_message("user_openid_abc", "答案在这里。") is False
    assert calls["n"] == 1  # 只试一次，不循环重试


def test_slow_path_pushes_answer_when_enabled(client, monkeypatch):
    _enable_push(monkeypatch)
    _mock_token(monkeypatch)
    push_calls = {"n": 0}
    _mock_push_post(monkeypatch, push_calls)
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.5, "慢模型答案：bridge 是默认模式。")
    r = _post_text(client, "Docker 网络模式有哪些")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "请稍等，查询中。。。" in body
    assert "继续" not in body  # 推送开启，占位提示里不提“继续”
    time.sleep(0.8)  # 等后台跑完 + 推送
    assert push_calls["n"] == 1
    assert push_calls["payload"]["text"]["content"] == "慢模型答案：bridge 是默认模式。"
    # 答案照样缓存着，粉丝发“继续”也能拿到（安全网）
    assert "慢模型答案" in _post_text(client, "继续").get_data(as_text=True)


def test_fast_path_does_not_push(client, monkeypatch):
    _enable_push(monkeypatch)
    _mock_token(monkeypatch)
    push_calls = {"n": 0}
    _mock_push_post(monkeypatch, push_calls)
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "bridge 是默认模式。")
    r = _post_text(client, "Docker 网络模式有哪些")
    assert "bridge 是默认模式" in r.get_data(as_text=True)
    assert "请稍等，查询中。。。" not in r.get_data(as_text=True)
    time.sleep(0.3)  # 后台线程跑完宽限期
    assert push_calls["n"] == 0  # 快路径直接回了答案，不再重复推


def test_push_failure_falls_back_to_continue(client, monkeypatch):
    _enable_push(monkeypatch)
    _mock_token(monkeypatch)
    push_calls = {"n": 0}
    _mock_push_post(monkeypatch, push_calls, errcode=48001, errmsg="api unauthorized")
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.5, "慢模型答案。")
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    time.sleep(0.8)
    assert push_calls["n"] == 1  # 推过一次，失败了
    # 推送失败也不丢答案，“继续”照样能取到
    assert "慢模型答案" in _post_text(client, "继续").get_data(as_text=True)


def test_slow_path_no_push_when_disabled_makes_no_token_request(client, monkeypatch):
    # 推送关闭时，后台跑完也不调 token 接口
    monkeypatch.setattr(mpb.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.5, "慢模型答案。")
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    time.sleep(0.8)
    assert "慢模型答案" in _post_text(client, "继续").get_data(as_text=True)


def test_answer_question_uses_top2_chunks(monkeypatch, index):
    # v1.4：检索只取前 2 块，提示词更小、回得更快
    captured = {}

    def fake_call(q, c):
        captured["n"] = len(c)
        return "ok"
    monkeypatch.setattr(mpb, "call_llm", fake_call)
    ans = mpb.answer_question("Docker 网络模式有哪些", index, user_hash="u-top2")
    assert ans == "ok"
    assert captured["n"] == 2


def test_call_llm_truncates_context_chunks_to_800(monkeypatch):
    # v1.4：每块上下文截到 800 字，2000 字的块不会原样塞进提示词
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["user"] = json["messages"][1]["content"]
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(mpb, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(mpb, "LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(mpb.requests, "post", fake_post)
    long_chunk = {"source": "x.md", "heading": "h", "text": "A" * 2000}
    mpb.call_llm("问题", [long_chunk])
    assert captured["user"].count("A") == 800


def test_health_push_enabled_true(monkeypatch, client):
    _enable_push(monkeypatch)
    data = client.get("/health").get_json()
    assert data["push_enabled"] is True
    assert data["llm_max_tokens"] == 400

# ---------------- v1.4 新关注门禁 ----------------
# 用量记录写进临时文件（autouse fixture 已把 USAGE_FILE 指到 /tmp），
# 不碰仓库 data/usage.json。subscribe 事件是门禁计时的唯一可信来源；
# 转发 5 篇微信不给服务器发、也没接口可查，所以只写在提示里、不自动数。

def _event_xml(event, from_user="user_openid_abc", to_user="gh_test"):
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        "<CreateTime>1700000000</CreateTime>"
        f"<MsgType><![CDATA[event]]></MsgType>"
        f"<Event><![CDATA[{event}]]></Event>"
        "</xml>"
    )


def _post_event(client, event, from_user="user_openid_abc"):
    return client.post("/mp/callback", data=_event_xml(event, from_user),
                       content_type="application/xml")


def _usage_data():
    with open(mpb.USAGE_FILE, encoding="utf-8") as f:
        return json.load(f)


def _seed_usage(user_hash, subscribe_days_ago=0, question_count=0, unlocked=False):
    entry = {
        "subscribe_time": time.time() - subscribe_days_ago * 86400,
        "subscribed": True,
        "question_count": question_count,
        "unlocked": unlocked,
        "first_question_at": time.time() - subscribe_days_ago * 86400,
    }
    with open(mpb.USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump({user_hash: entry}, f, ensure_ascii=False)


def test_subscribe_event_records_time_and_welcomes(client):
    r = _post_event(client, "subscribe")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "欢迎关注网络小徒" in body
    assert "免费提问一次" in body
    assert "转发5篇" in body
    uh = mpb._user_hash("user_openid_abc")
    data = _usage_data()
    assert uh in data
    assert data[uh]["subscribed"] is True
    assert abs(data[uh]["subscribe_time"] - time.time()) < 5


def test_first_question_allowed_second_gated(client, monkeypatch):
    calls = {"n": 0}

    def fake_call(q, c):
        calls["n"] += 1
        return "答案。"
    monkeypatch.setattr(mpb, "call_llm", fake_call)
    r1 = _post_text(client, "Docker 网络模式有哪些")
    assert "答案" in r1.get_data(as_text=True)
    assert calls["n"] == 1
    # 第二个问题：门禁拦住，大模型根本没调
    r2 = _post_text(client, "K8s Service 是什么")
    body = r2.get_data(as_text=True)
    assert "新关注用户仅可免费提问一次" in body
    assert "关注满一星期并转发5篇文章后" in body
    assert "还差7天" in body
    assert calls["n"] == 1


def test_gate_message_days_remaining(monkeypatch, index):
    uh = "u-days"
    _seed_usage(uh, subscribe_days_ago=2, question_count=1)
    ans = mpb.answer_question("K8s Service 是什么", index, user_hash=uh)
    assert "你已关注2天，还差5天" in ans


def test_seven_day_old_user_unlimited(client, monkeypatch):
    uh = mpb._user_hash("user_openid_abc")
    _seed_usage(uh, subscribe_days_ago=8, question_count=5)
    calls = {"n": 0}
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: (calls.__setitem__("n", calls["n"] + 1), "答案。")[1])
    assert "答案" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    assert "答案" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)
    assert calls["n"] == 2


def test_exactly_seven_days_unlocks(monkeypatch, index):
    uh = "u-7d"
    _seed_usage(uh, subscribe_days_ago=7, question_count=1)
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "答案。")
    assert mpb.answer_question("K8s Service 是什么", index, user_hash=uh) == "答案。"


def test_admin_unlock_allows_second_question(client, monkeypatch):
    uh = mpb._user_hash("user_openid_abc")
    _seed_usage(uh, subscribe_days_ago=0, question_count=1)
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "答案。")
    assert "仅可免费提问一次" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)
    r = client.post("/admin/unlock", json={"openid": "user_openid_abc"},
                    headers={"X-Admin-Token": "admin-secret-test"})
    assert r.status_code == 200
    assert r.get_json()["unlocked"] is True
    assert _usage_data()[uh]["unlocked"] is True
    assert "答案" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)


def test_admin_unlock_unauthorized(client):
    r = client.post("/admin/unlock", json={"openid": "user_openid_abc"})
    assert r.status_code == 401
    r = client.post("/admin/unlock", json={"openid": "user_openid_abc"},
                    headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401
    r = client.post("/admin/unlock", json={},
                    headers={"X-Admin-Token": "admin-secret-test"})
    assert r.status_code == 400


def test_gated_user_continue_still_works(client, monkeypatch):
    # 第一问是免费那次：慢模型 -> 占位 -> “继续”拿答案，门禁不拦“继续”
    monkeypatch.setattr(mpb, "LLM_FIRST_WAIT_SEC", 0.1)
    monkeypatch.setattr(mpb, "LLM_TIMEOUT_SEC", 5)
    _mock_slow_llm(monkeypatch, 0.5, "慢模型答案。")
    assert "请稍等，查询中。。。" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    time.sleep(0.7)
    assert "慢模型答案" in _post_text(client, "继续").get_data(as_text=True)
    # 免费次数已用完，再问新问题被拦住
    assert "仅可免费提问一次" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)


def test_help_bypasses_gating(client, monkeypatch):
    uh = mpb._user_hash("user_openid_abc")
    _seed_usage(uh, subscribe_days_ago=0, question_count=1)
    calls = {"n": 0}
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: (calls.__setitem__("n", calls["n"] + 1), "答案。")[1])
    r = _post_text(client, "帮助")
    assert "TCP 三次握手" in r.get_data(as_text=True)
    assert calls["n"] == 0  # 帮助不算提问，不调大模型
    assert _usage_data()[uh]["question_count"] == 1  # 次数也没变
    # 帮助之后再问新问题，照样被拦住
    assert "仅可免费提问一次" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)


def test_unsubscribe_resubscribe_keeps_earliest(client, monkeypatch):
    uh = mpb._user_hash("user_openid_abc")
    _post_event(client, "subscribe")
    first_time = _usage_data()[uh]["subscribe_time"]
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "答案。")
    assert "答案" in _post_text(client, "Docker 网络模式有哪些").get_data(as_text=True)
    assert _usage_data()[uh]["question_count"] == 1
    _post_event(client, "unsubscribe")
    assert _usage_data()[uh]["subscribed"] is False
    time.sleep(0.05)
    _post_event(client, "subscribe")
    data = _usage_data()[uh]
    assert data["subscribed"] is True
    assert data["subscribe_time"] == first_time  # 不重置关注时间
    assert data["question_count"] == 1  # 不重置提问次数，防取关重关刷次数
    assert "仅可免费提问一次" in _post_text(client, "K8s Service 是什么").get_data(as_text=True)


def test_corrupt_usage_file_no_crash(client, monkeypatch):
    with open(mpb.USAGE_FILE, "w", encoding="utf-8") as f:
        f.write("{not json at all")
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "答案。")
    r = _post_text(client, "Docker 网络模式有哪些")
    assert r.status_code == 200
    assert "答案" in r.get_data(as_text=True)  # 损坏文件从空开始，第一问照常放行


def test_first_question_without_subscribe_event_uses_first_seen(monkeypatch, index):
    # 部署前就关注的用户：没收到过 subscribe 事件，按首次提问时间起算
    monkeypatch.setattr(mpb, "call_llm", lambda q, c: "答案。")
    assert mpb.answer_question("Docker 网络模式有哪些", index, user_hash="u-early") == "答案。"
    data = _usage_data()
    assert "u-early" in data
    assert abs(data["u-early"]["subscribe_time"] - time.time()) < 5
    assert data["u-early"]["question_count"] == 1


def test_health_gated_users(client, monkeypatch):
    uh = mpb._user_hash("user_openid_abc")
    _seed_usage(uh, subscribe_days_ago=0, question_count=0)
    data = client.get("/health").get_json()
    assert data["gating_enabled"] is True
    assert data["gated_users"] == 1
    # 解锁后不再计入
    client.post("/admin/unlock", json={"openid": "user_openid_abc"},
                headers={"X-Admin-Token": "admin-secret-test"})
    assert client.get("/health").get_json()["gated_users"] == 0
