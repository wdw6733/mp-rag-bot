#!/usr/bin/env python3
"""公众号轻量 RAG 机器人 v1.1。

服务号（未微信认证）发文字消息 -> 微信 POST 到 /mp/callback ->
检索 knowledge/ 里的 Markdown 作参考 -> 调用户自配的大模型综合回答 ->
5 秒内用被动回复 XML 把答案发回去。

回答模式（v1.1）：大模型是主力，知识库只是参考/ grounding，不是门槛。
检索到相关摘录就把摘录当参考发给大模型；检索不到也照常让大模型
用通用知识回答，绝不回“知识库里没有”。只有大模型不可用（未配 Key /
超时 / 报错）时才退回知识库原文摘录；知识库也没有相关内容时，
回一句简短的“大模型暂时不可用，请稍后再问”。

未认证服务号的限制（v1.1 不做、也做不了）：
- 只有 5 秒内被动回复，没有 48 小时客服消息；
- 服务器配置启用后，自定义菜单不可用。
认证（300 元/年）后才能用客服消息突破 5 秒、才能用自定义菜单，
那是可选的后续升级，不是 v1.1 的必需项。

凭证与模型只从环境变量读，绝不写进代码、README、日志。
模型不固定：LLM_API_BASE / LLM_API_KEY / LLM_MODEL 由用户在 .env 里
自由配置，改完 docker compose restart 即可，不用改代码、不用重新打镜像。
日志只记问题长度和命中来源，不记完整 OpenID（只记前 8 位哈希）。
"""
import hashlib
import os
import time
import xml.etree.ElementTree as ET

import requests
from flask import Flask, request

from retrieval import KnowledgeIndex

VERSION = "1.1.0"
PORT = int(os.environ.get("MP_PORT", "18790"))
KNOWLEDGE_DIR = os.environ.get(
    "KNOWLEDGE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge"),
)
MP_TOKEN = os.environ.get("MP_TOKEN", "").strip()
# MP_APPID / MP_APPSECRET：v1.1 不需要。客服消息接口没接（未认证服务号
# 本来也用不了），不要在 .env 里配这两个。
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://anyrouter.top/v1").strip() or "https://anyrouter.top/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
# 模型不固定：只在 .env 为空时 fallback 到 deepseek-chat，.env.example 里
# LLM_MODEL 本来就是空的，配什么模型以用户的 .env 为准。
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip() or "deepseek-chat"


def _int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


# 微信要求 5 秒内回复，LLM 超时默认 4 秒、留 1 秒余量；都可在 .env 里改。
LLM_TIMEOUT_SEC = _int_env("LLM_TIMEOUT_SEC", 4)
LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 800)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()
MAX_ANSWER_CHARS = 1000  # 5 秒被动回复要短：截断到约 1000 字（v1.0 是 1800）

DEFAULT_SYSTEM_PROMPT = (
    "你是公众号 IT 知识助手，回答网络、通信工程、Docker、Kubernetes、"
    "云原生等 IT 技术问题。用中文回答，通俗易懂的深度，不要考试级深度；"
    "回答要简短精炼，控制在 1000 字以内，优先给结论和关键步骤；"
    "不要编造命令、参数、端口、版本号或价格，不确定的就明说不确定。"
    "如果下面给了知识库摘录，把它当参考，先用摘录里的内容回答；"
    "没有摘录时用你的通用知识直接回答。"
)
SYSTEM_PROMPT = os.environ.get("LLM_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT

HELP_TEXT = (
    "你好，我是 IT 知识助手。直接发问题就行，例如：\n"
    "· TCP 三次握手是怎么回事？\n"
    "· 光纤熔接要注意什么？\n"
    "· Docker 有哪些网络模式？\n"
    "· K8s Service 是什么？\n"
    "· 云原生都包括什么？\n"
    "知识库（网络、通信工程、Docker、Kubernetes、云原生）作参考，"
    "由大模型综合回答，知识库里没有的话题也能问。\n"
    "模型可在 .env 里自由配置，不用改代码。"
)
LLM_UNAVAILABLE_TEXT = "大模型暂时不可用，请稍后再问。"
HELP_COMMANDS = {"帮助", "help", "菜单", "menu", "/help", "/menu", "你好", "hi", "hello"}


def log(*args):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, flush=True)


def _user_hash(openid):
    if not openid:
        return "anon"
    return hashlib.sha256(openid.encode("utf-8")).hexdigest()[:8]


def verify_signature(token, timestamp, nonce, signature):
    """微信公众号签名：sha1(sorted([token, timestamp, nonce]) 拼起来)。"""
    if not token or not timestamp or not nonce or not signature:
        return False
    raw = "".join(sorted([token, timestamp, nonce]))
    expect = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return expect == signature


def _xml_text(root, tag):
    el = root.find(tag)
    return (el.text or "").strip() if el is not None else ""


def parse_mp_message(xml_text):
    """解析公众号 POST 的 XML。返回 dict，解析失败抛 ValueError。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"XML 解析失败：{e}")
    return {
        "to_user": _xml_text(root, "ToUserName"),
        "from_user": _xml_text(root, "FromUserName"),
        "create_time": _xml_text(root, "CreateTime"),
        "msg_type": _xml_text(root, "MsgType"),
        "content": _xml_text(root, "Content"),
        "msg_id": _xml_text(root, "MsgId"),
    }


def build_text_reply(to_user, from_user, content):
    """被动回复文本消息 XML（To/From 对调，CDATA 包裹）。"""
    ts = int(time.time())
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{ts}</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "</xml>"
    )


def is_help_command(text):
    return (text or "").strip().lower() in HELP_COMMANDS


def format_sources(chunks):
    seen = []
    for c in chunks:
        label = f"{c['source']}（{c['heading']}）"
        if label not in seen:
            seen.append(label)
    return "来源：" + "；".join(seen)


def extractive_fallback(chunks):
    """LLM 不可用时的兜底：有相关摘录就给原文摘录，都没有就一句短回复。"""
    if not chunks:
        return LLM_UNAVAILABLE_TEXT
    parts = ["（大模型暂时不可用，以下是知识库原文摘录）"]
    for c in chunks:
        body = c["text"]
        if len(body) > 600:
            body = body[:600] + "……"
        parts.append(body)
    sources = format_sources(chunks)
    parts.append(sources)
    text = "\n\n".join(parts)
    if len(text) > MAX_ANSWER_CHARS:
        # 截断只截正文，来源行必须保留在末尾
        cut = MAX_ANSWER_CHARS - len("\n\n" + sources)
        text = text[:max(cut, 0)] + "\n\n" + sources
    return text[:MAX_ANSWER_CHARS]


def call_llm(question, chunks):
    """调 OpenAI 兼容接口。成功返回回答文本，失败抛异常（调用方兜底）。

    chunks 可以为空：有摘录时当参考发给大模型，没摘录时大模型直接
    用通用知识回答。模型、地址、超时、token 上限全部来自环境变量，
    代码里不固定任何模型选择。
    """
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY 未配置")
    if chunks:
        context = "\n\n---\n\n".join(
            f"【{c['source']} / {c['heading']}】\n{c['text']}" for c in chunks
        )
        user_content = f"知识库摘录（仅供参考）：\n{context}\n\n问题：{question}"
    else:
        user_content = f"问题：{question}"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": LLM_MAX_TOKENS,
    }
    url = LLM_API_BASE.rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {LLM_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise RuntimeError("LLM 返回空内容")
    answer = content.strip()
    if chunks and format_sources(chunks) not in answer:
        # 先给来源行留位置再截断，避免长回答把来源截掉
        sources = "\n\n" + format_sources(chunks)
        answer = answer[:max(MAX_ANSWER_CHARS - len(sources), 0)] + sources
    return answer[:MAX_ANSWER_CHARS]


def answer_question(question, index):
    """问答主流程：检索作参考 -> 大模型综合回答（失败则摘录兜底）。

    检索不到不拦路：大模型照常回答；大模型失败且检索有结果才回摘录，
    大模型失败且检索也没结果时回一句短的“请稍后再问”。
    """
    chunks = index.search(question, top_n=3)
    log("提问 len=", len(question), "命中：",
        ", ".join(f"{c['source']}#{c['heading']}" for c in chunks) or "无")
    try:
        return call_llm(question, chunks)
    except Exception as e:
        log("LLM 调用失败，改用兜底：", type(e).__name__, str(e)[:200])
        return extractive_fallback(chunks)


app = Flask(__name__)
INDEX = KnowledgeIndex(KNOWLEDGE_DIR)


@app.route("/health", methods=["GET"])
def health():
    stats = INDEX.stats()
    return {"status": "ok", "version": VERSION, **stats}


@app.route("/mp/callback", methods=["GET"])
def mp_verify():
    """公众号服务器配置验证：验签通过原样返回 echostr，否则 403。"""
    signature = request.args.get("signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    echostr = request.args.get("echostr", "")
    if verify_signature(MP_TOKEN, timestamp, nonce, signature):
        log("公众号 URL 验证通过")
        return echostr, 200, {"Content-Type": "text/plain"}
    log("公众号 URL 验证失败：签名不匹配")
    return "Forbidden", 403


@app.route("/mp/callback", methods=["POST"])
def mp_callback():
    """公众号消息回调：只处理文本消息，其他类型回空串（微信不再重试）。"""
    signature = request.args.get("signature", "")
    if signature:
        if not verify_signature(MP_TOKEN, request.args.get("timestamp", ""),
                                request.args.get("nonce", ""), signature):
            log("公众号消息验签失败")
            return "Forbidden", 403
    raw = request.get_data(as_text=True)
    try:
        msg = parse_mp_message(raw)
    except ValueError as e:
        log("消息 XML 解析失败：", str(e)[:200])
        return "", 200
    if msg["msg_type"] != "text":
        log("忽略非文本消息：", msg["msg_type"])
        return "", 200
    question = msg["content"]
    log("收到提问 user=", _user_hash(msg["from_user"]), "len=", len(question))
    if is_help_command(question):
        answer = HELP_TEXT
    elif not question:
        answer = HELP_TEXT
    else:
        answer = answer_question(question, INDEX)
    return build_text_reply(msg["from_user"], msg["to_user"], answer), 200, \
        {"Content-Type": "application/xml; charset=utf-8"}


@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    """重建索引：新增/修改 knowledge/ 下的 .md 后调这个接口，不用重启容器。"""
    if not ADMIN_TOKEN or request.headers.get("X-Admin-Token", "") != ADMIN_TOKEN:
        return {"error": "unauthorized"}, 401
    docs, chunks = INDEX.reload()
    log("索引已重建：文档", docs, "块", chunks)
    return {"status": "ok", "docs": docs, "chunks": chunks}


if __name__ == "__main__":
    log(f"公众号 RAG 机器人 v{VERSION} 启动：监听 0.0.0.0:{PORT}，"
        f"知识库 {INDEX.stats()}，LLM={LLM_MODEL}@{LLM_API_BASE}，"
        f"超时 {LLM_TIMEOUT_SEC}s，上限 {LLM_MAX_TOKENS} tokens，"
        f"MP_TOKEN={'已配置' if MP_TOKEN else '未配置'}，"
        f"LLM_API_KEY={'已配置' if LLM_API_KEY else '未配置'}")
    app.run(host="0.0.0.0", port=PORT)
