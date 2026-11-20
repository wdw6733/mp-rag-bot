#!/usr/bin/env python3
"""公众号轻量 RAG 机器人 v1.3.1。

服务号（未微信认证）发文字消息 -> 微信 POST 到 /mp/callback ->
检索 knowledge/ 里的 Markdown 作参考 -> 调用户自配的大模型综合回答 ->
5 秒内用被动回复 XML 把答案发回去。

回答模式（v1.1 起）：大模型是主力，知识库只是参考/ grounding，不是门槛。
检索到相关摘录就把摘录当参考发给大模型；检索不到也照常让大模型
用通用知识回答，绝不回“知识库里没有”。只有大模型不可用（未配 Key /
超时 / 报错）时才退回知识库原文摘录；知识库也没有相关内容时，
回一句简短的“大模型暂时不可用，请稍后再问”。

模型自动选择（v1.2）：LLM_MODEL 留空 = 自动模式（默认）。启动和每次
需要时调 GET {LLM_API_BASE}/models 发现网关上实际有的模型，按
快/便宜优先顺序自动挑一个；不同网关（AnyRouter / muse2api / wb）
的模型名不一样，硬编码一个名字会在别的网关上 400 报错。已配模型
优先用，400 模型不存在时自动换下一个。LLM_TIMEOUT_SEC 是全部
尝试的总预算，不是每次尝试的超时。

异步查询（v1.3）：大模型慢（muse2api 实测经常超过 4 秒）时，不能傻等，
因为微信被动回复必须在 5 秒内到，超时粉丝什么都收不到。所以问答分
两步：先在后台线程调大模型，主请求只等 LLM_FIRST_WAIT_SEC（默认 4 秒）；
等到了直接回答案，等不到先回一句“请稍等，查询中。。。”，后台继续跑到
LLM_TIMEOUT_SEC（默认 20 秒），结果缓存在内存里，粉丝再发“继续”或重发
同样的问题就能拿到答案。未认证的号没有客服消息，机器人不能主动再推
一条消息，只能等粉丝再发一条消息来取结果——这是这个号能做到的上限，
认证（300 元/年）后才能用客服消息直接把答案推过去。

未认证服务号的限制（v1.1 不做、也做不了）：
- 只有 5 秒内被动回复，没有 48 小时客服消息；
- 服务器配置启用后，自定义菜单不可用。
认证（300 元/年）后才能用客服消息突破 5 秒、才能用自定义菜单，
那是可选的后续升级，不是 v1.3 的必需项。

凭证与模型只从环境变量读，绝不写进代码、README、日志。
模型不固定：LLM_API_BASE / LLM_API_KEY / LLM_MODEL 由用户在 .env 里
自由配置，改完 docker compose restart 即可，不用改代码、不用重新打镜像。
日志只记问题长度、命中来源和模型名，不记完整 OpenID（只记前 8 位哈希），
绝不记 Key / Authorization 头。
"""
import hashlib
import os
import threading
import time
import xml.etree.ElementTree as ET

import requests
from flask import Flask, request

from retrieval import KnowledgeIndex

VERSION = "1.3.1"
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
# 模型不固定：LLM_MODEL 留空 = 自动模式（默认），见 discover_models()。
# 填了就优先用填的，400 模型不存在时自动换下一个。
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
# 快/便宜优先（为 5 秒被动回复红线）：自动模式按这个顺序挑。
PREFERRED_MODELS = [
    "deepseek-chat", "qwen-turbo", "qwen-plus", "gpt-4o-mini",
    "gpt-4o", "muse-spark", "deepseek-reasoner",
]
_SKIP_SUBSTRINGS = ("embed", "tts", "whisper", "image", "video", "audio")
_MODEL_CACHE_TTL_SEC = 600  # 模型列表缓存 10 分钟
# 内存缓存：{"models": [id, ...], "fetched_at": monotonic 秒}
_MODEL_CACHE = {"models": [], "fetched_at": 0.0}
# 自动模式下最近一次挑中的模型（health / 启动日志用；不做网络调用）。
_AUTO_MODEL = None


def _int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


def _float_env(name, default):
    try:
        return float(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


# v1.3 两个超时，分工不同，不要混淆：
# - LLM_TIMEOUT_SEC：后台大模型调用的总预算，默认 20 秒。20 秒还没回
#   就改用摘录兜底并缓存，粉丝发“继续”拿到的是兜底结果。
# - LLM_FIRST_WAIT_SEC：主请求等后台的第一段等待，默认 4 秒。微信被动
#   回复必须在 5 秒内到，所以这个值绝不能超过 4.5 秒（代码里会截断），
#   否则等不到答案、占位提示也发不出去，粉丝什么都收不到。
LLM_TIMEOUT_SEC = _int_env("LLM_TIMEOUT_SEC", 20)
LLM_FIRST_WAIT_SEC = _float_env("LLM_FIRST_WAIT_SEC", 4)
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
    "模型可在 .env 里自由配置，不用改代码；留空则自动选一个快的。"
)
LLM_UNAVAILABLE_TEXT = "大模型暂时不可用，请稍后再问。"
HELP_COMMANDS = {"帮助", "help", "菜单", "menu", "/help", "/menu", "你好", "hi", "hello"}

# v1.3 异步查询：占位提示与“继续”取结果。
# 占位提示（v1.3.1）：第一行固定为用户要的“请稍等，查询中。。。”，
# 第二行只留一句取结果的提示，两行，不写秒数。
STILL_WAITING_TEXT = "还在查询，请再等一下，稍后再发“继续”。"
CONTINUE_COMMANDS = {"继续", "继续查询", "好了吗", "好了么", "结果", "查看结果", "继续查"}
# 每个用户同时只保留一个查询任务：{user_hash: job}，内存保存，不写文件、
# 不建数据库。Flask 开发服务器是单进程，重启后缓存清空——这是已知限制，
# 见 README。job = {"question", "question_norm", "status", "answer",
# "created_at" (monotonic 秒), "done_event" (threading.Event), "thread"}。
_JOB_TTL_SEC = 600  # 查询结果保留 10 分钟，过期就丢掉
_JOBS = {}
_JOBS_LOCK = threading.Lock()


def _placeholder_text():
    return "请稍等，查询中。。。\n（稍后发“继续”查看结果）"


def _first_wait_sec():
    """首答等待秒数：配置值，但硬上限 4.5 秒——微信 5 秒红线，超了占位
    提示本身都发不出去。测试里可把 LLM_FIRST_WAIT_SEC 设得很小。"""
    try:
        v = float(LLM_FIRST_WAIT_SEC)
    except (TypeError, ValueError):
        v = 4.0
    return max(0.0, min(v, 4.5))


def _normalize_question(text):
    """缓存键用的问题归一化：去首尾空白、压掉中间多余空白、小写。"""
    return " ".join((text or "").strip().lower().split())


def _is_continue_command(text):
    """是不是取结果的命令：去掉尾巴上的标点再比，避免“继续。”认不出。"""
    t = (text or "").strip().lower().strip("。？?！!、，, ")
    return t in CONTINUE_COMMANDS


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


def _is_skip_model(model_id):
    low = (model_id or "").lower()
    return any(s in low for s in _SKIP_SUBSTRINGS)


def _ordered_auto_models(models):
    """自动模式的候选顺序：优先名单在先，其次非 embedding/音视频模型，
    最后才是 embedding/音视频模型（总比没有强）。大小写不敏感去重。"""
    by_lower = {}
    for m in models or []:
        if m and str(m).strip():
            by_lower.setdefault(str(m).strip().lower(), str(m).strip())
    ordered = []
    for pref in PREFERRED_MODELS:
        if pref in by_lower:
            ordered.append(by_lower[pref])
    for m in models or []:
        m = str(m).strip()
        if m and m not in ordered and not _is_skip_model(m):
            ordered.append(m)
    for m in models or []:
        m = str(m).strip()
        if m and m not in ordered:
            ordered.append(m)
    return ordered


def _parse_model_list(payload):
    if isinstance(payload, dict):
        data = payload.get("data", [])
    elif isinstance(payload, list):
        data = payload
    else:
        return []
    ids = []
    for item in data or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]).strip())
        elif isinstance(item, str) and item.strip():
            ids.append(item.strip())
    return [i for i in ids if i]


def discover_models(force=False):
    """GET {base}/models 发现网关上实际有的模型，缓存 10 分钟。

    失败或为空时返回之前缓存的列表（可能为空），绝不抛异常、不挂起：
    调用方会用优先名单盲试或走摘录兜底。只记模型名，不记 Key。
    """
    global _AUTO_MODEL
    now = time.monotonic()
    if not force and _MODEL_CACHE["models"] and now - _MODEL_CACHE["fetched_at"] < _MODEL_CACHE_TTL_SEC:
        return list(_MODEL_CACHE["models"])
    if not LLM_API_KEY:
        return list(_MODEL_CACHE["models"])
    url = LLM_API_BASE.rstrip("/") + "/models"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=3,
        )
        resp.raise_for_status()
        models = _parse_model_list(resp.json())
    except Exception as e:
        log("模型发现失败：", type(e).__name__, str(e)[:200])
        return list(_MODEL_CACHE["models"])
    if models:
        _MODEL_CACHE["models"] = models
        _MODEL_CACHE["fetched_at"] = now
        picked = _ordered_auto_models(models)
        if picked:
            _AUTO_MODEL = picked[0]
        log("模型发现成功：", len(models), "个，自动选用", _AUTO_MODEL)
    return list(_MODEL_CACHE["models"])


def pick_auto_model(models):
    """从已发现的列表里挑一个：优先名单在先，否则第一个非 embedding/
    音视频模型，都没有则取第一个。无列表返回 None。"""
    ordered = _ordered_auto_models(models)
    return ordered[0] if ordered else None


def _model_not_found_text(text):
    low = (text or "").lower()
    if "model" not in low:
        return False
    return ("not found" in low or "unavailable" in low
            or "11102" in low or "service info not found" in low)


def _error_body(exc):
    resp = getattr(exc, "response", None)
    if resp is None:
        return str(exc)
    text = getattr(resp, "text", "") or ""
    if text:
        return text
    try:
        return str(resp.json())
    except Exception:
        return str(exc)


def _llm_startup_info():
    """启动日志用的模型信息：(mode, model_display, models_info)。

    fixed: LLM=<已配模型>@<base>；auto: LLM=auto(<选中>)@<base>，
    models=<n> discovered 或 models=discovery failed。
    """
    if LLM_MODEL:
        return "fixed", LLM_MODEL, None
    models = discover_models()
    if models:
        picked = pick_auto_model(models)
        return "auto", picked or "auto:unknown", f"models={len(models)} discovered"
    return "auto", "auto:unknown", "models=discovery failed"


def call_llm(question, chunks):
    """调 OpenAI 兼容接口。成功返回回答文本，失败抛异常（调用方兜底）。

    chunks 可以为空：有摘录时当参考发给大模型，没摘录时大模型直接
    用通用知识回答。模型、地址、超时、token 上限全部来自环境变量。
    自动模式：LLM_MODEL 留空时先发现模型再按优先顺序挑；已配模型
    优先用，400 模型不存在时自动换下一个。LLM_TIMEOUT_SEC 是全部
    尝试的总预算，不是每次尝试的超时；每次尝试用剩余时间（至少 1 秒，
    但总耗时不会无限制拖下去——超过预算即停）。
    """
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY 未配置")
    deadline = time.monotonic() + LLM_TIMEOUT_SEC
    if chunks:
        context = "\n\n---\n\n".join(
            f"【{c['source']} / {c['heading']}】\n{c['text']}" for c in chunks
        )
        user_content = f"知识库摘录（仅供参考）：\n{context}\n\n问题：{question}"
    else:
        user_content = f"问题：{question}"
    candidates = []
    if LLM_MODEL:
        candidates.append(LLM_MODEL)
    else:
        models = discover_models()
        candidates = _ordered_auto_models(models) or list(PREFERRED_MODELS)
    tried = []
    last_error = RuntimeError("没有可用的模型")
    for model in candidates:
        if model in tried:
            continue
        tried.append(model)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        attempt_timeout = remaining if remaining >= 1 else 1.0
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": LLM_MAX_TOKENS,
        }
        url = LLM_API_BASE.rstrip("/") + "/chat/completions"
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {LLM_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=attempt_timeout,
            )
            status = getattr(resp, "status_code", None)
            if status is not None and status >= 400:
                body = getattr(resp, "text", "") or ""
                if status == 400 and _model_not_found_text(body):
                    log("模型不存在，换下一个：", model, body[:200])
                    last_error = RuntimeError(f"模型 {model} 不存在")
                    _extend_candidates(candidates, tried)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            body = _error_body(e)
            if _model_not_found_text(body):
                log("模型不存在，换下一个：", model, body[:200])
                last_error = RuntimeError(f"模型 {model} 不存在：{body[:200]}")
                _extend_candidates(candidates, tried)
                continue
            raise
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            raise
        except Exception:
            raise
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
    raise last_error


def _extend_candidates(candidates, tried):
    """400 模型不存在后刷新发现列表，把没试过的新模型追加到候选队尾。"""
    try:
        models = discover_models(force=True)
    except Exception:
        return
    for m in _ordered_auto_models(models):
        if m not in tried and m not in candidates:
            candidates.append(m)


def _prune_jobs_locked():
    """删掉过期的任务（调用方必须已持有 _JOBS_LOCK）。"""
    now = time.monotonic()
    for key in [k for k, j in _JOBS.items()
                if now - j["created_at"] > _JOB_TTL_SEC]:
        del _JOBS[key]


def _get_job(user_hash):
    """取某用户当前任务；过期或没有返回 None。线程安全。"""
    with _JOBS_LOCK:
        _prune_jobs_locked()
        return _JOBS.get(user_hash)


def _clear_job(user_hash, job):
    """结果取走后删掉任务（只删“还是这个任务”时才删，避免误删新任务）。"""
    with _JOBS_LOCK:
        if _JOBS.get(user_hash) is job:
            del _JOBS[user_hash]


def _reset_jobs():
    """清空全部任务：测试用，生产代码不调。"""
    with _JOBS_LOCK:
        _JOBS.clear()


def _start_job(user_hash, question, question_norm):
    job = {
        "question": question,
        "question_norm": question_norm,
        "status": "running",
        "answer": None,
        "created_at": time.monotonic(),
        "done_event": threading.Event(),
        "thread": None,
    }
    with _JOBS_LOCK:
        _prune_jobs_locked()
        _JOBS[user_hash] = job
    return job


def _run_job(job, user_hash, question, chunks):
    """后台线程：调大模型，成功就缓存答案，失败/超时就缓存摘录兜底。

    结果只在“这个任务还是该用户的当前任务”时才算已缓存；中途用户问了
    新问题、任务被取代了，算完也丢弃（但照样置 done_event，让还在首答
    等待里的那次请求能拿到答案，而不是干等到占位超时）。
    """
    try:
        answer = call_llm(question, chunks)
    except requests.exceptions.Timeout as e:
        log("异步查询超时，改用兜底：", type(e).__name__, str(e)[:200])
        answer = extractive_fallback(chunks)
    except Exception as e:
        log("LLM 调用失败，改用兜底：", type(e).__name__, str(e)[:200])
        answer = extractive_fallback(chunks)
    with _JOBS_LOCK:
        job["status"] = "done"
        job["answer"] = answer
        still_current = _JOBS.get(user_hash) is job
    job["done_event"].set()
    if still_current:
        log("异步结果已缓存 user=", user_hash, "len=", len(answer))
    else:
        log("异步结果已丢弃（被新问题取代）user=", user_hash)


def answer_question(question, index, user_hash="anon"):
    """问答主流程（v1.3 异步）：检索作参考 -> 后台调大模型。

    快路径：大模型在 LLM_FIRST_WAIT_SEC（默认 4 秒）内回完，直接回答案，
    和 v1.2 一样；任务随即清除，之后再发“继续”只会回帮助。
    慢路径：4 秒没回完，先回占位提示“请稍等，查询中。。。”，后台继续跑到
    LLM_TIMEOUT_SEC（默认 20 秒）；结果缓存 10 分钟，粉丝发“继续”或重发
    同样的问题来取。重发同样的问题、且任务还在跑时不重复开任务。
    “继续”类命令：有已完成的任务就取走答案并清除；还在跑就回“还在查询”；
    没有任务就回帮助。检索不到不拦路，失败兜底逻辑与 v1.2 相同。
    """
    if _is_continue_command(question):
        job = _get_job(user_hash)
        if job is None:
            return HELP_TEXT
        if job["status"] == "running":
            return STILL_WAITING_TEXT
        answer = job["answer"]
        _clear_job(user_hash, job)
        return answer
    norm = _normalize_question(question)
    job = _get_job(user_hash)
    if job is not None and job["question_norm"] == norm:
        if job["status"] == "running":
            return STILL_WAITING_TEXT
        answer = job["answer"]
        _clear_job(user_hash, job)
        return answer
    chunks = index.search(question, top_n=3)
    log("提问 len=", len(question), "命中：",
        ", ".join(f"{c['source']}#{c['heading']}" for c in chunks) or "无")
    if job is not None:
        log("新问题取代旧查询 user=", user_hash)
    job = _start_job(user_hash, question, norm)
    thread = threading.Thread(
        target=_run_job, args=(job, user_hash, question, chunks),
        daemon=True)
    job["thread"] = thread
    thread.start()
    if job["done_event"].wait(timeout=_first_wait_sec()):
        answer = job["answer"]
        _clear_job(user_hash, job)
        return answer
    log("提问 len=", len(question), "异步查询中 user=", user_hash)
    return _placeholder_text()


app = Flask(__name__)
INDEX = KnowledgeIndex(KNOWLEDGE_DIR)


@app.route("/health", methods=["GET"])
def health():
    stats = INDEX.stats()
    if LLM_MODEL:
        llm_mode, llm_model = "fixed", LLM_MODEL
    else:
        llm_mode, llm_model = "auto", _AUTO_MODEL or "auto:unknown"
    with _JOBS_LOCK:
        _prune_jobs_locked()
        active_jobs = len(_JOBS)
    return {"status": "ok", "version": VERSION, **stats,
            "llm_mode": llm_mode, "llm_model": llm_model,
            "llm_timeout_sec": LLM_TIMEOUT_SEC,
            "llm_first_wait_sec": _first_wait_sec(),
            "async_jobs": active_jobs}


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
    user_hash = _user_hash(msg["from_user"])
    log("收到提问 user=", user_hash, "len=", len(question))
    if is_help_command(question):
        answer = HELP_TEXT
    elif not question:
        answer = HELP_TEXT
    else:
        answer = answer_question(question, INDEX, user_hash=user_hash)
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
    _mode, _model, _models_info = _llm_startup_info()
    if _mode == "fixed":
        _llm_log = f"LLM={_model}@{LLM_API_BASE}"
    else:
        _llm_log = (f"LLM=auto({_model})@{LLM_API_BASE}，"
                    f"{_models_info or 'models=discovery failed'}")
    log(f"公众号 RAG 机器人 v{VERSION} 启动：监听 0.0.0.0:{PORT}，"
        f"知识库 {INDEX.stats()}，{_llm_log}，"
        f"首答等待 {_first_wait_sec()}s，总预算 {LLM_TIMEOUT_SEC}s，"
        f"上限 {LLM_MAX_TOKENS} tokens，"
        f"MP_TOKEN={'已配置' if MP_TOKEN else '未配置'}，"
        f"LLM_API_KEY={'已配置' if LLM_API_KEY else '未配置'}")
    app.run(host="0.0.0.0", port=PORT)
