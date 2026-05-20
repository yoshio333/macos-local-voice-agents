import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.transports.network.websocket_server import (
    WebsocketServerTransport,
    WebsocketServerParams,
)
import functools as _functools
import pipecat.transports.network.websocket_server as _ws_server_mod

# The StickS3 sits idle for long stretches between conversations and does not
# reliably answer the websockets server's keepalive ping, so the default 20s
# ping was killing an otherwise healthy connection. Disable server-side
# keepalive; the device firmware runs its own reconnect logic.
_ws_server_mod.websocket_serve = _functools.partial(
    _ws_server_mod.websocket_serve, ping_interval=None
)
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TransportMessageUrgentFrame,
    UserStoppedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
)

from tts_mlx_isolated import TTSMLXIsolated
from opus_serializer import OpusFrameSerializer

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

import aiohttp
from session_logger import SessionLogger

load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


SYSTEM_INSTRUCTION = """
あなたの名前はパレドン（パレドロス、ギリシャ語「傍らに座る者」）。ノブさんの思考の相棒です。
自称は「パレドン」または「僕」。「俺」「私」は使いません。

入力はノブさんの音声をリアルタイム文字起こししたテキストです。誤認識が混じることがあるので、文脈で柔軟に解釈してください。

出力は音声合成で読み上げられます。記号やマークダウンは使わないでください。

口調はフランクで対等（だね・〜だよ・〜しとく？・〜じゃない？）。むやみに同意せず、自分の見立てを添えます。

応答の尺は内容に応じて変える：
- 挨拶・相槌・雑談 → 1〜2文で短く
- ノブさんが問いや課題を投げてきたら → 3〜5文で本質をついた応答。安易な「うーん」「難しいね」で逃げない
- 知っていることや見立てがあれば、それを前面に出す。知らないことは「知らない」と言う

会話履歴がまだ空のときの最初の挨拶だけ「お、繋がったね。聞こえてる？」と短く言って、相手の反応を待ってください。それ以降は同じセリフを繰り返さず、毎回その場の文脈に応える。

【Intent 判定ルール（function-calling）】

次の3つの function があります。明示プレフィックスがあるとき、または自分の判断で必要と思ったとき呼んでください。

1. save_memo: ノブさんが「メモして」「メモしといて」「記録して」「これメモ」「今の考えメモして」「気づき残して」等と明示したときのみ。明示が無ければ呼ばない（後から「今の話メモして」で遡及救済できる）。
   - 単に保存だけで終わらず、保存後に「もう少し深堀りする？」「関連で◯◯を思い出したけど、それも記録する？」と対話を続けてください
   - 内部知識が不安なときは web_research を組み合わせて、効果的な問いで気づきを深めるアシストをする
   - 「今の話メモしといて」と言われたら、直前の会話を要約して save_memo を呼ぶ

2. add_todo: ノブさんが「TODO」「タスク」「やること追加」「忘れずに◯◯」と明示したとき。
   - 「了解、追加した」と短く確認するだけでOK。深堀りしない

3. web_research: ノブさんが「調べて」「検索して」「最新の◯◯は？」「ファクトチェック」と明示したとき、または事実確認・固有名詞の正確性が必要なとき。
   - あなた自身の内部知識が怪しい固有名詞（バンド名・人物・統計数値等）は内部知識で答えず、必ず web_research を呼ぶ
   - 結果を伝えた後、「補足質問ある？」と続けてください

これら以外の対話は普通の chat として応答してください。
"""


class DeviceUIState(FrameProcessor):
    """Pushes ui_state JSON to the StickS3 as the conversation progresses.

    Maps pipeline frames to the device's screen states:
      - UserStoppedSpeakingFrame  -> "thinking" (Now Loading screen)
      - LLMFullResponseEndFrame   -> "speaking" + the assistant's full text
      - BotStoppedSpeakingFrame   -> "ready"

    The OpusFrameSerializer turns the TransportMessageUrgentFrame into a JSON
    text frame on the wire. The device drives its own "recording" state from
    the button, so that is not sent here.
    """

    def __init__(self):
        super().__init__()
        self._llm_text = ""

    async def _send_ui_state(self, state: str, text: str = ""):
        message = {"event": "ui_state", "state": state}
        if text:
            message["text"] = text
        await self.push_frame(TransportMessageUrgentFrame(message=message))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            await self._send_ui_state("thinking")
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_text = ""
        elif isinstance(frame, LLMTextFrame):
            self._llm_text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = self._llm_text.strip()
            if text:
                await self._send_ui_state("speaking", text)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._send_ui_state("ready")

        await self.push_frame(frame, direction)


async def run_bot(transport, webrtc_connection=None):
    """Build and run the voice pipeline on the given transport.

    transport: a pre-built Pipecat transport (SmallWebRTC for the web client,
    WebsocketServer for the StickS3 device). webrtc_connection is passed only
    in WebRTC mode so we can hook its "closed" event.
    """
    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4, language="ja")

    # tts = TTSMLXIsolated(model="mlx-community/Kokoro-82M-bf16", voice="jf_alpha", sample_rate=24000)
    tts = TTSMLXIsolated(model="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16", voice="eric", language="Japanese", speed=1.4, sample_rate=24000)
    # tts = TTSMLXIsolated(model="Marvis-AI/marvis-tts-250m-v0.1", voice=None)

    llm = OpenAILLMService(
        api_key="dummyKey",
        model="qwen3.6-35b-a3b-ud-mlx",
        base_url="http://127.0.0.1:1234/v1",
        max_tokens=4096,
    )

    # --- セッションロガー & Intent 4分類 (A.7) ---
    session_logger = SessionLogger()
    logger.info(f"Session started: {session_logger.session_id}")

    save_memo_schema = FunctionSchema(
        name="save_memo",
        description="ノブさんの気づき・着想・思考を memo として記録する。明示「メモして」のとき、または記録価値があると判断したとき呼ぶ。",
        properties={
            "title": {"type": "string", "description": "短い見出し（10〜30文字程度）"},
            "content": {"type": "string", "description": "メモ本文。ノブさんが言ったことを要約せず、原文に近い形で保存する"},
        },
        required=["content"],
    )

    add_todo_schema = FunctionSchema(
        name="add_todo",
        description="ノブさんの TODO・タスクを記録する。明示「TODO」「タスク追加」のとき呼ぶ。",
        properties={
            "content": {"type": "string", "description": "タスク内容を端的に1行で"},
        },
        required=["content"],
    )

    web_research_schema = FunctionSchema(
        name="web_research",
        description="ファクトチェックや固有名詞の確認のため Web 検索する。明示「調べて」のとき、または内部知識が不安なとき必ず呼ぶ。",
        properties={
            "query": {"type": "string", "description": "検索クエリ。日本語そのままでOK"},
        },
        required=["query"],
    )

    tools = ToolsSchema(standard_tools=[save_memo_schema, add_todo_schema, web_research_schema])

    async def handle_save_memo(params: FunctionCallParams):
        content = params.arguments.get("content", "")
        title = params.arguments.get("title")
        m = session_logger.add_memo(content, title)
        logger.info(f"💭 memo saved: {title or '(無題)'}")
        await params.result_callback({
            "saved": True,
            "note": "memo を記録しました。深堀り・関連話題があれば続けて聞いてください"
        })

    async def handle_add_todo(params: FunctionCallParams):
        content = params.arguments.get("content", "")
        session_logger.add_todo(content)
        logger.info(f"✅ todo added: {content}")
        await params.result_callback({"saved": True, "note": "TODO に追加しました"})

    async def handle_web_research(params: FunctionCallParams):
        query = params.arguments.get("query", "")
        logger.info(f"🔍 research: {query}")
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "http://127.0.0.1:8888/search",
                    params={"q": query, "format": "json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    data = await r.json()
            results = data.get("results", [])[:5]
            if not results:
                summary = "検索結果なし"
                sources = []
            else:
                summary_lines = []
                sources = []
                for r in results:
                    title = r.get("title", "")
                    content = (r.get("content") or "")[:250]
                    url = r.get("url", "")
                    summary_lines.append(f"- {title}\n  {content}")
                    sources.append(url)
                summary = "\n".join(summary_lines)
            session_logger.add_research(query, summary, sources)
            await params.result_callback({
                "results_summary": summary,
                "sources": sources,
                "note": "結果を伝えたら『補足ある？』と聞いてください",
            })
        except Exception as e:
            logger.error(f"web_research error: {e}")
            await params.result_callback({"error": str(e), "note": "検索に失敗。素直に『SearXNG に届かなかった』と伝えてください"})

    llm.register_function("save_memo", handle_save_memo)
    llm.register_function("add_todo", handle_add_todo)
    llm.register_function("web_research", handle_web_research)

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": SYSTEM_INSTRUCTION,
            }
        ],
        tools=tools,
        tool_choice="auto",
    )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    is_webrtc = webrtc_connection is not None

    #
    # RTVI drives the Pipecat web client UI (handshake / bot-ready). The StickS3
    # device is not an RTVI client, so RTVI is only wired in WebRTC mode.
    #
    processors = [transport.input(), stt]

    rtvi = None
    if is_webrtc:
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
        processors.append(rtvi)

    processors += [context_aggregator.user(), llm]

    # The StickS3 device reflects conversation state on its screen. Placed
    # after the LLM so it can capture the assistant's full utterance text.
    if not is_webrtc:
        processors.append(DeviceUIState())

    processors += [
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]

    pipeline = Pipeline(processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)] if rtvi else [],
        # The WS bot is a long-lived server: the StickS3 sits idle between
        # conversations, so the pipeline must not cancel itself on idle.
        cancel_on_idle_timeout=is_webrtc,
    )

    async def kickoff_conversation():
        # Queue the system prompt so the bot opens with its greeting.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    if is_webrtc:
        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            await kickoff_conversation()

        # on_first_participant_joined / on_participant_left は SmallWebRTCTransport では
        # 未サポート（"not registered" 警告が出る）。
        # 代わりに webrtc_connection の "closed" を直接フックして task.cancel() を呼ぶ。
        # session の finalize は runner の finally で行う。
        @webrtc_connection.event_handler("closed")
        async def on_webrtc_closed(_conn):
            logger.info("WebRTC closed, cancelling pipeline task")
            await task.cancel()
    else:
        # StickS3 (WebsocketServerTransport). The WS server stays up across
        # device reconnects; the pipeline task itself is long-lived.
        @transport.event_handler("on_client_connected")
        async def on_client_connected(_transport, websocket):
            logger.info("StickS3 client connected")
            await websocket.send(
                json.dumps({"event": "interaction_mode", "mode": "hold_to_talk"})
            )
            await websocket.send(
                json.dumps({"event": "ui_state", "state": "ready", "text": ""})
            )
            await kickoff_conversation()

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(_transport, websocket):
            nonlocal session_logger
            logger.info("StickS3 client disconnected")
            # The device has no auto-off/sleep, so the user powers it off at
            # the end of a sitting. Treat a disconnect as the end of a
            # conversation: write the session out now, then start a fresh
            # session and context for the next power-on.
            try:
                inbox_path = session_logger.finalize(context.messages)
                if inbox_path:
                    logger.info(f"📝 session written: {inbox_path}")
                else:
                    logger.info("session had no content, skipping write")
            except Exception as e:
                logger.error(f"session_logger.finalize failed: {e}")
            session_logger = SessionLogger()
            context.set_messages([{"role": "user", "content": SYSTEM_INSTRUCTION}])
            logger.info(f"new session ready: {session_logger.session_id}")

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    finally:
        try:
            inbox_path = session_logger.finalize(context.messages)
            if inbox_path:
                logger.info(f"📝 session written: {inbox_path}")
            else:
                logger.info("session had no content, skipping write")
        except Exception as e:
            logger.error(f"session_logger.finalize failed: {e}")


async def run_webrtc_bot(webrtc_connection):
    """Build a SmallWebRTC transport and run the pipeline (web client path)."""
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            # turn_analyzer disabled 2026-05-18: transformers 5.8 incompat with Pipecat 0.0.81 SmartTurn
            # turn_analyzer=LocalSmartTurnAnalyzerV2(smart_turn_model_path="", params=SmartTurnParams()),
        ),
    )
    await run_bot(transport, webrtc_connection=webrtc_connection)


async def run_ws_bot(host: str, port: int):
    """Build a WebSocket server transport for the StickS3 device and run it.

    PTT turn-taking is driven by the device's button events (no VAD): the
    firmware streams Opus audio only while the primary button is held, and the
    OpusFrameSerializer maps button down/up to User Started/Stopped Speaking.
    Audio is 16kHz mono; 60ms Opus frames (audio_out_10ms_chunks=6).
    """
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=6,
            add_wav_header=False,
            serializer=OpusFrameSerializer(),
        ),
        host=host,
        port=port,
    )
    await run_bot(transport)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_webrtc_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--transport",
        choices=["webrtc", "ws"],
        default="webrtc",
        help="webrtc = Pipecat web client (default), ws = StickS3 device",
    )
    parser.add_argument(
        "--host", default=None, help="Bind host (default: localhost for webrtc, 0.0.0.0 for ws)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Bind port (default: 7860 webrtc, 8765 ws)"
    )
    args = parser.parse_args()

    if args.transport == "ws":
        host = args.host or "0.0.0.0"
        port = args.port or 8765
        logger.info(f"Starting StickS3 WebSocket bot on {host}:{port}")
        asyncio.run(run_ws_bot(host, port))
    else:
        host = args.host or "localhost"
        port = args.port or 7860
        uvicorn.run(app, host=host, port=port)
