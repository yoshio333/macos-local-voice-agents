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
    BotStartedSpeakingFrame,
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
あなたはノブさんの「書紀AI」。音声で放り込まれる思考の断片やメモを受け取り、短く整理して記録するのが仕事。名前はパレドンだが、聞かれない限り名乗らない。
自称は「パレドン」または「僕」。「俺」「私」は使いません。

入力はノブさんの音声をリアルタイム文字起こししたテキストです。誤認識が混じることがあるので、文脈で柔軟に解釈してください。

出力は音声合成で読み上げられます。記号やマークダウンは使わないでください。

【口調】
普通のカジュアルな話し方。フランクで対等（「〜だね」「〜だよ」「〜しとくね」「なるほど」「了解」）。落ち着いて自然に。
キャラ付け・軽口・冗談・茶々・方言は不要。素っ気なくはしないが、盛らない。

【役割】
あなたは壁打ち相手ではありません。長い講釈・分析・お説教はしない。ノブさんの言葉を受け止めて記録するのが仕事。ただし無味乾燥にはならず、軽い相づちや一言の反応は添えてOK。

【応答の長さ】
原則1〜2文。短く。
- 受け止めるときも「了解」だけで終わらせず、一言反応を添える程度
- 聞き取りが曖昧なら短い確認を1つ
- 自分の見立てを長々語らない

【知らないこと・検索】
あなたは検索できません。事実確認・最新情報・固有名詞の正確さが要る話を振られたら、知ったかぶりせず「それはスマホのAIに聞いて」と正直に返す。

【最初の挨拶】
会話履歴が空のときの最初だけ、ごく短く一言（聞こえてるかの確認程度）で相手の反応を待つ。名乗らない・自己紹介しない・キャラを出さない。「繋がったよ、聞こえてる？」程度でよい。言い回しは毎回少し変える。それ以降は挨拶しない。

【function-calling】
次の3つの function は、該当する依頼が来たら必ず呼ぶこと。返事だけで済ませない。
1. save_memo: ノブさんが「メモして」「記録して」「これメモ」等と明示したとき、または明らかに記録すべき着想を述べたとき呼ぶ。本文はノブさんの言葉に近い形で保存。呼んだら「メモしたよ」と一言だけ。
2. add_todo: ノブさんが「TODO」「タスク」「やること」「忘れずに◯◯」と明示したとき呼ぶ。「了解、追加したよ」と一言だけ。
3. control_appliance: 家電操作・赤外線リモコンの学習・発光テストの依頼で必ず呼ぶ。
   - 「赤外線テスト」「IRテスト」「発光テスト」等 → action="test"（target不要）
   - 「◯◯のリモコン覚えて」「リモコン登録して」等 → action="learn"、target=家電の短い名前
   - 「台所つけて」「テレビ消して」等の家電操作 → action="send"、target=家電の短い名前
   learn を呼んだあとは「リモコンをデバイスに向けてボタンを押して」と促すこと。
これら以外は普通の chat として、短く応答してください。
"""


class DeviceUIState(FrameProcessor):
    """Pushes ui_state JSON to the StickS3 as the conversation progresses.

    Maps pipeline frames to the device's screen states:
      - UserStoppedSpeakingFrame  -> "thinking" (Now Loading screen)
      - BotStartedSpeakingFrame   -> "speaking" (face + mouth animation) as
        soon as audio playback begins, with the text gathered so far
      - LLMFullResponseEndFrame   -> "speaking" refreshed with the full text
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
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Audio playback is starting -> switch the device to the speaking
            # face (mouth animates) right away, with whatever text we have.
            await self._send_ui_state("speaking", self._llm_text.strip())
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Full utterance text is complete -> refresh it on screen (the
            # face is already speaking; this just updates the text).
            text = self._llm_text.strip()
            if text:
                await self._send_ui_state("speaking", text)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._send_ui_state("ready")

        await self.push_frame(frame, direction)


async def run_bot(transport, webrtc_connection=None, serializer=None):
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

    control_appliance_schema = FunctionSchema(
        name="control_appliance",
        description="赤外線リモコンで家電を操作する、または新しいリモコンを学習する。家電の点灯・消灯などの操作、リモコンの登録/学習を頼まれたとき呼ぶ。",
        properties={
            "action": {
                "type": "string",
                "enum": ["send", "learn", "test"],
                "description": "send=学習済みの家電を操作 / learn=新しいリモコンを学習 / test=赤外線LEDの発光テスト",
            },
            "target": {
                "type": "string",
                "description": "家電の名前。短く（例: 台所, テレビ, エアコン, 寝室）。test のときは任意",
            },
        },
        required=["action"],
    )

    # web_research (SearXNG) was dropped: this device is an input/capture
    # companion, not a research tool — factual lookups go to a phone AI.
    tools = ToolsSchema(
        standard_tools=[save_memo_schema, add_todo_schema, control_appliance_schema]
    )

    # Set when a StickS3 connects (ws mode); the IR commands are pushed
    # straight down this socket, bypassing the audio pipeline.
    device_ws = None

    async def handle_save_memo(params: FunctionCallParams):
        content = params.arguments.get("content", "")
        title = params.arguments.get("title")
        m = session_logger.add_memo(content, title)
        logger.info(f"💭 memo saved: {title or '(無題)'}")
        await params.result_callback({
            "saved": True,
            "note": "メモを記録した。「メモしたよ」と伝えつつ、内容に一言反応を添えて返してください"
        })

    async def handle_add_todo(params: FunctionCallParams):
        content = params.arguments.get("content", "")
        session_logger.add_todo(content)
        logger.info(f"✅ todo added: {content}")
        await params.result_callback({"saved": True, "note": "TODO に追加しました"})

    async def handle_control_appliance(params: FunctionCallParams):
        action = params.arguments.get("action", "")
        target = params.arguments.get("target", "")
        if device_ws is None:
            await params.result_callback(
                {"ok": False, "note": "デバイスが繋がっていない。繋がっていないと正直に伝えて"}
            )
            return
        if action not in ("send", "learn", "test"):
            await params.result_callback(
                {"ok": False, "note": f"未対応の操作 ({action})"}
            )
            return
        try:
            await device_ws.send(
                json.dumps({"event": "ir", "action": action, "name": target})
            )
        except Exception as e:
            logger.error(f"ir command send failed: {e}")
            await params.result_callback(
                {"ok": False, "note": "デバイスへの送信に失敗した"}
            )
            return
        if action == "learn":
            logger.info(f"📡 ir learn requested: {target}")
            await params.result_callback({
                "ok": True,
                "note": f"「{target}」の学習モードに入った。リモコンをデバイスに向けてボタンを押すよう、ノブさんに促して",
            })
        elif action == "test":
            logger.info("📡 ir test (NEC bring-up)")
            await params.result_callback({
                "ok": True,
                "note": "赤外線の発光テスト信号を送った。スマホのカメラでデバイス上部のIR LEDが光るか見るよう、ノブさんに伝えて",
            })
        else:
            logger.info(f"📡 ir send: {target}")
            await params.result_callback({
                "ok": True,
                "note": f"「{target}」を操作する信号を送った。一言で伝えて",
            })

    llm.register_function("save_memo", handle_save_memo)
    llm.register_function("add_todo", handle_add_todo)
    llm.register_function("control_appliance", handle_control_appliance)

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

    # Wire the serializer's device-event callback (StickS3 ws mode only).
    # ir_result frames arrive out-of-turn; turn them into a spoken note by
    # appending a message and kicking the LLM.
    if serializer is not None:
        async def handle_device_event(msg):
            if msg.get("event") != "ir_result":
                logger.debug(f"unhandled device event: {msg}")
                return
            action = msg.get("action", "")
            name = msg.get("name", "")
            ok = bool(msg.get("ok"))
            logger.info(f"📡 ir_result action={action} name={name} ok={ok}")
            # Speak the outcome: learn always, send only when it failed
            # (a successful send was already acknowledged at call time).
            if action == "learn":
                note = (
                    f"赤外線リモコン「{name}」の学習に成功した"
                    if ok
                    else f"赤外線リモコン「{name}」の学習に失敗した（コードを捕捉できなかった）"
                )
            elif action == "send" and not ok:
                note = f"家電「{name}」の操作に失敗した（まだ学習されていないかもしれない）"
            else:
                return
            context.add_message(
                {"role": "user", "content": f"（システム通知：{note}。ノブさんに一言で伝えて）"}
            )
            await task.queue_frames(
                [context_aggregator.user().get_context_frame()]
            )

        serializer.on_control = handle_device_event

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
            nonlocal device_ws
            device_ws = websocket
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
            nonlocal session_logger, device_ws
            device_ws = None
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
    serializer = OpusFrameSerializer()
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=6,
            add_wav_header=False,
            serializer=serializer,
        ),
        host=host,
        port=port,
    )
    await run_bot(transport, serializer=serializer)


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
