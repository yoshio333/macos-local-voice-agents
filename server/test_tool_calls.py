"""A.7 デバッグ：Pipecat OpenAILLMService 経由で Qwen3.6 に tool_calls させる最小テスト。

直接 curl が通って Pipecat で通らない問題の切り分け：
- このスクリプトが tool_calls を出力 → bot.py 側の context/handler 接続を疑う
- このスクリプトも text のみ → Pipecat の tools 送信ロジックに問題
"""
import asyncio
import json
import os
import sys

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


async def main():
    llm = OpenAILLMService(
        api_key="dummyKey",
        model="qwen3.6-35b-a3b-ud-mlx",
        base_url="http://127.0.0.1:1234/v1",
        max_tokens=512,
    )

    save_memo_schema = FunctionSchema(
        name="save_memo",
        description="ノブさんの気づき・着想・思考を memo として記録する。「メモして」と言われたら必ず呼ぶ。",
        properties={
            "title": {"type": "string", "description": "短い見出し"},
            "content": {"type": "string", "description": "メモ本文"},
        },
        required=["content"],
    )
    add_todo_schema = FunctionSchema(
        name="add_todo",
        description="TODO 追加。「TODOに」「タスク追加」と言われたら必ず呼ぶ。",
        properties={"content": {"type": "string"}},
        required=["content"],
    )
    web_research_schema = FunctionSchema(
        name="web_research",
        description="Web 検索。「調べて」と言われたら必ず呼ぶ。",
        properties={"query": {"type": "string"}},
        required=["query"],
    )
    tools = ToolsSchema(standard_tools=[save_memo_schema, add_todo_schema, web_research_schema])

    context = OpenAILLMContext(
        [
            {"role": "system", "content": "あなたは音声アシスタント。「メモして」「TODOに」「調べて」のいずれかがあればツールを必ず呼ぶ。"},
            {"role": "user", "content": "メモして：今日の気づきは、子育て中のリズム調整がプロジェクト進行と似ているということ。"},
        ],
        tools=tools,
        tool_choice="auto",
    )
    # adapter 設定（OpenAILLMService が context にやってくれる手順を手動で）
    context.set_llm_adapter(llm.get_llm_adapter())

    # build_chat_completion_params が実際に送るパラメータを覗く
    messages = context.get_messages()
    params = llm.build_chat_completion_params(context, messages)
    print("=== params sent to LM Studio ===")
    pretty = {
        k: (v if k != "messages" else f"<{len(v)} msgs>")
        for k, v in params.items()
        if v is not None
    }
    # tools をJSON化して詳細表示
    if "tools" in pretty and pretty["tools"]:
        print("tools:", json.dumps([t for t in pretty["tools"]], ensure_ascii=False, default=str, indent=2)[:1000])
    print("tool_choice:", pretty.get("tool_choice"))
    print("model:", pretty.get("model"))
    print("stream:", pretty.get("stream"))

    # 実際にストリーミングして chunks を確認
    print("\n=== streaming chunks ===")
    text_buf = []
    tool_calls_seen = []
    chunks = await llm.get_chat_completions(context, messages)
    async for chunk in chunks:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if delta.content:
            text_buf.append(delta.content)
        if delta.tool_calls:
            tool_calls_seen.append(delta.tool_calls)
    print("TEXT:", "".join(text_buf)[:300])
    print("TOOL_CALLS seen:", len(tool_calls_seen))
    if tool_calls_seen:
        print("first:", tool_calls_seen[0])


if __name__ == "__main__":
    asyncio.run(main())
