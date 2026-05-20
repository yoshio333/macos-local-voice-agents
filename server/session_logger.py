"""
mini パレドンの音声セッション記録 + ガチパレドンへの引き継ぎファイル生成

役割：
- run_bot 起動ごとに 1セッションを開始
- memo / todo / research の発生を蓄積
- finalize() で：
  - 09_voice-sessions/raw/<日時>.json に生 messages をダンプ
  - 00_inbox/voice-sessions/<日時>.md にガチパレドン処理用ファイルを書き出し
"""
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/Users/nobu-ai-agent/claude-workspace")
INBOX_DIR = WORKSPACE_ROOT / "00_inbox" / "voice-sessions"
RAW_DIR = WORKSPACE_ROOT / "09_voice-sessions" / "raw"


class SessionLogger:
    def __init__(self):
        now = datetime.now()
        self.session_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.start_time = now
        self.memos: List[Dict[str, Any]] = []
        self.todos: List[Dict[str, Any]] = []
        self.researches: List[Dict[str, Any]] = []
        self.finalized = False

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def add_memo(self, content: str, title: Optional[str] = None) -> Dict:
        m = {"ts": self._now(), "content": content, "title": title}
        self.memos.append(m)
        return m

    def add_todo(self, content: str) -> Dict:
        t = {"ts": self._now(), "content": content}
        self.todos.append(t)
        return t

    def add_research(self, query: str, summary: str, sources: List[str]) -> Dict:
        r = {"ts": self._now(), "query": query, "summary": summary, "sources": sources}
        self.researches.append(r)
        return r

    def finalize(self, llm_messages: List[Dict[str, Any]]) -> Optional[Path]:
        """セッション終了：raw ダンプ + inbox ファイル生成

        Args:
            llm_messages: OpenAILLMContext.messages の中身

        Returns:
            生成された inbox ファイルパス（実質的に発話が無い場合は None）
        """
        if self.finalized:
            return None
        self.finalized = True

        # user/assistant 発話だけ抽出（SYSTEM_INSTRUCTION は最初の user ロールに含まれている）
        turns: List[Dict[str, Any]] = []
        for m in llm_messages:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str):
                continue
            if role == "user" and content.strip().startswith("あなたの名前はパレドン"):
                continue  # SYSTEM_INSTRUCTION はスキップ
            if role in ("user", "assistant") and content.strip():
                turns.append({"role": role, "content": content.strip()})

        # 中身が無いなら何も書かない（誤接続・即切断対策）
        if not turns and not self.memos and not self.todos and not self.researches:
            return None

        # raw ダンプ
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DIR / f"{self.session_id}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self._now(),
                "llm_messages": llm_messages,
                "memos": self.memos,
                "todos": self.todos,
                "researches": self.researches,
            }, f, ensure_ascii=False, indent=2)

        # inbox ファイル
        INBOX_DIR.mkdir(parents=True, exist_ok=True)
        inbox_path = INBOX_DIR / f"{self.session_id}.md"
        with open(inbox_path, "w", encoding="utf-8") as f:
            f.write(self._format_md(turns))

        # 音声セッションファイルを workspace リポに commit & push
        # （MacBook Air 等の他デバイスから git pull で参照できるように）
        self._git_sync([raw_path, inbox_path])

        return inbox_path

    def _git_sync(self, paths: List[Path]) -> None:
        """書き出した音声セッションファイルだけを workspace リポに commit & push。

        失敗しても bot を落とさない（ネット断・push 競合等は warning ログのみ）。
        push が非 fast-forward で弾かれたら pull --rebase して 1 回だけ再試行。
        commit はローカルに残るので、次回以降の push で carry される。
        """
        def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", str(WORKSPACE_ROOT), *args],
                capture_output=True, text=True, timeout=60, check=check,
            )

        try:
            rel = [str(p.relative_to(WORKSPACE_ROOT)) for p in paths]
            git("add", *rel)
            # ステージに何も無い（既出ファイル等）なら commit しない
            if git("diff", "--cached", "--quiet", check=False).returncode == 0:
                logger.info("git_sync: nothing to commit")
                return
            git("commit", "-m", f"voice-session: {self.session_id}")
            try:
                git("push", "origin", "main")
            except subprocess.CalledProcessError:
                logger.warning("git_sync: push rejected, retrying after rebase")
                git("pull", "--rebase", "origin", "main")
                git("push", "origin", "main")
            logger.info(f"git_sync: pushed voice-session {self.session_id}")
        except Exception as e:
            logger.warning(f"git_sync failed (commit kept locally): {e}")

    def _format_md(self, turns: List[Dict[str, Any]]) -> str:
        lines = [
            f"# voice-session {self.session_id}",
            "",
            f"- 開始: {self.start_time.isoformat(timespec='seconds')}",
            f"- 終了: {self._now()}",
            f"- 生データ: [09_voice-sessions/raw/{self.session_id}.json](../../09_voice-sessions/raw/{self.session_id}.json)",
            "",
            "## ガチパレドンへの引き継ぎ",
            "",
            "このセッションには以下の処理対象がある。手動「voice-session 処理して」で起動するときに参照。",
            "",
        ]

        if self.memos:
            lines.append(f"### 💭 memo ({len(self.memos)}件)")
            lines.append("")
            for i, m in enumerate(self.memos, 1):
                title = m.get("title") or "(タイトル無し)"
                lines.append(f"**[{i}] {title}** — {m['ts']}")
                lines.append("")
                lines.append(m["content"])
                lines.append("")

        if self.todos:
            lines.append(f"### ✅ TODO ({len(self.todos)}件)")
            lines.append("")
            for i, t in enumerate(self.todos, 1):
                lines.append(f"- [ ] {t['content']}  _<{t['ts']}>_")
            lines.append("")

        if self.researches:
            lines.append(f"### 🔍 research ({len(self.researches)}件)")
            lines.append("")
            for i, r in enumerate(self.researches, 1):
                lines.append(f"**[{i}] {r['query']}** — {r['ts']}")
                lines.append("")
                lines.append(r["summary"])
                if r.get("sources"):
                    lines.append("")
                    lines.append("ソース：")
                    for url in r["sources"]:
                        if url:
                            lines.append(f"- {url}")
                lines.append("")

        if not (self.memos or self.todos or self.researches):
            lines.append("（Intent 振り分け対象なし。chat のみのセッション。）")
            lines.append("")

        lines.append("## chat 全文")
        lines.append("")
        for t in turns:
            who = "🧑 ノブさん" if t["role"] == "user" else "🤖 パレドン"
            lines.append(f"**{who}**: {t['content']}")
            lines.append("")

        return "\n".join(lines)
