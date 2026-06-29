# .codex/agents/ — Codex subagents

Codex subagents are defined as **`.toml`** files here (one per agent). They are the Codex
equivalent of Claude's `.md` agents in `.claude/agents/`.

**There are currently no project agents to convert** — `.claude/agents/` is empty. This
directory and README exist so the convention is ready the moment one is added.

## Converting a Claude agent → Codex agent

A Claude agent (`.claude/agents/<name>.md`) has YAML frontmatter (`name`, `description`,
`tools`, `model`) followed by a markdown instruction body. To mirror it for Codex, create
`.codex/agents/<name>.toml` and move the body into `developer_instructions`:

```toml
# .codex/agents/<name>.toml
name = "<name>"
description = "<one-line role guidance, from the Claude agent's description>"

# The Claude agent's markdown body goes here verbatim.
developer_instructions = """
<the full instruction body from .claude/agents/<name>.md>
"""

# Optional, if the Claude agent pinned them:
# model = "gpt-5.5"
# tools = ["..."]   # map Claude tool names to Codex equivalents
```

Keep the two in sync: when a `.claude/agents/*.md` agent changes, update its
`.codex/agents/*.toml` counterpart (see the "Codex adapter & sync contract" in `CLAUDE.md`).
