# Skills

Agent-facing documentation for this repository's tooling. These live in the repo
so any agent — or any human — can use them, rather than being installed into one
assistant's private configuration.

| Skill | Covers |
| --- | --- |
| [`transcribe/SKILL.md`](transcribe/SKILL.md) | The `bin/transcribe` JSON CLI: actions, envelope, error types, workflow |

## Format

Each skill is a directory containing `SKILL.md`: YAML frontmatter (`name`,
`description`) followed by Markdown. The `description` states what the skill does
and when to reach for it, which is what most agent runtimes match against.

The format is deliberately plain. Nothing in it is specific to a single vendor,
and any agent that can read a file can use it.

## Using it

**Any agent, no setup** — point the agent at the file:

> Read `skills/transcribe/SKILL.md`, then transcribe `input/interview.m4a`.

This always works and needs no integration. The rest of this section is
convenience.

**Claude Code** — discovered automatically from `.claude/skills/`. Link it once
per machine:

```bash
mkdir -p .claude/skills
ln -s ../../skills/transcribe .claude/skills/transcribe
```

Or install it for every project: `ln -s "$PWD/skills/transcribe" ~/.claude/skills/transcribe`

**Cursor / Windsurf / Copilot** — these read a rules file rather than a skill
directory. Reference the skill from it so there is one source of truth:

```bash
mkdir -p .cursor/rules
printf 'For audio transcription, follow @skills/transcribe/SKILL.md\n' \
  > .cursor/rules/transcribe.mdc
```

**Codex / Aider / Claude.md-style agents** — add one line to the repository's
agent instructions file (`AGENTS.md`, `CLAUDE.md`, `CONVENTIONS.md`):

```markdown
For audio transcription, follow `skills/transcribe/SKILL.md`.
```

**MCP or custom harnesses** — load `SKILL.md` as a prompt or resource. The CLI is
already self-describing, so an agent that can run `bin/transcribe list-actions`
can recover the full parameter catalog at runtime without any preloaded context.

## Keeping it honest

The CLI is the source of truth for its own interface. When actions or parameters
change, `bin/transcribe list-actions` reflects it immediately; `SKILL.md` carries
the judgment an agent cannot derive from a catalog — which backend to choose,
what costs money, which errors are worth retrying.
