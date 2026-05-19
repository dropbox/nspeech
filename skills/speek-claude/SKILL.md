---
name: speek
description: Speak a short summary aloud using Kokoro TTS. Use proactively at the end of any completed task to announce what was done in one sentence. Also use when the user explicitly asks to hear something spoken.
allowed-tools: Bash
---

# Speek — Text-to-Speech via Kokoro TTS

Speak text aloud using the `speek` command (GPU-accelerated Kokoro TTS).

## When to use

- **Proactively** at the conclusion of any task: summarize what you accomplished in one short sentence and speak it.
- When the user explicitly asks you to say or speak something.

## Usage

```bash
speek "Done. The encoder now uses private Metal buffers with blit staging."
```

Keep the text short (one sentence, under 20 words). Speek loads a 134MB model on first invocation (~1s), so don't call it repeatedly in quick succession.

## Guidelines

- Summarize the *outcome*, not the process: "Fixed the segfault in the decoder" not "I edited gpu_decoder_metal.rs lines 45-80"
- Use natural, conversational phrasing
- Do not speak errors or failures — only successful completions
- Do not speak if the task was trivial (e.g. answering a yes/no question)
- Run in background so it doesn't block your response to the user
