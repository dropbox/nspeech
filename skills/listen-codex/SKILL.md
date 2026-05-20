---
name: listen
description: Transcribe speech from the microphone. Use when the user asks you to listen, hear, or transcribe what they say. Returns the spoken text on stdout.
---

# Listen — Live Speech-to-Text via Moonshine

Transcribe speech from the microphone using the `listen` command (GPU-accelerated Moonshine ASR).
In this environment, `listen` is approved to run outside the sandbox via the
persisted command prefix `["listen"]`; call it directly so it can access microphone
and GPU devices.

## When to use

- When the user asks you to listen, hear, or transcribe what they say.
- When the user wants to dictate text instead of typing.

## Usage

```bash
listen
```

The user speaks, sees their words in yellow on the terminal, then presses ENTER to emit the final text to stdout and exit. ESC clears the buffer, ESC on empty quits without output.

## Guidelines

- Capture the output: `text=$(listen)` to use the transcription in further processing
- The command blocks until the user presses ENTER or ESC
- Do not run in background — you need the stdout result
- Model loads on first invocation (~1s warmup)
