---
name: listen
description: Transcribe speech from the microphone. Use when the user asks you to listen, hear, or transcribe what they say. Returns the spoken text on stdout.
allowed-tools: Bash
---

# Listen — Live Speech-to-Text via Moonshine

Transcribe speech from the microphone using the `listen` command (GPU-accelerated Moonshine ASR).

## When to use

- When the user asks you to listen, hear, or transcribe what they say.
- When the user wants to dictate text instead of typing.

## Usage

This command requires microphone access and cannot run inside the sandbox.
Ask the user to run it from the CLI prompt with `!`:

```
! listen
```

The user speaks, sees their words in yellow on the terminal, then presses ENTER to emit the final text to stdout and exit. ESC clears the buffer, ESC on empty quits without output.

## Guidelines

- Always tell the user to run `! listen` — do NOT call it via Bash (sandbox blocks mic access)
- The command blocks until the user presses ENTER or ESC
- Model loads on first invocation (~1s warmup)
