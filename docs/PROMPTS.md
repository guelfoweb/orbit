# PROMPTS.md

Manual regression prompts for Orbit against the current local backend, preferably native `orbit-server` on `http://127.0.0.1:11976`.

Run from the repository root with a clean temporary home when possible:

```bash
HOME_DIR="$(mktemp -d)" HOME="$HOME_DIR" .venv/bin/orbit \
  --base-url http://127.0.0.1:11976 \
  --workdir workdir
```

Tools are opt-in:

```text
/tools off = chat only
/tools on  = unrestricted local shell
```

## Chat

Use:

```text
/tools off
```

1. `Explain why local CPU-only inference is slower than GPU inference, in three concise bullets.`
2. `Tell me what grep is used for, but do not run any command.`
3. `Write a short five-line story about a lighthouse and a storm.`
4. `What is the difference between a command decision and a final answer in an agentic CLI?`
5. `Give me a compact checklist for reviewing suspicious JavaScript safely, in up to eight bullets, without analyzing any local file.`

Expected: normal answer, no shell command.

## Thinking

Use:

```text
/tools off
/think on
```

6. `Explain the plan before the final answer.`
7. `Who designed you?`

Expected:
- visible reasoning is shown first
- final answer is still produced
- if max tokens are exhausted in one-shot mode, Orbit suggests rerunning with `--max-tokens`, not `/continue`

Interactive continuation check:

1. run `explain the plan before the final answer` with a low output budget
2. if the answer stops at `length`, run `/continue`

Expected:
- `/continue` resumes from the current native context
- no `no active continuation state` error after a normal completed turn

## Tools on

Use only in a safe workdir or isolated lab:

```text
/tools on
```

6. `List all files and directories in this workdir, including subdirectories.`
7. `Read text/summary.txt and summarize it in one sentence.`
8. `Search local text files for the word Virgilio and summarize the matching lines.`
9. `Inspect samples/suspicious_dropper_demo.js as a text file and identify suspicious JavaScript patterns, without executing it.`
10. `List files under text and return only the filenames.`
11. `Create new-note.md with a three-line note about safe local shell tool usage.`
12. `Create new-script.py containing a small add(a, b) function and no example execution block.`
13. `In edit-target.txt replace beta with BETA and tell me what changed.`
14. `Append a final line "delta" to edit-patch.txt.`
15. `Create a directory named tmp-edit-dir, then tell me whether it was created.`
16. `Search online for Dante Alighieri and return four concise facts with source names.`
17. `Fetch https://example.com and summarize the page in two short bullets.`
18. `Search the web for Agenzia per l'Italia Digitale and explain what it is in up to four bullets.`
19. `Fetch https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html and explain the central thesis in Italian.`
20. `Search online for official information about Linux Mint and report the project website.`
21. `Tell me the specs of this computer.`
22. `How much free memory is available on this machine?`
23. `Show disk usage for the current workdir filesystem.`
24. `Count the lines in text/summary.txt and return only the number with filename.`
25. `Show the kernel version and machine architecture.`
26. `Inspect samples/suspicious_dropper_demo.js without executing it. Return only the first suspicious URL/IP/encoded-payload/execution-pattern evidence you can extract, keeping command output bounded.`
27. `Inspect the source content of samples/vulnerable_service.py, then report vulnerable functions and exploitation impact. Do not rely only on directory listing.`
28. `Check whether strings, file, readelf, objdump, jadx, and apktool are available on this system, then say which ones are present.`
29. `Extract printable strings from samples/suspicious_dropper_demo.js and report suspicious network or execution-related strings.`
30. `Create a temporary lab directory named shell-lab-test, write a marker file containing "orbit-ok" inside it, show the marker contents, and then remove the directory.`
31. `Read pdf/piccolo.pdf and summarize the document topic in one concise sentence.`
32. `Read pdf/grande.pdf and summarize the document topic in one concise sentence.`

Expected: command chosen by the model, bounded evidence, concise final answer.

## Tools + Thinking

Use:

```text
/tools on
/think on
```

33. `Use the shell tool to print the current working directory, then answer with only that path.`
34. `Use the shell tool to list the first three entries in the current directory. Think briefly, then answer with only the listed names, compactly. No categories or explanations.`

Expected:
- a short visible planning thought may appear before tool use
- the internal route phase must not consume the whole budget with extra reasoning
- the shell tool is still called
- the final answer is produced after the tool result
