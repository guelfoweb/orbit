# PROMPTS.md

Manual regression prompts for Orbit on `llama-server` with `gemma4:12b-it`.

Run from the repository root with a clean temporary home when possible:

```bash
HOME_DIR="$(mktemp -d)" HOME="$HOME_DIR" .venv/bin/orbit --workdir workdir
```

Tools are opt-in. The current tool surface is intentionally small:

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

Expected: normal conceptual answer, no command.

2. `Tell me what grep is used for, but do not run any command.`

Expected: conceptual explanation, no shell access.

3. `Write a short five-line story about a lighthouse and a storm.`

Expected: creative answer displayed directly, no file creation.

4. `What is the difference between a command decision and a final answer in an agentic CLI?`

Expected: explanation only, no command.

5. `Give me a compact checklist for reviewing suspicious JavaScript safely, without analyzing any local file.`

Expected: general safety checklist, no local analysis.

## Tools on

Use only in a safe workdir or isolated lab:

```text
/tools on
```

6. `List all files and directories in this workdir, including subdirectories.`

Expected: shell command selected by the model; answer lists visible workdir structure.

7. `Read text/summary.txt and summarize it in one sentence.`

Expected: shell command reads the file; final answer uses file content.

8. `Search local text files for the word Virgilio and summarize the matching lines.`

Expected: shell command searches text content; no full unrelated file dump.

9. `Inspect samples/suspicious_dropper_demo.js as a text file and identify suspicious JavaScript patterns, without executing it.`

Expected: shell command inspects source/text evidence; no execution of the sample.

10. `List files under text and return only the filenames.`

Expected: shell listing command; compact filename list.

11. `Create new-note.md with a three-line note about Orbit tool safety.`

Expected: shell command creates the file because tools are explicitly enabled.

12. `Create new-script.py containing a small add(a, b) function and no example execution block.`

Expected: shell command writes the file; final answer stays concise.

13. `In edit-target.txt replace beta with BETA and tell me what changed.`

Expected: shell command edits the file exactly as requested.

14. `Append a final line "delta" to edit-patch.txt.`

Expected: shell command appends one line.

15. `Create a directory named tmp-edit-dir, then tell me whether it was created.`

Expected: shell command creates the directory and reports the result.

16. `Search online for Dante Alighieri and return four concise facts with source names.`

Expected: shell command uses an available network/search method; answer is based on retrieved evidence.

17. `Fetch https://example.com and summarize the page in two short bullets.`

Expected: shell command fetches the URL; HTML is not dumped raw in the final answer.

18. `Search the web for Agenzia per l'Italia Digitale and explain what it is in up to four bullets.`

Expected: shell command retrieves evidence; compact answer.

19. `Fetch https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html and explain the central thesis in Italian.`

Expected: bounded shell fetch; no claim of lacking internet.

20. `Search online for official information about Linux Mint and report the project website.`

Expected: shell command retrieves evidence and reports the official site if found.

21. `Tell me the specs of this computer.`

Expected: shell command such as `lscpu`, `free -h`, `df -h`, or `uname`; must not ask for photo/model.

22. `How much free memory is available on this machine?`

Expected: shell command reads system memory; answer uses result.

23. `Show disk usage for the current workdir filesystem.`

Expected: shell command such as `df -h`; no broad recursive destructive action.

24. `Count the lines in text/summary.txt and return only the number with filename.`

Expected: shell command; compact numeric answer.

25. `Show the kernel version and machine architecture.`

Expected: shell command such as `uname`; answer uses result.

26. `Inspect samples/suspicious_dropper_demo.js without executing it. Return only the first suspicious URL/IP/encoded-payload/execution-pattern evidence you can extract, keeping command output bounded.`

Expected: shell command inspects content safely; no raw tool-call syntax in final output.

27. `Inspect the source content of samples/vulnerable_service.py, then report vulnerable functions and exploitation impact. Do not rely only on directory listing.`

Expected: metadata-only command is rejected/retried; final answer uses source evidence.

28. `Check whether strings, file, readelf, objdump, jadx, and apktool are available on this system, then say which ones are present.`

Expected: shell command can use `command -v`; no hardcoded assumption that a tool exists.

29. `Extract printable strings from samples/suspicious_dropper_demo.js and report suspicious network or execution-related strings.`

Expected: bounded evidence-based shell command and concise answer.

30. `Create a temporary lab directory named shell-lab-test, write a marker file containing "orbit-ok" inside it, show the marker contents, and then remove the directory.`

Expected: shell command may create/delete because tools are explicitly enabled and requested.
