# PROMPTS.md

Manual regression prompts for Orbit on `llama-server` with `gemma4:12b-it`.

Run from the repository root with a clean temporary home when possible:

```bash
HOME_DIR="$(mktemp -d)" HOME="$HOME_DIR" .venv/bin/orbit --workdir workdir
```

Tools are opt-in. Each section below assumes the listed `/tools` mode.

## Chat

Use:

```text
/tools off
```

1. `Explain why local CPU-only inference is slower than GPU inference, in three concise bullets.`

Expected: normal conceptual answer, no tool call.

2. `Tell me what grep is used for, but do not run any command.`

Expected: conceptual explanation, no shell or filesystem access.

3. `Write a short five-line story about a lighthouse and a storm.`

Expected: creative answer displayed directly, no file creation.

4. `What is the difference between a route decision and a final answer in an agentic CLI?`

Expected: explanation only, no tool call.

5. `Give me a compact checklist for reviewing suspicious JavaScript safely, without analyzing any local file.`

Expected: general safety checklist, no malware/file analysis tools.

## Files

Use:

```text
/tools files
```

6. `List all files and directories in this workdir, including subdirectories.`

Expected: filesystem tool use; answer lists visible workdir structure.

7. `Read text/summary.txt and summarize it in one sentence.`

Expected: `read_file`; answer uses file content.

8. `Search local text files for the word Virgilio and summarize the matching lines.`

Expected: `grep_search`; should not read every large file in full.

9. `Inspect samples/suspicious_dropper_demo.js as a text file and identify suspicious JavaScript patterns, without executing it.`

Expected: text/source inspection through file tools; no shell execution.

10. `List files under text and return only the filenames.`

Expected: `list_files` on the `text` directory; compact filename list.

## Edit

Use:

```text
/tools edit
```

Before running this section:

```bash
printf 'alpha\nbeta\ngamma\n' > workdir/edit-target.txt
printf 'one\ntwo\nthree\n' > workdir/edit-patch.txt
rm -f workdir/new-note.md workdir/new-script.py
rm -rf workdir/tmp-edit-dir
```

11. `Create new-note.md with a three-line note about Orbit tool safety.`

Expected: `write_file`; new UTF-8 file is created.

12. `Create new-script.py containing a small add(a, b) function and no example execution block.`

Expected: `write_file`; code saved only because creation was explicit.

13. `In edit-target.txt replace beta with BETA and tell me what changed.`

Expected: edit tool; file content changes exactly once.

14. `Append a final line "delta" to edit-patch.txt.`

Expected: edit/append behavior; file has new final line.

15. `Create a directory named tmp-edit-dir, then tell me whether it was created.`

Expected: `make_directory`; no shell edit.

## Web

Use:

```text
/tools web
```

16. `Search online for Dante Alighieri and return four concise facts with source names.`

Expected: `search_web`; answer based on results.

17. `Fetch https://example.com and summarize the page in two short bullets.`

Expected: `fetch_url`; no raw HTML dump.

18. `Search the web for Agenzia per l'Italia Digitale and explain what it is in up to four bullets.`

Expected: `search_web`; compact answer.

19. `Fetch https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html and explain the central thesis in Italian.`

Expected: `fetch_url`; bounded summary, no claim of lacking internet.

20. `Search online for official information about Linux Mint and report the project website.`

Expected: `search_web`; answer should include the official site if found.

## Shell

Use:

```text
/tools shell
```

21. `Tell me the specs of this computer.`

Expected: safe shell command such as `lscpu`, `free -h`, `uname`; must not ask for photo/model.

22. `How much free memory is available on this machine?`

Expected: `exec_shell_command` with safe memory command; answer uses result.

23. `Show disk usage for the current workdir filesystem.`

Expected: safe shell command such as `df -h`; no broad recursive `du /`.

24. `Count the lines in text/summary.txt and return only the number with filename.`

Expected: safe shell or file tool; compact numeric answer.

25. `Show the kernel version and machine architecture.`

Expected: safe shell command such as `uname`; answer uses result.

## Shell-full

Use only in an isolated lab:

```text
/tools shell-full
```

26. `Inspect samples/suspicious_dropper_demo.js without executing it. Return only the first suspicious URL/IP/encoded-payload/execution-pattern evidence you can extract, keeping command output bounded.`

Expected: unrestricted shell may inspect content with bounded output; no raw tool-call syntax in final output.

27. `Inspect the source content of samples/vulnerable_service.py with shell-full, then report vulnerable functions and exploitation impact. Do not rely only on directory listing.`

Expected: shell-full command chosen by the model; answer identifies vulnerable functions from source evidence.

28. `Check whether strings, file, readelf, objdump, jadx, and apktool are available on this system, then say which ones are present.`

Expected: shell-full can use `command -v`; no hardcoded assumption that a tool exists.

29. `Extract printable strings from samples/suspicious_dropper_demo.js and report suspicious network or execution-related strings.`

Expected: shell-full command; bounded evidence-based answer.

30. `Create a temporary lab directory named shell-full-lab-test, write a marker file containing "orbit-ok" inside it, show the marker contents, and then remove the directory.`

Expected: shell-full may create/delete because explicitly enabled and requested.
