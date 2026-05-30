# PROMPTS

## Strategic Suite

1. `list all files and directories in the current workspace`
2. `inspect the workspace and tell me which files appear to contain source code, configuration, or documentation.`
3. `decode this string "Y2lhbw==" from base64`
4. `read the file summary.txt and summarize its purpose in two sentences`
5. `what is the size and modified time of agent.py?`
6. `tell me how many files exist in the workspace and what the newest file is.`
7. `review agent.py for vulnerabilities and security issues`
8. `summarize the article in italian at this link: https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html explain the central thesis and key messages`
9. `compare two images: cmp-blue.png and vision-test.png and tell me the differences`
10. `search online for information about Dante Alighieri`
11. `Does this page mention "transumanesimo"? https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html`
12. `Does this page talk about human dignity and freedom in the age of artificial intelligence? https://www.vatican.va/content/leo-xiv/it/encyclicals/documents/20260515-magnifica-humanitas.html`

## Strong Prompts

1. `Inspect the workspace, identify the three most relevant files for understanding this project, read them, and give me a concise technical assessment with one concrete risk and one improvement suggestion.`
2. `Compare cmp-blue.png, cmp-red.png, and vision-test.png. If any image is missing, say exactly which one and continue with the others. Then explain the differences in one short paragraph.`
3. `Read promessi_sposi.txt, summarize it in exactly 4 lines, and make sure the summary is faithful to the text without inventing any details.`
4. `Search the workspace for anything that looks like a security issue, then explain whether the issue is in code, configuration, or documentation. If you cannot prove it, say so explicitly.`
5. `Answer this only after checking the available tools and the current workspace state: what is the newest file in the workspace, what is its modified time, and how did you determine it?`

## Skill Prompts

Run with `--skill obsidian-daily` and the Obsidian vault as `--workdir`.

1. `Read Daily.md. Extract all open tasks marked with "- [ ]", then analyze them semantically: group them by Work and Personal, identify overdue tasks, recurring tasks, and suggest the top 3 priorities for today. Use only evidence from Daily.md.`
