# Structured File Tools Shadow: Technical Stop

## Scope

Orbit production represents file reads and content searches through the
existing shell tool. An additive benchmark candidate introduced strict
`read_file` and `grep_search` schemas only for generation-only observation; it
never added those names to production tools or executed either candidate.

The comparison retained all production tools and added the two schemas, so it
did not credit an unrealistically smaller tool set. It covered path integrity,
including spaces, apostrophes, and non-ASCII characters, along with exact tool
and argument agreement.

## Measured Results

- the larger schema increased cold prompt and evaluated-token cost;
- cold wall time did not improve;
- exact argument fidelity regressed in relevant path and search cases;
- schema safety alone did not offset the model-adherence regression.

## Conclusion

The benchmark-specific schemas, runtime module, harness switches, and tests
were intentionally discarded after RC23. No structured file tool is active or
available to routing.

## Reopening Criteria

Reconsider this work only with a new model or template and a process-isolated
additive-schema comparison that demonstrates all of the following:

- exact tool and argument fidelity on ordinary and complex paths;
- no increase in unwanted tool attempts or truncation;
- lower total prompt/evaluated-token cost;
- lower or neutral cold and warm wall time;
- no routing, lifecycle, canonical-contract, or safety regression.
