# Validation pilot

The 30-paper pilot compared LLM-extracted descriptors against manual expert annotation. Papers were drawn from an earlier 586-paper version of the corpus and processed with the same schema, model and prompt.

Files for this folder:

- `annotation_protocol.md`: how the expert annotations were made (which fields, the rules for ranges and units, how a value was judged correct, omitted or false).
- `pilot_annotations.csv`: one row per paper and field, giving the expert value, the extracted value and the judgment.
- `pilot_summary.csv`: phrase recall (89%) and false extraction rate (below 2%), with the counts behind them.
