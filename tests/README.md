# Test suite

Run the complete Python suite from the project root:

```bash
python -m pytest tests
```

The production stage files `translate_test.py`, `remove_text_test.py`, and
`typeset_test.py` intentionally remain at the project root for compatibility.
Despite their historical names, they are runtime pipeline entry points.

