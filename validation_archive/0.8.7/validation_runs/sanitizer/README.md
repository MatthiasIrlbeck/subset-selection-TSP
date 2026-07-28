# Sanitizer validation

Status: completed.

Commands run:
- Debug ASan/UBSan build with Python tests enabled.
- Full CTest under ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 and UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1.
- Manual hybrid CLI smoke under ASan/UBSan.

Results:
- Full sanitizer CTest passed 13/13.
- Manual ASan smoke output is stored in asan-smoke.json.
- asan-smoke.json validated against schema/results.schema.json.
