# Coverage-guided fuzzing

The repository provides dedicated libFuzzer targets for the option parser,
instance/KNN construction, mutable tour operations, and bounded end-to-end
solver execution. They are built only with Clang.

```bash
CC=clang CXX=clang++ cmake -S . -B build-fuzz -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTING=ON \
  -DALDOUS_TSP_BUILD_FUZZERS=ON \
  -DALDOUS_TSP_ENABLE_WARNINGS=ON \
  -DALDOUS_TSP_ENABLE_WERROR=ON
cmake --build build-fuzz --target aldous_tsp_fuzzers --parallel
ctest --test-dir build-fuzz -L fuzz-smoke --output-on-failure
```

For a longer local campaign, copy the committed seeds to a writable corpus
so exploratory mutations do not dirty the source tree:

```bash
mkdir -p build-fuzz/corpus
cp -a fuzz/corpus/. build-fuzz/corpus/
./build-fuzz/fuzz_cli -dict=fuzz/dictionaries/cli.dict \
  -max_total_time=600 -print_funcs=0 build-fuzz/corpus/cli
./build-fuzz/fuzz_instance -max_total_time=600 -print_funcs=0 \
  build-fuzz/corpus/instance
./build-fuzz/fuzz_tour -max_total_time=600 -print_funcs=0 \
  build-fuzz/corpus/tour
./build-fuzz/fuzz_solver -max_total_time=600 -print_funcs=0 \
  -timeout=15 build-fuzz/corpus/solver
```

CTest smoke runs copy the committed seeds into a writable build-tree corpus, so coverage discoveries never modify the source checkout.

The scheduled GitHub Actions workflow runs each target with ASan and UBSan,
retains minimized corpora and crash artifacts, and also supports manual runs.
Input sizes and solver budgets are deliberately capped so a corpus entry cannot
turn a fuzz worker into an unbounded benchmark.

Unexpected C++ exceptions are not swallowed by the harnesses: valid-by-construction inputs that violate an invariant become reproducible crash artifacts. Parser rejection of malformed CLI text remains an ordinary return path.
