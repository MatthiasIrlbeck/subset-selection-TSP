#include "test_common.hpp"

namespace {

ALDOUS_TEST(test_full_nearest_neighbor_constructor) {
    Rng rng(0x5eed1234U);
    for (bool periodic : {false, true}) {
        for (int trial = 0; trial < 12; ++trial) {
            const int n = 7 + trial;
            Instance inst;
            inst.periodic = periodic;
            inst.generate(n, rng);
            // Exercise both a narrow retained row and a saturated row. The
            // constructor must fall back to an exact scan after consuming KNN.
            const int knn = (trial % 2 == 0) ? std::min(3, n - 1) : n - 1;
            inst.build_knn(knn, KnnBackend::GridExact);
            std::vector<int> all(static_cast<std::size_t>(n));
            std::iota(all.begin(), all.end(), 0);
            for (int start : {0, n / 2, n - 1}) {
                const std::vector<int> expected =
                    nearest_neighbor_order(inst, all, start);
                const std::vector<int> actual =
                    nearest_neighbor_full_order(inst, start);
                require(actual == expected,
                        "KNN-guided full-TSP construction matches exact nearest-neighbor order");
            }
        }
    }

    // Pin deterministic node-ID tie breaking at a symmetric point.
    Instance tied;
    tied.set_points({{0.0, 0.0}, {1.0, 0.0}, {-1.0, 0.0}, {0.0, 2.0}});
    tied.build_knn(1, KnnBackend::GridExact);
    const std::vector<int> tied_order = nearest_neighbor_full_order(tied, 0);
    require(tied_order.size() == 4U && tied_order[1] == 1,
            "full nearest-neighbor construction breaks equal-distance ties by node ID");
}

ALDOUS_TEST(test_tsp_screening_thread_invariance) {
    Instance inst;
    inst.periodic = true;
    Rng generator(0x12345678U);
    inst.generate(96, generator);
    inst.build_knn(16, KnnBackend::GridExact);

    SolverOptions serial;
    serial.seed = 777;
    serial.tsp_candidate_starts = 9;
    serial.tsp_restarts = 4;
    serial.tsp_farthest_starts = 0;
    serial.tsp_ils = 12;
    serial.tsp_patience = 5;
    serial.restart_threads = 1;
    serial.final_exhaustive_k = 0;

    SolverOptions parallel = serial;
    parallel.restart_threads = 4;
    Rng serial_rng(9191);
    Rng parallel_rng(9191);
    const SolveResult a = solve_tsp(inst, serial_rng, serial);
    const SolveResult b = solve_tsp(inst, parallel_rng, parallel);

    require(a.tour.nodes == b.tour.nodes && a.tour.length == b.tour.length,
            "screened full-TSP result is invariant to restart worker count");
    require(a.stats.tsp_candidate_starts == 9U
                && a.stats.tsp_promoted_restarts == 4U
                && a.stats.tsp_restarts == 4U,
            "full-TSP telemetry distinguishes screened and promoted starts");
    require(a.restarts.size() == 4U && b.restarts.size() == a.restarts.size(),
            "full-TSP emits one record per promoted candidate");
    for (std::size_t i = 0; i < a.restarts.size(); ++i) {
        const RestartRecord& lhs = a.restarts[i];
        const RestartRecord& rhs = b.restarts[i];
        require(lhs.length == rhs.length && lhs.kind == rhs.kind
                    && lhs.role == rhs.role
                    && lhs.seed_variant == rhs.seed_variant
                    && lhs.strong_polished == rhs.strong_polished,
                "screened TSP restart records are thread invariant");
        require(lhs.kind == RestartKind::TspNearestNeighbor
                    && lhs.role == RestartRole::RacedProduction
                    && lhs.strong_polished,
                "default screened TSP records scalable promoted starts");
    }
}

ALDOUS_TEST(test_staged_search_funnel) {
    Instance inst;
    inst.periodic = true;
    Rng generator(808080);
    inst.generate(140, generator);
    inst.build_knn(18, KnnBackend::GridExact);

    SolverOptions staged;
    staged.seed = 8081;
    staged.subset_restarts = 6;
    staged.staged_search = true;
    staged.strong_polish_finalists = 2;
    staged.strong_polish_min_jaccard = 0.02;
    staged.sa_iters = 250;
    staged.restart_threads = 3;
    staged.final_exhaustive_k = 0;
    staged.pair_exchange_passes = 0;
    staged.ruin_recreate_rounds = 1;
    staged.ejection_chain_starts = 1;
    staged.disable_path_relink = true;

    Rng staged_rng(8082);
    const SolveResult result = solve_subset(inst, 56, staged_rng, staged);
    require(result.stats.strong_polish_candidates == 6U,
            "staged search records every eligible post-SA candidate");
    require(result.stats.strong_polish_finalists == 2U,
            "staged search strongly polishes only the configured finalists");
    require(static_cast<std::size_t>(std::count_if(
                result.restarts.begin(), result.restarts.end(),
                [](const RestartRecord& record) { return record.strong_polished; })) == 2U,
            "restart provenance identifies exactly the promoted finalists");

    // Promoting every restart must reproduce the historical inline trajectory.
    SolverOptions all_staged = staged;
    all_staged.subset_restarts = 4;
    all_staged.strong_polish_finalists = 4;
    all_staged.restart_threads = 1;
    SolverOptions inline_search = all_staged;
    inline_search.staged_search = false;
    Rng all_staged_rng(8083);
    Rng inline_rng(8083);
    const SolveResult staged_all = solve_subset(inst, 56, all_staged_rng, all_staged);
    const SolveResult inline_all = solve_subset(inst, 56, inline_rng, inline_search);
    require(staged_all.tour.nodes == inline_all.tour.nodes
                && staged_all.tour.length == inline_all.tour.length,
            "staging all candidates preserves the legacy inline result");
    require(staged_all.restarts.size() == inline_all.restarts.size(),
            "all-finalist staging preserves the restart population");
    for (std::size_t i = 0; i < staged_all.restarts.size(); ++i) {
        require(staged_all.restarts[i].length == inline_all.restarts[i].length,
                "all-finalist staging resumes each restart from its exact RNG state");
    }
}

ALDOUS_TEST(test_path_relink_literal_work_budgets) {
    Instance inst;
    inst.periodic = true;
    Rng generator(91919);
    inst.generate(180, generator);
    inst.build_knn(20, KnnBackend::GridExact);

    SolverOptions options;
    options.seed = 91920;
    options.subset_restarts = 7;
    options.strong_polish_finalists = 3;
    options.sa_iters = 100;
    options.restart_threads = 2;
    options.final_exhaustive_k = 0;
    options.path_relink_top = 3;
    options.path_relink_diverse_reserve = 1;
    options.path_relink_max_pairs = 1;
    options.path_relink_max_removed = 64;
    options.path_relink_max_removed_sum = 64;
    options.path_relink_max_candidate_scans = 200000;

    Rng solve_rng(91921);
    const SolveResult bounded = solve_subset(inst, 72, solve_rng, options);
    require(bounded.stats.path_relink_pairs_considered <= 3U,
            "literal path-relink node cap exposes at most choose(top,2) pairs");
    require(bounded.stats.path_relink_attempts <= 1U,
            "path-relink pair-attempt budget is enforced literally");
    require(bounded.stats.path_relink_removed_sum <= 64U,
            "path-relink cumulative symmetric-difference budget is enforced");
    require(bounded.stats.path_relink_candidate_scans <= 200000U,
            "path-relink candidate-scan budget is enforced");

    SolverOptions blocked = options;
    blocked.path_relink_max_candidate_scans = 1;
    Rng blocked_rng(91921);
    const SolveResult no_work = solve_subset(inst, 72, blocked_rng, blocked);
    require(no_work.stats.path_relink_attempts == 0U
                && no_work.stats.path_relink_candidate_scans == 0U,
            "a candidate-scan budget below every ranked pair prevents relinking work");
    require(no_work.stats.path_relink_pairs_skipped_budget > 0U,
            "budget-rejected path-relink pairs are reported");
}

ALDOUS_TEST(test_campaign_stream_identity_contract) {
    RunOptions base;
    base.N = 10;
    base.instances = 3;
    base.threads = 2;
    base.p_values = {0.5};
    base.include_instance_rows = true;
    base.campaign_id = "identity-test";
    base.campaign_shard = 7;
    base.replicate_offset = 40;
    base.point_seed = 111;
    base.search_seed = 222;
    base.solver_policy_id = "exact-calibration";
    base.fidelity_level = "strong";
    base.solver.exact_subset_max_n = 10;
    base.solver.restart_threads = 1;

    const ResultsDocument first = ExperimentRunner(base).run();
    RunOptions changed_search = base;
    changed_search.search_seed = 333;
    const ResultsDocument second = ExperimentRunner(changed_search).run();
    RunOptions changed_points = base;
    changed_points.point_seed = 444;
    const ResultsDocument third = ExperimentRunner(changed_points).run();

    require(first.instance_rows.size() == 3U
                && second.instance_rows.size() == first.instance_rows.size()
                && third.instance_rows.size() == first.instance_rows.size(),
            "campaign identity test retains every instance row");
    for (std::size_t i = 0; i < first.instance_rows.size(); ++i) {
        const InstanceResultRow& a = first.instance_rows[i];
        const InstanceResultRow& b = second.instance_rows[i];
        const InstanceResultRow& c = third.instance_rows[i];
        require(a.replicate_id == 40U + i,
                "replicate IDs use the campaign-global offset");
        require(a.point_stream_id == b.point_stream_id
                    && a.search_stream_id != b.search_stream_id,
                "changing only the search seed preserves point streams");
        require(a.point_stream_id != c.point_stream_id,
                "changing the point seed changes point streams");
        require(a.values == b.values,
                "exact calibration values are independent of the search seed");
    }
    const std::string json = results_to_json(first);
    require(json.find("\"campaign_metadata\"") != std::string::npos
                && json.find("\"campaign_id\": \"identity-test\"")
                    != std::string::npos
                && json.find("\"point_stream_id\": \"")
                    != std::string::npos,
            "JSON exposes campaign metadata and fixed-width stream identifiers");
}

ALDOUS_TEST(test_json_numeric_precision) {
    ResultsDocument doc;
    doc.N = 12;
    doc.instances_done = 1;
    doc.instances_target = 1;
    doc.options.N = 12;
    doc.options.instances = 1;
    doc.options.solver.grid_cell = 1e-12;
    doc.p_values = {1e-9, 2e-9};
    PValueSummary s1;
    s1.k = 3;
    s1.values = {1e-12};
    s1.mean = 1e-12;
    s1.min = 1e-12;
    s1.max = 1e-12;
    PValueSummary s2 = s1;
    s2.values = {2e-12};
    s2.mean = 2e-12;
    s2.min = 2e-12;
    s2.max = 2e-12;
    doc.summary["1e-09"] = s1;
    doc.summary["2e-09"] = s2;
    const std::string text = results_to_json(doc);
    require(text.find("\"grid_cell\": 0") == std::string::npos, "tiny grid-cell does not serialize as zero");
    require(text.find("e-12") != std::string::npos || text.find("e-13") != std::string::npos,
            "tiny grid-cell is serialized with scientific precision");
    require(text.find("1e-09") != std::string::npos || text.find("1.0000000000000001e-09") != std::string::npos,
            "tiny p-value is serialized with scientific precision");
}


ALDOUS_TEST(test_json_escape_regression) {
    const std::string escaped = json_escape("quote\" backslash\\ newline\n tab\t control\x01");
    require(escaped.find("\\\"") != std::string::npos, "JSON escaping handles quotes");
    require(escaped.find("\\\\") != std::string::npos, "JSON escaping handles backslashes");
    require(escaped.find("\\n") != std::string::npos, "JSON escaping handles newlines");
    require(escaped.find("\\t") != std::string::npos, "JSON escaping handles tabs");
    require(escaped.find("\\u0001") != std::string::npos, "JSON escaping handles control characters");

    const std::string valid_utf8 = "caf\xc3\xa9 \xf0\x9f\x8c\x8d";
    require(json_escape(valid_utf8) == valid_utf8,
            "valid UTF-8 is preserved by JSON string serialization");
    const std::string malformed = std::string("bad:") + static_cast<char>(0xc0)
        + static_cast<char>(0xaf) + ":end";
    const std::string repaired = json_escape(malformed);
    require(repaired.find("\\ufffd") != std::string::npos
                && repaired.find(static_cast<char>(0xc0)) == std::string::npos,
            "malformed UTF-8 is deterministically replaced at the JSON boundary");
}

ALDOUS_TEST(test_json_numbers_ignore_global_locale) {
    class CommaDecimal final : public std::numpunct<char> {
    protected:
        char do_decimal_point() const override { return ','; }
    };

    const std::locale previous = std::locale();
    std::locale::global(std::locale(previous, new CommaDecimal));
    ResultsDocument doc;
    doc.N = 4;
    doc.instances_done = 1;
    doc.instances_target = 1;
    doc.wall_seconds = 1.25;
    doc.p_values = {0.5};
    PValueSummary summary;
    summary.k = 3;
    summary.mean = 1.25;
    summary.min = 1.25;
    summary.max = 1.25;
    summary.values = {1.25};
    doc.summary[p_value_key(0.5)] = summary;
    const std::string text = results_to_json(doc);
    std::locale::global(previous);
    require(text.find("1.25") != std::string::npos,
            "JSON doubles use a locale-independent decimal point");
    require(text.find("1,25") == std::string::npos,
            "comma-decimal locales cannot corrupt JSON numbers");
    require(p_value_key(0.5) == "0.5",
            "probability keys are locale independent");
}


ALDOUS_TEST(test_public_api_validation_and_concurrent_atomic_writers) {
    RunOptions invalid_probability;
    invalid_probability.N = 12;
    invalid_probability.instances = 1;
    invalid_probability.threads = 1;
    invalid_probability.p_values = {-0.5, 1.5};
    bool rejected_probability = false;
    try {
        (void)ExperimentRunner(invalid_probability);
    } catch (const std::invalid_argument&) {
        rejected_probability = true;
    }
    require(rejected_probability,
            "direct ExperimentRunner API rejects out-of-domain probabilities");

    Instance instance;
    Rng point_rng(0xabc123U);
    instance.generate(12, point_rng);
    instance.build_knn(6);
    SolverOptions options;
    options.subset_restarts = 1;
    options.sa_iters = 0;
    options.restart_threads = 1;
    options.disable_path_relink = true;
    bool rejected_cardinality = false;
    try {
        Rng solve_rng(17U);
        (void)solve_subset(instance, -7, solve_rng, options);
    } catch (const std::invalid_argument&) {
        rejected_cardinality = true;
    }
    require(rejected_cardinality,
            "direct subset API rejects invalid cardinalities instead of clamping");

    OutputDurability parsed = OutputDurability::None;
    require(parse_output_durability("full", parsed)
                && parsed == OutputDurability::Full
                && std::string(output_durability_name(parsed)) == "full",
            "output durability metadata round-trips through its public parser");

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / ("aldous_tsp_atomic_" + std::to_string(
               static_cast<unsigned long long>(
                   std::chrono::steady_clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(directory);
    const std::filesystem::path target = directory / "result.json";
    constexpr int writers = 8;
    constexpr int rounds = 20;
    std::vector<std::string> payloads;
    payloads.reserve(static_cast<std::size_t>(writers * rounds));
    for (int writer = 0; writer < writers; ++writer) {
        for (int round = 0; round < rounds; ++round) {
            payloads.push_back(
                "{\"writer\":" + std::to_string(writer)
                + ",\"round\":" + std::to_string(round)
                + ",\"padding\":\"" + std::string(2048U, static_cast<char>('a' + writer))
                + "\"}\n");
        }
    }
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int writer = 0; writer < writers; ++writer) {
        threads.emplace_back([&, writer] {
            for (int round = 0; round < rounds; ++round) {
                const std::size_t index = static_cast<std::size_t>(writer * rounds + round);
                std::string error;
                if (!write_text_file_atomic(
                        target.string(), payloads[index], OutputDurability::None, &error)) {
                    ++failures;
                }
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    require(failures.load() == 0,
            "concurrent atomic writers do not collide on a shared temporary filename");
    std::ifstream input(target, std::ios::binary);
    const std::string final_payload(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    require(std::find(payloads.begin(), payloads.end(), final_payload) != payloads.end(),
            "concurrent atomic output is one complete writer payload");

    const std::filesystem::path no_clobber_target = directory / "no-clobber.json";
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<int> committed{0};
    std::atomic<int> refused{0};
    std::vector<std::thread> no_clobber_threads;
    for (int writer = 0; writer < writers; ++writer) {
        no_clobber_threads.emplace_back([&, writer] {
            ++ready;
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const AtomicWriteResult result = write_text_file_atomic(
                no_clobber_target.string(), payloads[static_cast<std::size_t>(writer)],
                ReplacePolicy::NoReplace, OutputDurability::None);
            if (result.committed()) {
                ++committed;
            } else if (result.message.find("Refusing to overwrite") != std::string::npos) {
                ++refused;
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != writers) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : no_clobber_threads) {
        thread.join();
    }
    require(committed.load() == 1 && refused.load() == writers - 1,
            "atomic no-clobber commits exactly one concurrent writer");
    const AtomicWriteResult existing_result = write_text_file_atomic(
        no_clobber_target.string(), "replacement", ReplacePolicy::NoReplace,
        OutputDurability::Full);
    require(!existing_result.committed()
                && existing_result.state == OutputCommitState::NotCommitted,
            "no-clobber reports an existing target as not committed");

    for (const std::filesystem::directory_entry& entry
         : std::filesystem::directory_iterator(directory)) {
        require(entry.path() == target || entry.path() == no_clobber_target,
                "successful atomic writes leave no orphaned temporary files");
    }
    std::filesystem::remove_all(directory);
}

ALDOUS_TEST(test_json_atomic) {
    ResultsDocument doc;
    doc.N = 10;
    doc.instances_done = 1;
    doc.instances_target = 1;
    doc.options.solver.grid_cell = 1.0e-12;
    doc.p_values = {1.0e-9, 2.0e-9, 1.0};
    PValueSummary s;
    s.k = 10;
    s.values = {0.7};
    s.mean = 0.7;
    s.min = 0.7;
    s.max = 0.7;
    doc.summary["1.0"] = s;
    const std::string text = results_to_json(doc);
    require(text.find("\"schema_version\": 16") != std::string::npos, "JSON schema version present");
    require(text.find("\"timing\"") != std::string::npos
                && text.find("\"solver_wall_seconds\"") != std::string::npos
                && text.find("\"control_reference_seconds\"") != std::string::npos
                && text.find("\"aggregation_seconds\"") != std::string::npos
                && text.find("\"experiment_wall_seconds\"") != std::string::npos,
            "JSON includes explicit experiment-phase timing");
    require(text.find("\"git_tree\"") != std::string::npos
                && text.find("\"source_dirty\"") != std::string::npos
                && text.find("\"revision_source\"") != std::string::npos
                && text.find("\"source_refnames\"") != std::string::npos,
            "JSON includes source commit/tree provenance and dirty state");
    require(text.find("\"build_metadata\"") != std::string::npos, "JSON includes build metadata");
    require(text.find("\"memory_plan\"") != std::string::npos
                && text.find("\"estimated_instance_bytes\"") != std::string::npos
                && text.find("\"estimated_held_karp_call_bytes\"") != std::string::npos
                && text.find("\"estimated_oracle_call_bytes\"") != std::string::npos
                && text.find("\"held_karp_concurrency\"") != std::string::npos
                && text.find("\"limited_by_budget\"") != std::string::npos,
            "JSON includes phase-aware memory and effective-concurrency telemetry");
    require(text.find("\"oracle_call_records\"") != std::string::npos, "JSON includes oracle call records");
    require(text.find("\"configuration_fingerprint\"") != std::string::npos
                && text.find("\"method_fingerprint\"") != std::string::npos,
            "JSON includes exact configuration and method fingerprints");
    require(text.find("\"summary_rows\"") != std::string::npos, "JSON includes array-form summary rows");
    require(text.find("\"knn_build_seconds\"") != std::string::npos, "JSON includes KNN timing stats");
    require(text.find("\"pair_exchange_max_k\"") != std::string::npos,
            "JSON includes the pair-exchange safety gate");
    require(text.find("\"pair_exchange_skipped_large_k\"") != std::string::npos,
            "JSON includes pair-exchange gate telemetry");
    require(text.find("\"knn_requested_grid_instances\"") != std::string::npos, "JSON includes effective KNN backend stats");
    require(text.find("\"instance_rows\"") != std::string::npos, "JSON includes per-instance row container");
    require(text.find("\"target_compile_options\"") != std::string::npos, "JSON includes target compile options metadata");
    require(text.find("\"effective_optimization_level\"") != std::string::npos, "JSON includes structured optimization-level metadata");
    require(text.find("e-13") != std::string::npos || text.find("e-12") != std::string::npos,
            "JSON preserves tiny grid-cell scale");
    require(text.find("e-09") != std::string::npos || text.find("e-9") != std::string::npos,
            "JSON preserves tiny p-value scale");
    require(text.find("0.0000000000") == std::string::npos, "JSON does not fixed-format tiny values to zero");
    const std::filesystem::path out = std::filesystem::temp_directory_path() / "aldous_tsp_test_results.json";
    std::string err;
    const AtomicWriteResult write_result = write_text_file_atomic(
        out.string(), text, ReplacePolicy::ReplaceExisting, OutputDurability::Full);
    require(write_result.satisfies(OutputDurability::Full), "atomic write succeeds");
    require(write_result.total_seconds >= write_result.write_seconds
                && write_result.total_seconds >= write_result.commit_seconds
                && write_result.total_seconds >= write_result.synchronization_seconds,
            "atomic write reports internally consistent phase timing");
    std::ifstream in(out, std::ios::binary);
    std::string roundtrip((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    require(roundtrip == text, "atomic write roundtrip");
    std::filesystem::remove(out);
}


} // namespace
