#include <aldous_tsp/exact_subset.hpp>
#include <aldous_tsp/instance.hpp>
#include <aldous_tsp/restart.hpp>
#include <aldous_tsp/rng.hpp>

int main() {
    static_assert(aldous_tsp::kRestartKindCount == 10U,
                  "installed restart metadata must be complete");
    aldous_tsp::RestartKind parsed = aldous_tsp::RestartKind::Random;
    if (!aldous_tsp::restart_kind_from_code(7, parsed)
        || parsed != aldous_tsp::RestartKind::Dense) {
        return 2;
    }
    aldous_tsp::Rng rng(123);
    aldous_tsp::Instance inst;
    inst.generate(8, rng);
    inst.build_knn(3, aldous_tsp::KnnBackend::GridExact);
    if (!inst.verify_knn(8, rng)) {
        return 1;
    }
    const aldous_tsp::ExactSubsetSolution exact =
        aldous_tsp::exact_subset_cycle(inst, 4);
    return exact.solved && exact.proven_optimal && exact.cycle.size() == 4U
        ? 0
        : 3;
}
