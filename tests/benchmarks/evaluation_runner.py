import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BENCHMARKS = [

    (
        "Generation Benchmark",
        "tests.benchmarks.generation_benchmark"
    ),

    (
        "Retrieval Benchmark",
        "tests.benchmarks.retrieval_benchmark"
    ),

    (
        "RAG Benchmark",
        "tests.benchmarks.rag_benchmark"
    ),

    (
        "Reasoning Benchmark",
        "tests.benchmarks.reasoning_benchmark"
    ),

    (
        "Coding Benchmark",
        "tests.benchmarks.coding_benchmark"
    ),

    (
        "Mathematics Benchmark",
        "tests.benchmarks.math_benchmark"
    ),

    (
        "Hallucination Benchmark",
        "tests.benchmarks.hallucination_benchmark"
    )

]


def run_benchmark(name, module):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    start = time.perf_counter()

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT
    )

    elapsed = time.perf_counter() - start

    passed = result.returncode == 0

    print()

    print("Status :", "PASS" if passed else "FAIL")
    print(f"Time   : {elapsed:.2f} sec")

    return {
        "name": name,
        "module": module,
        "status": passed,
        "time": elapsed
    }


def main():

    print("=" * 70)
    print("NEXORALM PHASE 4 EVALUATION")
    print("=" * 70)

    results = []

    total_start = time.perf_counter()

    for name, module in BENCHMARKS:

        results.append(
            run_benchmark(
                name,
                module
            )
        )

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    passed = sum(item["status"] for item in results)
    failed = len(results) - passed

    for item in results:

        status = "PASS" if item["status"] else "FAIL"

        print(
            f"{item['name']:<30}"
            f"{status:<8}"
            f"{item['time']:.2f} sec"
        )

    print("\n" + "-" * 70)

    print(f"Benchmarks Run     : {len(results)}")
    print(f"Benchmarks Passed  : {passed}")
    print(f"Benchmarks Failed  : {failed}")
    print(f"Success Rate       : {(passed/len(results))*100:.2f}%")
    print(f"Total Runtime      : {total_time:.2f} sec")

    print("=" * 70)


if __name__ == "__main__":
    main()