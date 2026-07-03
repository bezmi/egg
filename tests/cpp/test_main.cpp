// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_main.cpp — entry point for the combined cpp_tests executable.
//
// The metric/solve tests register themselves as Boost.UT `suite`s at namespace
// scope (test_metric.cpp, test_solve.cpp). Boost.UT's global runner executes
// every registered suite from its destructor at program exit and sets a nonzero
// process exit code on any failure, so main only needs to exist.
int main() { return 0; }
