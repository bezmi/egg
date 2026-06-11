// ut_cfg.hpp — shared Boost.UT configuration for the C++ test executables.
//
// Boost.UT's stock reporters are quiet on success: the default runner uses the
// JUnit reporter, and even the verbose `reporter<printer>` buffers per-test
// lines and only flushes them on failure — so a passing run shows nothing but
// an aggregate assert count. We install a custom reporter (`verbose_reporter`)
// that streams each test and its result to stdout as it runs, including any
// `<< message` attached to a failing expect() and `log << ...` lines.
//
// `cfg` is a variable template; this is a partial specialization of it. It MUST
// be visible in every test translation unit that registers/runs tests, so each
// test TU includes THIS header (not <boost/ut.hpp> directly). Otherwise suites
// in different TUs would bind to different runners (ODR mismatch).
#pragma once

#include <boost/ut.hpp>
#include <cstddef>
#include <iostream>
#include <string_view>

namespace egg_test
{

// Streams test progress directly to stdout. Implements exactly the events the
// Boost.UT runner dispatches unconditionally (suite_begin/end and test_finish
// are probed with `requires`, so omitting them is fine).
class verbose_reporter
{
  public:
    auto on(boost::ut::events::run_begin) -> void {}

    auto on(boost::ut::events::suite_begin suite) -> void
    { std::cout << "\n[suite] " << suite.name << '\n'; }
    auto on(boost::ut::events::suite_end) -> void {}

    auto on(boost::ut::events::test_begin tb) -> void
    {
        std::cout << "  - " << tb.name << '\n';
        fail_at_begin_ = asserts_fail_;
    }

    auto on(boost::ut::events::test_run) -> void {}

    auto on(boost::ut::events::test_skip s) -> void
    {
        std::cout << "  - " << s.name << " ... SKIPPED\n";
        ++tests_skip_;
    }

    auto on(boost::ut::events::test_end te) -> void
    {
        const bool failed = asserts_fail_ > fail_at_begin_;
        if (failed) {
            ++tests_fail_;
            std::cout << "    => FAILED  (" << te.name << ")\n";
        } else {
            ++tests_pass_;
            std::cout << "    => PASSED\n";
        }
    }

    template <class TMsg> auto on(boost::ut::events::log<TMsg> l) -> void { std::cout << l.msg; }

    template <class TExpr> auto on(boost::ut::events::assertion_pass<TExpr>) -> void
    { ++asserts_pass_; }

    template <class TExpr> auto on(boost::ut::events::assertion_fail<TExpr> a) -> void
    {
        ++asserts_fail_;
        boost::ut::printer p {};
        p << a.expr;
        std::cout << "    [FAIL] " << a.location.file_name() << ':' << a.location.line() << "  ["
                  << p.str() << "]\n";
    }

    auto on(boost::ut::events::exception e) -> void
    {
        ++asserts_fail_;
        std::cout << "    [FAIL] unexpected exception: " << e.what() << '\n';
    }

    auto on(const boost::ut::events::fatal_assertion&) -> void { ++asserts_fail_; }

    auto on(boost::ut::events::summary) -> void
    {
        std::cout << "\n========================================\n"
                  << "tests:   " << (tests_pass_ + tests_fail_) << " total | " << tests_pass_
                  << " passed | " << tests_fail_ << " failed" << (tests_skip_ ? " | " : "")
                  << (tests_skip_ ? std::to_string(tests_skip_) + " skipped" : "")
                  << "\nasserts: " << (asserts_pass_ + asserts_fail_) << " total | "
                  << asserts_pass_ << " passed | " << asserts_fail_ << " failed\n";
        std::cout.flush();
    }

  private:
    std::size_t tests_pass_ {}, tests_fail_ {}, tests_skip_ {};
    std::size_t asserts_pass_ {}, asserts_fail_ {};
    std::size_t fail_at_begin_ {};
};

}  // namespace egg_test

template <class... Ts>
inline auto boost::ut::cfg<boost::ut::override, Ts...> =
  boost::ut::runner<egg_test::verbose_reporter> {};
