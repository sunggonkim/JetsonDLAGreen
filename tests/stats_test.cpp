#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "jetson_dla_green/stats.hpp"

namespace {

void expect_near(const double actual, const double expected,
                 const std::string_view label) {
  if (std::abs(actual - expected) > 1.0e-12) {
    std::cerr << label << ": expected " << expected << ", got " << actual
              << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <typename Function>
void expect_invalid_argument(Function&& function, const std::string_view label) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  std::cerr << label << ": expected std::invalid_argument\n";
  std::exit(EXIT_FAILURE);
}

}  // namespace

int main() {
  const std::array<double, 5> samples{5.0, 1.0, 4.0, 2.0, 3.0};
  const jdg::LatencySummary summary = jdg::summarize(samples);
  if (summary.count != samples.size()) {
    std::cerr << "sample count mismatch\n";
    return EXIT_FAILURE;
  }
  expect_near(summary.mean, 3.0, "mean");
  expect_near(summary.p50, 3.0, "p50");
  expect_near(summary.p95, 4.8, "p95");
  expect_near(summary.p99, 4.96, "p99");
  expect_near(summary.p999, 4.996, "p999");
  expect_near(summary.maximum, 5.0, "maximum");

  const std::array<double, 1> singleton{7.5};
  expect_near(jdg::summarize(singleton).p99, 7.5, "singleton");

  const std::array<double, 0> empty{};
  expect_invalid_argument([&empty] { static_cast<void>(jdg::summarize(empty)); },
                          "empty samples");
  const std::array<double, 1> invalid{-1.0};
  expect_invalid_argument(
      [&invalid] { static_cast<void>(jdg::summarize(invalid)); },
      "negative sample");
  expect_invalid_argument(
      [&samples] { static_cast<void>(jdg::percentile(samples, 1.1)); },
      "invalid quantile");

  return EXIT_SUCCESS;
}
