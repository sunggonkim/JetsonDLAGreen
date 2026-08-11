#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace jdg {

struct LatencySummary {
  std::size_t count{};
  double mean{};
  double p50{};
  double p95{};
  double p99{};
  double p999{};
  double maximum{};
};

inline double percentile(const std::span<const double> sorted_values,
                         const double quantile) {
  if (sorted_values.empty()) {
    throw std::invalid_argument("percentile requires at least one sample");
  }
  if (quantile < 0.0 || quantile > 1.0 || !std::isfinite(quantile)) {
    throw std::invalid_argument("quantile must be finite and in [0, 1]");
  }

  const double position =
      quantile * static_cast<double>(sorted_values.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted_values[lower] +
         ((sorted_values[upper] - sorted_values[lower]) * fraction);
}

inline LatencySummary summarize(const std::span<const double> values) {
  if (values.empty()) {
    throw std::invalid_argument("latency summary requires at least one sample");
  }

  std::vector<double> sorted(values.begin(), values.end());
  long double total = 0.0L;
  for (const double value : sorted) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument(
          "latency samples must be finite and non-negative");
    }
    total += static_cast<long double>(value);
  }
  std::sort(sorted.begin(), sorted.end());

  return LatencySummary{
      .count = sorted.size(),
      .mean = static_cast<double>(total / sorted.size()),
      .p50 = percentile(sorted, 0.50),
      .p95 = percentile(sorted, 0.95),
      .p99 = percentile(sorted, 0.99),
      .p999 = percentile(sorted, 0.999),
      .maximum = sorted.back(),
  };
}

}  // namespace jdg
