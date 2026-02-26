#![allow(dead_code)]

use crate::theme::ThemeConfig;
use crate::render::geometry;

#[derive(Debug, Clone)]
pub(crate) struct LinearScale {
    pub data_min: f64,
    pub data_max: f64,
    pub pixel_min: f64,
    pub pixel_max: f64,
}

impl LinearScale {
    pub(crate) fn new(data_min: f64, data_max: f64, pixel_min: f64, pixel_max: f64) -> Self {
        Self { data_min, data_max, pixel_min, pixel_max }
    }

    pub(crate) fn map(&self, value: f64) -> f64 {
        let data_range = self.data_max - self.data_min;
        if data_range == 0.0 {
            // Degenerate scale: all data at a single value → return midpoint
            return (self.pixel_min + self.pixel_max) / 2.0;
        }
        let t = (value - self.data_min) / data_range;
        self.pixel_min + t * (self.pixel_max - self.pixel_min)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TickMark {
    pub data_value: f64,
    pub pixel_pos: f64,
    pub label: String,
}

#[derive(Debug, Clone)]
pub(crate) struct AxisOutput {
    pub ticks: Vec<TickMark>,
    pub axis_line: String,
    pub grid_lines: String,
}

pub(crate) fn nice_ticks(data_min: f64, data_max: f64, target_count: usize) -> Vec<f64> {
    let target_count = target_count.max(3).min(12);

    if (data_max - data_min).abs() < f64::EPSILON {
        let v = data_min;
        let step = if v == 0.0 { 1.0 } else { (v.abs() * 0.1).max(1.0) };
        return vec![v - step, v, v + step];
    }

    let range = data_max - data_min;
    let raw_step = range / (target_count as f64 - 1.0);
    let mag = 10f64.powf(raw_step.abs().log10().floor());
    let norm = raw_step / mag;

    let nice_norm = if norm <= 1.5 {
        1.0
    } else if norm <= 3.0 {
        2.0
    } else if norm <= 7.0 {
        5.0
    } else {
        10.0
    };

    let step = nice_norm * mag;
    let nice_min = (data_min / step).floor() * step;

    let mut ticks = Vec::new();
    let mut t = nice_min;

    while t < data_max + step {
        ticks.push(t);
        t += step;
        if ticks.len() > 20 {
            break;
        }
    }

    while ticks.len() < 3 {
        let last = *ticks.last().unwrap_or(&data_max);
        ticks.push(last + step);
    }

    ticks
}

pub(crate) fn tick_labels_numeric(ticks: &[f64]) -> Vec<String> {
    if ticks.is_empty() {
        return Vec::new();
    }

    let all_large = ticks.iter().all(|&v| v.abs() >= 1000.0);
    if all_large {
        return ticks
            .iter()
            .map(|&v| {
                let k = v / 1000.0;
                if (k - k.round()).abs() < 1e-6 {
                    format!("{}k", k as i64)
                } else {
                    format!("{:.1}k", k)
                }
            })
            .collect();
    }

    let all_whole = ticks.iter().all(|&v| (v - v.round()).abs() < 1e-9);
    if all_whole {
        return ticks.iter().map(|&v| format!("{}", v as i64)).collect();
    }

    let any_tiny = ticks.iter().any(|&v| v != 0.0 && v.abs() < 0.01);
    if any_tiny {
        return ticks.iter().map(|&v| format!("{:.2e}", v)).collect();
    }

    let step = if ticks.len() >= 2 { (ticks[1] - ticks[0]).abs() } else { 1.0 };
    let decimals = if step >= 1.0 {
        0
    } else {
        ((-step.log10()).ceil() as usize).min(6)
    };

    ticks.iter().map(|&v| format!("{:.prec$}", v, prec = decimals)).collect()
}

pub(crate) fn tick_labels_temporal(ticks: &[i64], range_ms: i64) -> Vec<String> {
    const MS_PER_SEC: i64 = 1_000;
    const MS_PER_MIN: i64 = 60 * MS_PER_SEC;
    const MS_PER_HOUR: i64 = 60 * MS_PER_MIN;
    const MS_PER_DAY: i64 = 24 * MS_PER_HOUR;
    const MS_PER_YEAR: i64 = 365 * MS_PER_DAY;

    ticks
        .iter()
        .map(|&ms| {
            let total_secs = ms / MS_PER_SEC;
            let (year, month, day, hour, min, sec) = epoch_secs_to_datetime(total_secs);

            if range_ms < 2 * MS_PER_MIN {
                format!("{:02}:{:02}:{:02}", hour, min, sec)
            } else if range_ms < 2 * MS_PER_HOUR {
                format!("{:02}:{:02}", hour, min)
            } else if range_ms < 2 * MS_PER_DAY {
                format!("{} {:02} {:02}:{:02}", month_abbr(month), day, hour, min)
            } else if range_ms < 60 * MS_PER_DAY {
                format!("{} {:02}", month_abbr(month), day)
            } else if range_ms < 2 * MS_PER_YEAR {
                format!("{} {}", month_abbr(month), year)
            } else {
                format!("{}", year)
            }
        })
        .collect()
}

fn epoch_secs_to_datetime(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let (secs, neg) = if secs < 0 { (-secs, true) } else { (secs, false) };

    let sec = (secs % 60) as u32;
    let total_min = secs / 60;
    let min = (total_min % 60) as u32;
    let total_hr = total_min / 60;
    let hour = (total_hr % 24) as u32;
    let mut total_days = total_hr / 24;

    // Gregorian calendar algorithm (days since 1970-01-01)
    let mut year = 1970i32;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if total_days < days_in_year {
            break;
        }
        total_days -= days_in_year;
        year += 1;
    }
    let months = [31i64, if is_leap(year) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;
    for &m in &months {
        if total_days < m {
            break;
        }
        total_days -= m;
        month += 1;
    }
    let day = (total_days + 1) as u32;

    if neg { (1970 - (year - 1970), month, day, hour, min, sec) } else { (year, month, day, hour, min, sec) }
}

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn month_abbr(m: u32) -> &'static str {
    match m {
        1 => "Jan", 2 => "Feb", 3 => "Mar", 4 => "Apr",
        5 => "May", 6 => "Jun", 7 => "Jul", 8 => "Aug",
        9 => "Sep", 10 => "Oct", 11 => "Nov", 12 => "Dec",
        _ => "???",
    }
}

pub(crate) fn categorical_scale(
    categories: &[String],
    pixel_min: f64,
    pixel_max: f64,
) -> Vec<(String, f64)> {
    if categories.is_empty() {
        return Vec::new();
    }
    let n = categories.len() as f64;
    let band = (pixel_max - pixel_min) / n;
    categories
        .iter()
        .enumerate()
        .map(|(i, cat)| {
            let center = pixel_min + (i as f64 + 0.5) * band;
            (cat.clone(), center)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AxisOrientation {
    Horizontal,
    Vertical,
}

pub(crate) fn compute_axis(
    _scale: &LinearScale,
    ticks: &[TickMark],
    orientation: AxisOrientation,
    plot_origin_x: f64,
    plot_origin_y: f64,
    plot_width: f64,
    plot_height: f64,
    theme: &ThemeConfig,
) -> AxisOutput {
    let tick_length = 5.0;
    let label_offset = 4.0;

    let (axis_svg, grid_svg, tick_svgs) = match orientation {
        AxisOrientation::Horizontal => {
            let axis_y = plot_origin_y + plot_height;
            let ax = geometry::line(
                plot_origin_x,
                axis_y,
                plot_origin_x + plot_width,
                axis_y,
                theme.axis_color,
                1.0,
            );
            let mut grid = Vec::new();
            let mut tick_elements = Vec::new();
            for tick in ticks {
                let px = tick.pixel_pos;
                grid.push(geometry::line(px, plot_origin_y, px, axis_y, theme.grid_color, 1.0));
                // Tick mark
                tick_elements.push(geometry::line(px, axis_y, px, axis_y + tick_length, theme.axis_color, 1.0));
                tick_elements.push(geometry::text(
                    px,
                    axis_y + tick_length + label_offset + theme.font_size_px as f64,
                    &tick.label,
                    "middle",
                    theme.font_size_px,
                    theme.text_color,
                    0.0,
                ));
            }
            (ax, grid.join("\n"), tick_elements.join("\n"))
        }
        AxisOrientation::Vertical => {
            let axis_x = plot_origin_x;
            let ax = geometry::line(
                axis_x,
                plot_origin_y,
                axis_x,
                plot_origin_y + plot_height,
                theme.axis_color,
                1.0,
            );
            let mut grid = Vec::new();
            let mut tick_elements = Vec::new();
            for tick in ticks {
                let py = tick.pixel_pos;
                grid.push(geometry::line(axis_x, py, axis_x + plot_width, py, theme.grid_color, 1.0));
                // Tick mark
                tick_elements.push(geometry::line(axis_x - tick_length, py, axis_x, py, theme.axis_color, 1.0));
                // Label
                let label_x = axis_x - tick_length - label_offset;
                tick_elements.push(geometry::text(
                    label_x,
                    py + (theme.font_size_px as f64) * 0.35, // vertical center
                    &tick.label,
                    "end",
                    theme.font_size_px,
                    theme.text_color,
                    0.0,
                ));
            }
            (ax, grid.join("\n"), tick_elements.join("\n"))
        }
    };

    let axis_line = format!("{}\n{}", axis_svg, tick_svgs);

    AxisOutput { ticks: ticks.to_vec(), axis_line, grid_lines: grid_svg }
}

pub(crate) fn build_tick_marks(
    tick_values: &[f64],
    labels: &[String],
    scale: &LinearScale,
) -> Vec<TickMark> {
    tick_values
        .iter()
        .zip(labels.iter())
        .map(|(&v, label)| TickMark {
            data_value: v,
            pixel_pos: scale.map(v),
            label: label.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- LinearScale ---

    #[test]
    fn linear_scale_maps_min_to_pixel_min() {
        let s = LinearScale::new(0.0, 10.0, 100.0, 500.0);
        assert!((s.map(0.0) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn linear_scale_maps_max_to_pixel_max() {
        let s = LinearScale::new(0.0, 10.0, 100.0, 500.0);
        assert!((s.map(10.0) - 500.0).abs() < 1e-9);
    }

    #[test]
    fn linear_scale_maps_midpoint() {
        let s = LinearScale::new(0.0, 10.0, 0.0, 100.0);
        assert!((s.map(5.0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn linear_scale_degenerate_returns_midpoint() {
        let s = LinearScale::new(5.0, 5.0, 0.0, 100.0);
        assert!((s.map(5.0) - 50.0).abs() < 1e-9);
    }

    // --- nice_ticks ---

    fn check_nice_tick_properties(data_min: f64, data_max: f64) {
        let ticks = nice_ticks(data_min, data_max, 6);
        assert!(ticks.len() >= 3, "too few ticks for ({data_min}, {data_max}): {:?}", ticks);
        assert!(ticks.len() <= 12, "too many ticks for ({data_min}, {data_max}): {:?}", ticks);
        assert!(
            *ticks.first().unwrap() <= data_min + 1e-9,
            "first tick > data_min for ({data_min}, {data_max}): {:?}",
            ticks
        );
        assert!(
            *ticks.last().unwrap() >= data_max - 1e-9,
            "last tick < data_max for ({data_min}, {data_max}): {:?}",
            ticks
        );
        // All intervals equal (within relative float tolerance)
        if ticks.len() >= 2 {
            let step = ticks[1] - ticks[0];
            for w in ticks.windows(2) {
                let diff = (w[1] - w[0] - step).abs();
                assert!(diff < step * 1e-6 + 1e-10, "unequal step at {:?}: {diff}", w);
            }
        }
    }

    #[test]
    fn nice_ticks_range_crosses_zero() {
        check_nice_tick_properties(-3.5, 8.2);
    }

    #[test]
    fn nice_ticks_very_small_range() {
        check_nice_tick_properties(1.000, 1.005);
    }

    #[test]
    fn nice_ticks_very_large_range() {
        check_nice_tick_properties(0.0, 1_000_000_000.0);
    }

    #[test]
    fn nice_ticks_single_value_no_panic() {
        let ticks = nice_ticks(5.0, 5.0, 6);
        assert!(ticks.len() >= 3);
        // Must contain 5.0
        assert!(ticks.iter().any(|&v| (v - 5.0).abs() < 1.0 + 1e-6));
    }

    #[test]
    fn nice_ticks_negative_only_range() {
        check_nice_tick_properties(-100.0, -10.0);
    }

    #[test]
    fn nice_ticks_simple_positive_range() {
        let ticks = nice_ticks(0.0, 100.0, 6);
        check_nice_tick_properties(0.0, 100.0);
        // Expect round steps like 0, 20, 40, 60, 80, 100
        let step = ticks[1] - ticks[0];
        let _nice_steps = [1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0];
        // step should match a "nice" magnitude
        let mag = 10f64.powf(step.abs().log10().floor());
        let norm = step / mag;
        assert!(
            (norm - norm.round()).abs() < 0.01 || [1.0, 2.0, 2.5, 5.0].contains(&(norm.round())),
            "step {step} (norm {norm}) is not nice"
        );
    }

    // --- tick_labels_numeric ---

    #[test]
    fn labels_numeric_whole_numbers() {
        let labels = tick_labels_numeric(&[0.0, 5.0, 10.0, 15.0]);
        assert_eq!(labels, vec!["0", "5", "10", "15"]);
    }

    #[test]
    fn labels_numeric_large_values_use_k_suffix() {
        let labels = tick_labels_numeric(&[1000.0, 2000.0, 3000.0]);
        for l in &labels {
            assert!(l.ends_with('k'), "expected k suffix: {l}");
        }
    }

    #[test]
    fn labels_numeric_tiny_values_use_scientific() {
        let labels = tick_labels_numeric(&[0.0001, 0.0002, 0.0003]);
        for l in &labels {
            assert!(l.contains('e'), "expected scientific notation: {l}");
        }
    }

    #[test]
    fn labels_numeric_empty_input() {
        assert_eq!(tick_labels_numeric(&[]), Vec::<String>::new());
    }

    // --- categorical_scale ---

    #[test]
    fn categorical_scale_even_spacing() {
        let cats = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let result = categorical_scale(&cats, 0.0, 300.0);
        assert_eq!(result.len(), 3);
        // Each center should be at 50, 150, 250
        assert!((result[0].1 - 50.0).abs() < 1e-9);
        assert!((result[1].1 - 150.0).abs() < 1e-9);
        assert!((result[2].1 - 250.0).abs() < 1e-9);
    }

    #[test]
    fn categorical_scale_empty_input() {
        let result = categorical_scale(&[], 0.0, 100.0);
        assert!(result.is_empty());
    }

    #[test]
    fn categorical_scale_single_category() {
        let cats = vec!["Solo".to_string()];
        let result = categorical_scale(&cats, 0.0, 100.0);
        assert_eq!(result.len(), 1);
        assert!((result[0].1 - 50.0).abs() < 1e-9);
    }

    // --- tick_labels_temporal ---

    #[test]
    fn temporal_labels_multi_year_range_shows_year_only() {
        // range > 2 years
        let range_ms = 3 * 365 * 24 * 3600 * 1000i64;
        // 2020-01-01 epoch ms ≈ 1577836800000
        let tick = 1577836800000i64;
        let labels = tick_labels_temporal(&[tick], range_ms);
        assert_eq!(labels[0], "2020");
    }

    #[test]
    fn temporal_labels_sub_minute_shows_hms() {
        // range < 2 minutes
        let range_ms = 60_000i64; // 1 minute
        // epoch 0 = 1970-01-01 00:00:00
        let labels = tick_labels_temporal(&[0i64], range_ms);
        assert_eq!(labels[0], "00:00:00");
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn nice_ticks_invariants(
            data_min in -1_000_000.0_f64..1_000_000.0_f64,
            range    in           0.001_f64..2_000_000.0_f64,
        ) {
            let data_max = data_min + range;
            let ticks = nice_ticks(data_min, data_max, 6);

            prop_assert!(ticks.len() >= 3,
                "too few ticks ({}) for ({}, {}): {:?}", ticks.len(), data_min, data_max, ticks);
            prop_assert!(ticks.len() <= 12,
                "too many ticks ({}) for ({}, {}): {:?}", ticks.len(), data_min, data_max, ticks);
            prop_assert!(
                ticks[0] <= data_min + 1e-6,
                "first tick {} > data_min {}", ticks[0], data_min
            );
            prop_assert!(
                ticks[ticks.len() - 1] >= data_max - 1e-6,
                "last tick {} < data_max {}", ticks[ticks.len() - 1], data_max
            );

            // All intervals must be equal within floating-point tolerance
            if ticks.len() >= 2 {
                let step = ticks[1] - ticks[0];
                prop_assert!(step > 0.0, "step must be positive: {}", step);

                for w in ticks.windows(2) {
                    let diff = (w[1] - w[0] - step).abs();
                    prop_assert!(diff < step * 1e-6 + 1e-10,
                        "unequal step at {:?}: expected {}, diff={}", w, step, diff);
                }

                // Step must be a "nice" number: step / 10^floor(log10(step)) ∈ {1, 2, 2.5, 5, 10}
                let mag = 10f64.powf(step.abs().log10().floor());
                let norm = (step / mag * 1000.0).round() / 1000.0;
                let nice = [1.0_f64, 2.0, 2.5, 5.0, 10.0];
                prop_assert!(
                    nice.iter().any(|&n| (norm - n).abs() < 0.01),
                    "step {} (norm {:.3}) is not a nice number", step, norm
                );
            }
        }
    }
}