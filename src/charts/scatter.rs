use std::collections::HashMap;

use polars::frame::DataFrame;

use crate::charts::Chart;
use crate::dtype::{classify_column, VizDtype};
use crate::error::{CharcoalError, CharcoalWarning};
use crate::normalize::{to_epoch_ms, to_f64};
use crate::render::{
    Margin, SvgCanvas,
    axes::{
        AxisOrientation, LinearScale,
        build_tick_marks, compute_axis, nice_ticks,
        tick_labels_numeric, tick_labels_temporal,
    },
    geometry,
};
use crate::theme::{Theme, ThemeConfig};

pub(crate) const CANVAS_WIDTH:  u32   = 800;
pub(crate) const CANVAS_HEIGHT: u32   = 500;
const SUBSAMPLE_THRESHOLD:      usize = 500_000;
const DEFAULT_POINT_SIZE:       f64   = 6.0;
const DEFAULT_ROW_LIMIT:        usize = 1_000_000;
/// Colour used for the `null` category in `color_by` legends.
pub(crate) const NULL_COLOR: &str = "#AAAAAA";

#[derive(Clone)]
pub(crate) struct ScatterConfig {
    pub x_col:      Option<String>,
    pub y_col:      Option<String>,
    pub color_by:   Option<String>,
    pub size_by:    Option<String>,
    pub title:      Option<String>,
    pub x_label:    Option<String>,
    pub y_label:    Option<String>,
    pub theme:      Theme,
    pub opacity:    f64,
    pub point_size: f64,
    pub row_limit:  usize,
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            x_col:      None,
            y_col:      None,
            color_by:   None,
            size_by:    None,
            title:      None,
            x_label:    None,
            y_label:    None,
            theme:      Theme::Default,
            opacity:    1.0,
            point_size: DEFAULT_POINT_SIZE,
            row_limit:  DEFAULT_ROW_LIMIT,
        }
    }
}

pub struct ScatterBuilder<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: ScatterConfig,
}

pub struct ScatterWithX<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: ScatterConfig,
}

pub struct ScatterWithXY<'df> {
    pub(crate) df:     &'df DataFrame,
    pub(crate) config: ScatterConfig,
}

impl<'df> ScatterBuilder<'df> {
    pub(crate) fn new(df: &'df DataFrame) -> Self {
        Self { df, config: ScatterConfig::default() }
    }
}

macro_rules! impl_scatter_optional_setters {
    ($t:ty) => {
        impl<'df> $t {
            pub fn color_by(mut self, col: &str) -> Self {
                self.config.color_by = Some(col.to_string()); self
            }
            pub fn size_by(mut self, col: &str) -> Self {
                self.config.size_by = Some(col.to_string()); self
            }
            pub fn title(mut self, title: &str) -> Self {
                self.config.title = Some(title.to_string()); self
            }
            pub fn x_label(mut self, label: &str) -> Self {
                self.config.x_label = Some(label.to_string()); self
            }
            pub fn y_label(mut self, label: &str) -> Self {
                self.config.y_label = Some(label.to_string()); self
            }
            /// Visual theme. Defaults to [`Theme::Default`].
            pub fn theme(mut self, theme: Theme) -> Self {
                self.config.theme = theme; self
            }
            /// Point opacity in `[0.0, 1.0]`. Defaults to `1.0`.
            pub fn opacity(mut self, opacity: f64) -> Self {
                self.config.opacity = opacity.clamp(0.0, 1.0); self
            }
            /// Default point radius in pixels. Overridden per-point by `size_by`.
            pub fn point_size(mut self, px: f64) -> Self {
                self.config.point_size = px.max(0.5); self
            }
            /// Row limit above which `.build()` returns [`CharcoalError::DataTooLarge`].
            /// Default: 1,000,000.
            pub fn row_limit(mut self, limit: usize) -> Self {
                self.config.row_limit = limit; self
            }
        }
    };
}

impl_scatter_optional_setters!(ScatterBuilder<'df>);
impl_scatter_optional_setters!(ScatterWithX<'df>);
impl_scatter_optional_setters!(ScatterWithXY<'df>);

// ---------------------------------------------------------------------------
// Required-field transitions
// ---------------------------------------------------------------------------

impl<'df> ScatterBuilder<'df> {
    /// Set the x-axis column. Validated at `.build()`, not here.
    pub fn x(mut self, col: &str) -> ScatterWithX<'df> {
        self.config.x_col = Some(col.to_string());
        ScatterWithX { df: self.df, config: self.config }
    }
}

impl<'df> ScatterWithX<'df> {
    /// Set the y-axis column. Validated at `.build()`, not here.
    pub fn y(mut self, col: &str) -> ScatterWithXY<'df> {
        self.config.y_col = Some(col.to_string());
        ScatterWithXY { df: self.df, config: self.config }
    }
}

// ---------------------------------------------------------------------------
// Intermediate types produced by the free functions
// ---------------------------------------------------------------------------

/// A single scatter point ready for scale mapping.
struct Point {
    x:     f64,
    y:     f64,
    color: Option<String>,
    size:  f64,
}

/// Everything extracted from the DataFrame and config.
/// Produced once by `load_columns`; consumed by all downstream steps.
pub(crate) struct LoadedColumns {
    pub x_vals:        Vec<Option<f64>>,
    pub y_vals:        Vec<Option<f64>>,
    pub color_vals:    Option<Vec<Option<String>>>,
    pub size_vals:     Option<Vec<Option<f64>>>,
    pub x_is_temporal: bool,
    pub warnings:      Vec<CharcoalWarning>,
}

// ---------------------------------------------------------------------------
// Free functions — each owns exactly one job
// ---------------------------------------------------------------------------

/// Validates the row limit, classifies and normalises x and y, and loads the
/// optional `color_by` and `size_by` columns.
///
/// This is the only function that reads from `df` and `config`; everything
/// downstream works from the returned [`LoadedColumns`].
pub(crate) fn load_columns(
    df:     &DataFrame,
    config: &ScatterConfig,
) -> Result<LoadedColumns, CharcoalError> {
    let mut warnings = Vec::new();

    // --- Row limit ---
    let n_rows = df.height();
    if n_rows > config.row_limit {
        return Err(CharcoalError::DataTooLarge {
            rows:    n_rows,
            limit:   config.row_limit,
            message: format!(
                "DataFrame exceeds the {0} row render limit. \
                 Consider df.sample({1}) or an aggregation before charting.",
                config.row_limit,
                config.row_limit / 2,
            ),
        });
    }

    // --- X axis ---
    let x_col = config.x_col.as_deref().unwrap(); // guaranteed by typestate
    let x_viz = classify_column(df, x_col, None)?;
    if x_viz == VizDtype::Categorical {
        return Err(CharcoalError::UnsupportedColumn {
            col:     x_col.to_string(),
            dtype:   df.schema().get(x_col).unwrap().clone(),
            message: "Categorical columns cannot be used as the x-axis of a scatter chart. \
                      Use a numeric or temporal column instead.".to_string(),
        });
    }
    let x_is_temporal = x_viz == VizDtype::Temporal;
    let (x_vals, x_warns) = col_to_f64(df, x_col, x_is_temporal)?;
    warnings.extend(x_warns);

    // --- Y axis ---
    let y_col = config.y_col.as_deref().unwrap();
    let y_viz = classify_column(df, y_col, None)?;
    if y_viz == VizDtype::Categorical {
        return Err(CharcoalError::UnsupportedColumn {
            col:     y_col.to_string(),
            dtype:   df.schema().get(y_col).unwrap().clone(),
            message: "Categorical columns cannot be used as the y-axis of a scatter chart. \
                      Use a numeric or temporal column instead.".to_string(),
        });
    }
    let (y_vals, y_warns) = col_to_f64(df, y_col, y_viz == VizDtype::Temporal)?;
    warnings.extend(y_warns);

    // --- color_by (any dtype → String) ---
    let color_vals: Option<Vec<Option<String>>> = match &config.color_by {
        None => None,
        Some(col) => {
            classify_column(df, col, None)?; // just validates existence + not Unsupported
            let series  = df.column(col).map_err(|e| CharcoalError::RenderError(e.to_string()))?;
            let casted  = series
                .cast(&polars::datatypes::DataType::String)
                .map_err(|e| CharcoalError::RenderError(e.to_string()))?;
            let chunked = casted.str().map_err(|e| CharcoalError::RenderError(e.to_string()))?;
            Some(chunked.into_iter().map(|v| v.map(|s| s.to_string())).collect())
        }
    };

    // --- size_by (must be numeric) ---
    let size_vals: Option<Vec<Option<f64>>> = match &config.size_by {
        None => None,
        Some(col) => {
            classify_column(df, col, Some(VizDtype::Numeric))?;
            let (vals, w) = to_f64(df, col)?;
            warnings.extend(w);
            Some(vals)
        }
    };

    Ok(LoadedColumns { x_vals, y_vals, color_vals, size_vals, x_is_temporal, warnings })
}

/// Normalises a column to `Vec<Option<f64>>`, converting temporal → epoch-ms when needed.
fn col_to_f64(
    df: &DataFrame,
    col: &str,
    is_temporal: bool,
) -> Result<(Vec<Option<f64>>, Vec<CharcoalWarning>), CharcoalError> {
    if is_temporal {
        let (ms, w) = to_epoch_ms(df, col)?;
        Ok((ms.iter().map(|v| v.map(|i| i as f64)).collect(), w))
    } else {
        to_f64(df, col)
    }
}

/// Returns the row indices to render; emits `RowsSubsampled` when the dataset
/// exceeds 500 k rows.
fn subsample(n_rows: usize, warnings: &mut Vec<CharcoalWarning>) -> Vec<usize> {
    if n_rows > SUBSAMPLE_THRESHOLD {
        let step     = (n_rows as f64 / SUBSAMPLE_THRESHOLD as f64).ceil() as usize;
        let sampled: Vec<usize> = (0..n_rows).step_by(step).collect();
        let rendered = sampled.len();
        warnings.push(CharcoalWarning::RowsSubsampled { original: n_rows, rendered });
        sampled
    } else {
        (0..n_rows).collect()
    }
}

/// Zips the loaded column slices into `Point` values, skipping any row where
/// x or y is null.  Returns the point list and a null count for each axis.
fn build_points(
    cols:         &LoadedColumns,
    indices:      &[usize],
    default_size: f64,
) -> (Vec<Point>, usize, usize) {
    let mut points  = Vec::with_capacity(indices.len());
    let mut x_nulls = 0usize;
    let mut y_nulls = 0usize;

    for &i in indices {
        match (cols.x_vals[i], cols.y_vals[i]) {
            (None, _)           => x_nulls += 1,
            (_, None)           => y_nulls += 1,
            (Some(x), Some(y))  => {
                let color = cols.color_vals.as_ref().and_then(|cv| cv[i].clone());
                let size  = cols.size_vals.as_ref().and_then(|sv| sv[i]).unwrap_or(default_size);
                points.push(Point { x, y, color, size });
            }
        }
    }
    (points, x_nulls, y_nulls)
}

/// Assigns a palette hex colour to each unique category in first-seen order.
///
/// Returned map is passed to both `make_elements` (for point fill) and
/// `make_legend` (for the legend swatch).
pub(crate) fn resolve_colors(
    color_vals: &Option<Vec<Option<String>>>,
    theme:      &ThemeConfig,
) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let cv = match color_vals { None => return map, Some(v) => v };
    let mut idx = 0usize;
    for cat in cv.iter().flatten() {
        if !map.contains_key(cat) {
            map.insert(cat.clone(), theme.palette[idx % theme.palette.len()].to_string());
            idx += 1;
        }
    }
    map
}

/// Maps each `Point` through the supplied scales to produce an SVG `<circle>`.
fn make_elements(
    points:          &[Point],
    category_colors: &HashMap<String, String>,
    x_scale:         &LinearScale,
    y_scale:         &LinearScale,
    opacity:         f64,
    default_color:   &str,
) -> Vec<String> {
    points.iter().map(|p| {
        let fill = match &p.color {
            None      => default_color,
            Some(cat) => category_colors.get(cat).map(|s| s.as_str()).unwrap_or(NULL_COLOR),
        };
        geometry::circle(x_scale.map(p.x), y_scale.map(p.y), p.size / 2.0, fill, opacity)
    }).collect()
}

/// Builds the ordered `(label, hex)` legend pairs from the raw category column.
///
/// Iterates `color_vals` once so insertion order is preserved and the `null`
/// category appears wherever its first occurrence falls in the data.
pub(crate) fn make_legend(
    color_vals:      &Option<Vec<Option<String>>>,
    category_colors: &HashMap<String, String>,
) -> Option<Vec<(String, String)>> {
    let cv = color_vals.as_ref()?;
    if category_colors.is_empty() { return None; }

    let mut seen    = Vec::<String>::new();
    let mut entries = Vec::<(String, String)>::new();

    for v in cv {
        match v {
            None => {
                if !seen.contains(&"null".to_string()) {
                    seen.push("null".to_string());
                    entries.push(("null".to_string(), NULL_COLOR.to_string()));
                }
            }
            Some(cat) => {
                if !seen.contains(cat) {
                    seen.push(cat.clone());
                    if let Some(color) = category_colors.get(cat) {
                        entries.push((cat.clone(), color.clone()));
                    }
                }
            }
        }
    }
    if entries.is_empty() { None } else { Some(entries) }
}

// ---------------------------------------------------------------------------
// ScatterWithXY — the terminal builder state
// ---------------------------------------------------------------------------

impl<'df> ScatterWithXY<'df> {
    /// Validate all inputs, render the chart, and return a [`Chart`].
    ///
    /// # Errors
    ///
    /// | Variant | Condition |
    /// |---|---|
    /// | [`CharcoalError::DataTooLarge`]     | `df.height() > row_limit` |
    /// | [`CharcoalError::ColumnNotFound`]   | Any column name does not exist |
    /// | [`CharcoalError::UnsupportedColumn`]| x or y is a Categorical column |
    pub fn build(self) -> Result<Chart, CharcoalError> {
        let x_col  = self.config.x_col.clone().unwrap();
        let y_col  = self.config.y_col.clone().unwrap();
        let config = self.config.clone();

        // --- Data phase -------------------------------------------------
        let mut cols     = load_columns(self.df, &config)?;
        let     n_rows   = self.df.height();
        let     indices  = subsample(n_rows, &mut cols.warnings);

        let (points, x_nulls, y_nulls) = build_points(&cols, &indices, config.point_size);
        if x_nulls > 0 { cols.warnings.push(CharcoalWarning::NullsSkipped { col: x_col.clone(), count: x_nulls }); }
        if y_nulls > 0 { cols.warnings.push(CharcoalWarning::NullsSkipped { col: y_col.clone(), count: y_nulls }); }

        let theme_cfg       = ThemeConfig::from(&config.theme);
        let category_colors = resolve_colors(&cols.color_vals, &theme_cfg);

        // --- Scale phase ------------------------------------------------
        let (x_min, x_max) = data_range(points.iter().map(|p| p.x));
        let (y_min, y_max) = data_range(points.iter().map(|p| p.y));
        let x_tick_vals    = nice_ticks(x_min, x_max, 6);
        let y_tick_vals    = nice_ticks(y_min, y_max, 6);

        let canvas = SvgCanvas::new(CANVAS_WIDTH, CANVAS_HEIGHT, Margin::default_chart(), theme_cfg.clone());
        let ox = canvas.plot_origin_x();
        let oy = canvas.plot_origin_y();
        let pw = canvas.plot_width();
        let ph = canvas.plot_height();

        let x_scale = LinearScale::new(*x_tick_vals.first().unwrap(), *x_tick_vals.last().unwrap(), ox, ox + pw);
        let y_scale = LinearScale::new(*y_tick_vals.first().unwrap(), *y_tick_vals.last().unwrap(), oy + ph, oy);

        // --- Render phase -----------------------------------------------
        let elements = make_elements(&points, &category_colors, &x_scale, &y_scale, config.opacity, theme_cfg.palette[0]);
        let legend   = make_legend(&cols.color_vals, &category_colors);

        let x_labels = if cols.x_is_temporal {
            let range_ms    = (x_max - x_min) as i64;
            let x_tick_i64: Vec<i64> = x_tick_vals.iter().map(|&v| v as i64).collect();
            tick_labels_temporal(&x_tick_i64, range_ms)
        } else {
            tick_labels_numeric(&x_tick_vals)
        };
        let y_labels  = tick_labels_numeric(&y_tick_vals);
        let x_tmarks  = build_tick_marks(&x_tick_vals, &x_labels, &x_scale);
        let y_tmarks  = build_tick_marks(&y_tick_vals, &y_labels, &y_scale);
        let x_axis    = compute_axis(&x_scale, &x_tmarks, AxisOrientation::Horizontal, ox, oy, pw, ph, &theme_cfg);
        let y_axis    = compute_axis(&y_scale, &y_tmarks, AxisOrientation::Vertical,   ox, oy, pw, ph, &theme_cfg);

        // --- Assemble ---------------------------------------------------
        let title   = config.title.as_deref().unwrap_or("");
        let x_label = config.x_label.as_deref().unwrap_or(&x_col);
        let y_label = config.y_label.as_deref().unwrap_or(&y_col);
        let svg     = canvas.render(elements, x_axis, y_axis, title, x_label, y_label, legend);

        Ok(Chart { svg, warnings: cols.warnings, title: title.to_string(), width: CANVAS_WIDTH, height: CANVAS_HEIGHT })
    }
}

/// Returns `(min, max)` over the iterator, falling back to `(0.0, 1.0)` when
/// the iterator is empty (e.g. all rows were nulls).
fn data_range(iter: impl Iterator<Item = f64>) -> (f64, f64) {
    let (lo, hi) = iter.fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(lo, hi), v| (lo.min(v), hi.max(v)),
    );
    if lo.is_infinite() { (0.0, 1.0) } else { (lo, hi) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn empty_df() -> DataFrame { DataFrame::empty() }

    fn iris_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("sepal_length", &[5.1f64, 4.9, 4.7, 6.4, 6.3]),
            Series::new("sepal_width",  &[3.5f64, 3.0, 3.2, 3.2, 3.3]),
            Series::new("petal_length", &[1.4f64, 1.4, 1.3, 4.5, 6.0]),
            Series::new("species",      &["setosa", "setosa", "setosa", "versicolor", "virginica"]),
        ]).unwrap()
    }

    // ── Config defaults ──────────────────────────────────────────────────────

    #[test]
    fn defaults_are_sane() {
        let df = empty_df();
        let b  = ScatterBuilder::new(&df);
        assert!(b.config.color_by.is_none());
        assert!(b.config.size_by.is_none());
        assert!(b.config.title.is_none());
        assert!(matches!(b.config.theme, Theme::Default));
        assert!((b.config.opacity    - 1.0).abs()               < f64::EPSILON);
        assert!((b.config.point_size - DEFAULT_POINT_SIZE).abs() < f64::EPSILON);
        assert_eq!(b.config.row_limit, DEFAULT_ROW_LIMIT);
    }

    #[test]
    fn color_by_stores_col() {
        let df = empty_df();
        assert_eq!(ScatterBuilder::new(&df).color_by("species").config.color_by.as_deref(), Some("species"));
    }

    #[test]
    fn size_by_stores_col() {
        let df = empty_df();
        assert_eq!(ScatterBuilder::new(&df).size_by("petal_length").config.size_by.as_deref(), Some("petal_length"));
    }

    #[test]
    fn title_stores_string() {
        let df = empty_df();
        assert_eq!(ScatterBuilder::new(&df).title("Iris").config.title.as_deref(), Some("Iris"));
    }

    #[test]
    fn opacity_clamps_above_one() {
        let df = empty_df();
        assert!((ScatterBuilder::new(&df).opacity(1.5).config.opacity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn opacity_clamps_below_zero() {
        let df = empty_df();
        assert!((ScatterBuilder::new(&df).opacity(-0.3).config.opacity).abs() < f64::EPSILON);
    }

    #[test]
    fn point_size_clamps_to_minimum() {
        let df = empty_df();
        assert!(ScatterBuilder::new(&df).point_size(0.0).config.point_size >= 0.5);
    }

    #[test]
    fn row_limit_stores_value() {
        let df = empty_df();
        assert_eq!(ScatterBuilder::new(&df).row_limit(500_000).config.row_limit, 500_000);
    }

    // ── Setters survive state transitions ────────────────────────────────────

    #[test]
    fn setters_available_on_scatter_with_x() {
        let df = empty_df();
        let s  = ScatterWithX {
            df: &df,
            config: ScatterConfig { x_col: Some("x".to_string()), ..Default::default() },
        };
        let s = s.color_by("c").title("T").opacity(0.5);
        assert_eq!(s.config.color_by.as_deref(), Some("c"));
        assert_eq!(s.config.title.as_deref(),    Some("T"));
        assert!((s.config.opacity - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn setters_available_on_scatter_with_xy() {
        let df = empty_df();
        let s  = ScatterWithXY {
            df: &df,
            config: ScatterConfig {
                x_col: Some("x".to_string()),
                y_col: Some("y".to_string()),
                ..Default::default()
            },
        };
        let s = s.theme(Theme::Colorblind).point_size(8.0).row_limit(200_000);
        assert!(matches!(s.config.theme, Theme::Colorblind));
        assert!((s.config.point_size - 8.0).abs() < f64::EPSILON);
        assert_eq!(s.config.row_limit, 200_000);
    }

    #[test]
    fn chained_setters_return_same_df_reference() {
        let df = empty_df();
        let b  = ScatterBuilder::new(&df)
            .color_by("c").size_by("s").title("T")
            .theme(Theme::Minimal).opacity(0.8).point_size(5.0).row_limit(100_000);
        assert!(std::ptr::eq(b.df, &df));
    }

    #[test]
    fn optional_setters_survive_x_and_y_transitions() {
        let df = iris_df();
        let b  = ScatterBuilder::new(&df)
            .title("Iris")
            .x("sepal_length")
            .color_by("species")
            .y("sepal_width")
            .opacity(0.8);
        assert_eq!(b.config.title.as_deref(),    Some("Iris"));
        assert_eq!(b.config.color_by.as_deref(), Some("species"));
        assert!((b.config.opacity - 0.8).abs() < f64::EPSILON);
    }

    // ── .x() / .y() transitions ──────────────────────────────────────────────

    #[test]
    fn x_stores_col() {
        let df = iris_df();
        let b  = ScatterBuilder::new(&df).x("sepal_length");
        assert_eq!(b.config.x_col.as_deref(), Some("sepal_length"));
    }

    #[test]
    fn y_stores_col() {
        let df = iris_df();
        let b  = ScatterBuilder::new(&df).x("sepal_length").y("sepal_width");
        assert_eq!(b.config.x_col.as_deref(), Some("sepal_length"));
        assert_eq!(b.config.y_col.as_deref(), Some("sepal_width"));
    }

    // ── build() happy path ────────────────────────────────────────────────────

    #[test]
    fn build_produces_valid_svg_with_circles() {
        let df    = iris_df();
        let chart = ScatterBuilder::new(&df).x("sepal_length").y("sepal_width").build().unwrap();
        assert!(chart.svg().contains("<svg"),    "output must be SVG");
        assert!(chart.svg().contains("<circle"), "scatter must contain circles");
    }

    #[test]
    fn build_produces_correct_dimensions() {
        let df    = iris_df();
        let chart = ScatterBuilder::new(&df).x("sepal_length").y("sepal_width").build().unwrap();
        assert_eq!(chart.width(),  CANVAS_WIDTH);
        assert_eq!(chart.height(), CANVAS_HEIGHT);
    }

    #[test]
    fn build_single_row_does_not_panic() {
        let df = DataFrame::new(vec![
            Series::new("x", &[1.0f64]),
            Series::new("y", &[2.0f64]),
        ]).unwrap();
        ScatterBuilder::new(&df).x("x").y("y").build().expect("single row must build");
    }

    // ── build() error paths ───────────────────────────────────────────────────

    #[test]
    fn typo_in_x_gives_column_not_found_with_suggestion() {
        let df  = iris_df();
        let err = ScatterBuilder::new(&df).x("sepal_lenght").y("sepal_width").build().unwrap_err();
        match err {
            CharcoalError::ColumnNotFound { name, suggestion, .. } => {
                assert_eq!(name,       "sepal_lenght");
                assert_eq!(suggestion, "sepal_length");
            }
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }

    #[test]
    fn typo_in_y_gives_column_not_found() {
        let df  = iris_df();
        let err = ScatterBuilder::new(&df).x("sepal_length").y("sepal_wdith").build().unwrap_err();
        assert!(matches!(err, CharcoalError::ColumnNotFound { name, .. } if name == "sepal_wdith"));
    }

    #[test]
    fn categorical_x_gives_unsupported_column() {
        let df  = iris_df();
        let err = ScatterBuilder::new(&df).x("species").y("sepal_width").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { col, .. } if col == "species"));
    }

    #[test]
    fn categorical_y_gives_unsupported_column() {
        let df  = iris_df();
        let err = ScatterBuilder::new(&df).x("sepal_length").y("species").build().unwrap_err();
        assert!(matches!(err, CharcoalError::UnsupportedColumn { col, .. } if col == "species"));
    }

    #[test]
    fn exceeding_row_limit_gives_data_too_large() {
        let df  = iris_df();
        let err = ScatterBuilder::new(&df)
            .x("sepal_length").y("sepal_width").row_limit(3).build().unwrap_err();
        match err {
            CharcoalError::DataTooLarge { rows, limit, .. } => {
                assert_eq!(rows, 5); assert_eq!(limit, 3);
            }
            other => panic!("expected DataTooLarge, got {other:?}"),
        }
    }

    // ── Null handling ─────────────────────────────────────────────────────────

    #[test]
    fn nulls_in_x_emit_warning_and_skip_points() {
        let df = DataFrame::new(vec![
            Series::new("x", &[Some(1.0f64), None, Some(3.0), Some(4.0), Some(5.0)]),
            Series::new("y", &[Some(1.0f64), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]),
        ]).unwrap();
        let chart = ScatterBuilder::new(&df).x("x").y("y").build().unwrap();
        assert!(chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col == "x"
        )));
        assert_eq!(chart.svg().matches("<circle").count(), 4);
    }

    #[test]
    fn nulls_in_y_emit_warning_and_skip_points() {
        let df = DataFrame::new(vec![
            Series::new("x", &[Some(1.0f64), Some(2.0), Some(3.0)]),
            Series::new("y", &[Some(1.0f64), None,      Some(3.0)]),
        ]).unwrap();
        let chart = ScatterBuilder::new(&df).x("x").y("y").build().unwrap();
        assert!(chart.warnings().iter().any(|w| matches!(
            w, CharcoalWarning::NullsSkipped { col, .. } if col == "y"
        )));
        assert_eq!(chart.svg().matches("<circle").count(), 2);
    }

    // ── color_by ─────────────────────────────────────────────────────────────

    #[test]
    fn color_by_produces_legend() {
        let df    = iris_df();
        let chart = ScatterBuilder::new(&df)
            .x("sepal_length").y("sepal_width").color_by("species").build().unwrap();
        assert!(chart.svg().contains("setosa"));
        assert!(chart.svg().contains("versicolor"));
        assert!(chart.svg().contains("virginica"));
    }

    #[test]
    fn null_category_appears_in_legend() {
        let df = DataFrame::new(vec![
            Series::new("x",   &[1.0f64, 2.0, 3.0]),
            Series::new("y",   &[1.0f64, 2.0, 3.0]),
            Series::new("cat", &[Some("a"), None, Some("b")]),
        ]).unwrap();
        let chart = ScatterBuilder::new(&df).x("x").y("y").color_by("cat").build().unwrap();
        assert!(chart.svg().contains("null"));
    }

    #[test]
    fn null_category_uses_null_color() {
        let df = DataFrame::new(vec![
            Series::new("x",   &[1.0f64, 2.0]),
            Series::new("y",   &[1.0f64, 2.0]),
            Series::new("cat", &[None::<&str>, Some("a")]),
        ]).unwrap();
        let chart = ScatterBuilder::new(&df).x("x").y("y").color_by("cat").build().unwrap();
        assert!(chart.svg().contains(NULL_COLOR));
    }

    #[test]
    fn palette_cycles_when_categories_exceed_palette_length() {
        let cats: Vec<Option<&str>> = (0..9)
            .map(|i| Some(["a","b","c","d","e","f","g","h","i"][i]))
            .collect();
        let xs: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let df = DataFrame::new(vec![
            Series::new("x",   &xs),
            Series::new("y",   &xs),
            Series::new("cat", &cats),
        ]).unwrap();
        let chart = ScatterBuilder::new(&df).x("x").y("y").color_by("cat").build().unwrap();
        for label in ["a","b","c","d","e","f","g","h","i"] {
            assert!(chart.svg().contains(label), "legend missing '{label}'");
        }
    }

    // ── size_by ───────────────────────────────────────────────────────────────

    #[test]
    fn size_by_produces_varying_radii() {
        let df = DataFrame::new(vec![
            Series::new("x",    &[1.0f64, 2.0, 3.0]),
            Series::new("y",    &[1.0f64, 2.0, 3.0]),
            Series::new("size", &[4.0f64, 8.0, 16.0]),
        ]).unwrap();
        let svg = ScatterBuilder::new(&df).x("x").y("y").size_by("size").build().unwrap().svg().to_string();
        assert!(svg.contains(r#"r="2.00""#), "missing r=2.00");
        assert!(svg.contains(r#"r="4.00""#), "missing r=4.00");
        assert!(svg.contains(r#"r="8.00""#), "missing r=8.00");
    }

    // ── Unit tests for free functions ─────────────────────────────────────────

    #[test]
    fn data_range_empty_returns_fallback() {
        assert_eq!(data_range(std::iter::empty()), (0.0, 1.0));
    }

    #[test]
    fn data_range_single_value() {
        assert_eq!(data_range(std::iter::once(5.0)), (5.0, 5.0));
    }

    #[test]
    fn resolve_colors_assigns_palette_in_insertion_order() {
        let theme  = ThemeConfig::from(&Theme::Default);
        let cv     = Some(vec![Some("a".to_string()), Some("b".to_string()), Some("a".to_string())]);
        let map    = resolve_colors(&cv, &theme);
        assert_eq!(map.len(), 2);
        assert_eq!(map["a"], theme.palette[0]);
        assert_eq!(map["b"], theme.palette[1]);
    }

    #[test]
    fn build_points_counts_nulls_per_axis_independently() {
        let cols = LoadedColumns {
            x_vals:        vec![None, Some(2.0), Some(3.0)],
            y_vals:        vec![Some(1.0), None, Some(3.0)],
            color_vals:    None,
            size_vals:     None,
            x_is_temporal: false,
            warnings:      vec![],
        };
        let (pts, xn, yn) = build_points(&cols, &[0, 1, 2], 6.0);
        assert_eq!(pts.len(), 1);
        assert_eq!(xn, 1);
        assert_eq!(yn, 1);
    }

    #[test]
    fn make_legend_preserves_first_seen_order() {
        let mut map = HashMap::new();
        map.insert("b".to_string(), "#111".to_string());
        map.insert("a".to_string(), "#222".to_string());
        let cv      = Some(vec![Some("b".to_string()), Some("a".to_string()), Some("b".to_string())]);
        let entries = make_legend(&cv, &map).unwrap();
        assert_eq!(entries[0].0, "b");
        assert_eq!(entries[1].0, "a");
    }

    #[test]
    fn subsample_returns_all_indices_below_threshold() {
        let mut w = Vec::new();
        let idx   = subsample(10, &mut w);
        assert_eq!(idx, (0..10).collect::<Vec<_>>());
        assert!(w.is_empty());
    }

    #[test]
    fn subsample_emits_warning_above_threshold() {
        let mut w = Vec::new();
        subsample(SUBSAMPLE_THRESHOLD + 1, &mut w);
        assert!(w.iter().any(|warn| matches!(warn, CharcoalWarning::RowsSubsampled { .. })));
    }
}