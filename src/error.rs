//! Error and warning types for charcoal.
//!
//! [`CharcoalError`] is returned from all fallible public methods (`.build()`,
//! `.save_svg()`, etc.). [`CharcoalWarning`] is a non-fatal diagnostic attached
//! to a successfully-built [`crate::Chart`]; retrieve warnings with
//! [`crate::Chart::warnings`].

use polars::datatypes::DataType;
use thiserror::Error;

/// Errors that can be returned by charcoal's public API.
///
/// Every variant answers three questions: what went wrong, where in the user's
/// code, and what to do next. The `Display` output is intended to be shown
/// directly to end users without further wrapping.
///
/// # Examples
///
/// ```rust,no_run
/// use charcoal::{Chart, CharcoalError};
/// # let df = polars::frame::DataFrame::empty();
///
/// match Chart::scatter(&df).x("x").y("typo").build() {
///     Ok(chart) => chart.save_html("out.html").unwrap(),
///     Err(CharcoalError::ColumnNotFound { name, .. }) => {
///         eprintln!("column not found: {name}");
///     }
///     Err(e) => eprintln!("error: {e}"),
/// }
/// ```
#[derive(Debug, Error)]
pub enum CharcoalError {
    /// A column name passed to a builder method does not exist in the DataFrame.
    ///
    /// The error message includes a spelling suggestion (if one is found within
    /// edit distance 2) and lists all available column names.
    ///
    /// **Fix:** check the column name against `df.get_column_names()` or correct
    /// the typo shown in the suggestion.
    ///
    /// **Display example:**
    /// ```text
    /// column "sepal_lenght" not found
    ///   Did you mean: "sepal_length"
    ///   Available: sepal_length, sepal_width, petal_length, species
    /// ```
    #[error(
        "column \"{name}\" not found\n  Did you mean: \"{suggestion}\"\n  Available: {available}"
    )]
    ColumnNotFound {
        name: String,
        suggestion: String,
        available: String,
    },

    /// A column was found but its Polars dtype cannot be used in that chart role.
    ///
    /// For example, passing a `List` column as the x-axis of a scatter chart.
    /// The error message names the column, its dtype, and explains the constraint.
    ///
    /// **Fix:** cast or preprocess the column before charting.
    ///
    /// **Display example:**
    /// ```text
    /// column "tags" has unsupported type List(String)
    ///   List columns cannot be used as chart axes.
    /// ```
    #[error("column \"{col}\" has unsupported type {dtype:?}\n  {message}")]
    UnsupportedColumn {
        col: String,
        dtype: DataType,
        message: String,
    },

    /// A column has fewer non-null rows than the chart type requires to render.
    ///
    /// For example, a box plot needs at least 1 non-null value per group.
    ///
    /// **Fix:** filter the DataFrame or fill nulls before calling `.build()`.
    ///
    /// **Display example:**
    /// ```text
    /// column "value" has insufficient data: need at least 1 non-null rows, got 0
    ///   Consider dropping nulls or filtering before charting.
    /// ```
    #[error(
        "column \"{col}\" has insufficient data: need at least {required} non-null rows, got {got}\n  Consider dropping nulls or filtering before charting."
    )]
    InsufficientData {
        col: String,
        required: usize,
        got: usize,
    },

    /// The DataFrame exceeds the configured row render limit.
    ///
    /// The default limit is 1 000 000 rows. Above 500 000 rows a
    /// [`CharcoalWarning::RowsSubsampled`] warning is emitted instead.
    ///
    /// **Fix:** subsample before charting (`df.sample_n(500_000, ...)`) or raise
    /// the limit with `.row_limit(n)` on the builder.
    ///
    /// **Display example:**
    /// ```text
    /// DataFrame has 1200000 rows which exceeds the 1000000 row render limit
    ///   Consider df.sample(500_000) before charting.
    /// ```
    #[error("DataFrame has {rows} rows which exceeds the {limit} row render limit\n  {message}")]
    DataTooLarge {
        rows: usize,
        limit: usize,
        message: String,
    },

    /// An internal SVG rendering step failed.
    ///
    /// This is not expected during normal use. If you encounter it, please open
    /// an issue at <https://github.com/your-handle/charcoal> with the full error
    /// message and the chart type that triggered it.
    ///
    /// **Display example:**
    /// ```text
    /// SVG render error: zero-length axis range
    /// ```
    #[error("SVG render error: {0}")]
    RenderError(String),

    /// A file could not be read or written.
    ///
    /// Wraps [`std::io::Error`]. Triggered by [`crate::Chart::save_svg`],
    /// [`crate::Chart::save_html`], and [`crate::Chart::save_png`].
    ///
    /// **Fix:** check that the target directory exists and is writable.
    ///
    /// **Display example:**
    /// ```text
    /// I/O error: No such file or directory (os error 2)
    /// ```
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// --- CharcoalWarning ---

/// Non-fatal diagnostics attached to a successfully-built [`crate::Chart`].
///
/// Warnings indicate that charcoal made an automatic adjustment during rendering
/// (e.g. skipping null values, subsampling rows). They never prevent a chart from
/// being produced, but should be surfaced to users so they know what happened.
///
/// Retrieve warnings after `.build()` with [`crate::Chart::warnings`]:
///
/// ```rust,no_run
/// # use charcoal::Chart;
/// # let df = polars::frame::DataFrame::empty();
/// let chart = Chart::scatter(&df).x("x").y("y").build()?;
/// for w in chart.warnings() {
///     eprintln!("warning: {w}");
/// }
/// # Ok::<(), charcoal::CharcoalError>(())
/// ```
#[derive(Debug, Clone)]
pub enum CharcoalWarning {
    /// One or more null values in a column were excluded from rendering.
    ///
    /// Emitted whenever a null is encountered in any column role that does not
    /// support nulls natively (e.g. x-axis values, group keys). The `count`
    /// field tells how many rows were dropped.
    ///
    /// **Display example:**
    /// ```text
    /// 3 null value(s) in column "y" were skipped during rendering
    /// ```
    NullsSkipped { col: String, count: usize },

    /// The DataFrame had more than 500 000 rows and was subsampled before rendering.
    ///
    /// charcoal selects a deterministic subset of `rendered` rows to keep
    /// rendering fast. Use `.row_limit(n)` on the builder to change the threshold,
    /// or subsample the DataFrame yourself before calling `.build()` if you need
    /// control over which rows are kept.
    ///
    /// **Display example:**
    /// ```text
    /// DataFrame had 800000 rows; rendered a subsampled 500000 rows for performance
    /// ```
    RowsSubsampled { original: usize, rendered: usize },

    /// A box-plot notch was clamped because the IQR was too small to fit it.
    ///
    /// Notches extend ±1.57 × IQR / √n beyond the median. When this interval
    /// exceeds Q1 or Q3, it is clamped to the box edge and this warning is
    /// emitted. The chart still renders correctly — only the notch shape is
    /// affected.
    ///
    /// **Display example:**
    /// ```text
    /// notch for group "A" was clamped to the box bounds (IQR too small)
    /// ```
    NotchClamped { group: String },
}

impl std::fmt::Display for CharcoalWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullsSkipped { col, count } => write!(
                f,
                "{count} null value(s) in column \"{col}\" were skipped during rendering"
            ),
            Self::RowsSubsampled { original, rendered } => write!(
                f,
                "DataFrame had {original} rows; rendered a subsampled {rendered} rows for performance"
            ),
            Self::NotchClamped { group } => write!(
                f,
                "notch for group \"{group}\" was clamped to the box bounds (IQR too small)"
            ),
        }
    }
}

#[allow(dead_code)]
pub(crate) fn suggest_column<'a>(name: &str, available: &[&'a str]) -> &'a str {
    available
        .iter()
        .filter_map(|candidate| {
            let dist = levenshtein(name, candidate);
            if dist <= 2 {
                Some((dist, *candidate))
            } else {
                None
            }
        })
        .min_by_key(|(dist, _)| *dist)
        .map(|(_, candidate)| candidate)
        .unwrap_or("")
}

#[allow(dead_code)]
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1]
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::datatypes::DataType;

    // --- suggest_column tests ---

    #[test]
    fn test_suggest_one_char_typo() {
        let available = &["sepal_length", "sepal_width", "species"];
        assert_eq!(suggest_column("sepal_lenght", available), "sepal_length");
    }

    #[test]
    fn test_suggest_transposition() {
        let available = &["sepal_length", "sepal_width", "species"];
        assert_eq!(suggest_column("sepal_wdith", available), "sepal_width");
    }

    #[test]
    fn test_suggest_too_far_returns_empty() {
        let available = &["sepal_length", "sepal_width", "species"];
        assert_eq!(suggest_column("completely_wrong_name", available), "");
    }

    #[test]
    fn test_suggest_empty_available_returns_empty() {
        assert_eq!(suggest_column("sepal_length", &[]), "");
    }

    // --- Display output tests ---

    #[test]
    fn test_column_not_found_display() {
        let err = CharcoalError::ColumnNotFound {
            name: "sepal_lenght".to_string(),
            suggestion: "sepal_length".to_string(),
            available: "sepal_length, sepal_width, species".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("sepal_lenght"),
            "must contain the mistyped name"
        );
        assert!(msg.contains("sepal_length"), "must contain the suggestion");
        assert!(
            msg.contains("sepal_width"),
            "must contain available columns"
        );
    }

    #[test]
    fn test_unsupported_column_display() {
        let err = CharcoalError::UnsupportedColumn {
            col: "tags".to_string(),
            dtype: DataType::List(Box::new(DataType::String)),
            message: "List columns cannot be used as chart axes.".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("tags"), "must contain the column name");
        assert!(msg.contains("List"), "must contain the dtype");
        assert!(msg.contains("List columns"), "must contain the message");
    }

    #[test]
    fn test_data_too_large_display() {
        let err = CharcoalError::DataTooLarge {
            rows: 1_200_000,
            limit: 1_000_000,
            message: "Consider df.sample(500_000) before charting.".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("1200000"), "must contain the row count");
        assert!(msg.contains("1000000"), "must contain the limit");
        assert!(msg.contains("sample"), "must contain the recommendation");
    }
}