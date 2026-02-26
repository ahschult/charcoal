use polars::datatypes::DataType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CharcoalError {
    #[error(
        "column \"{name}\" not found\n  Did you mean: \"{suggestion}\"\n  Available: {available}"
    )]
    ColumnNotFound {
        name: String,
        suggestion: String,
        available: String,
    },

    #[error("column \"{col}\" has unsupported type {dtype:?}\n  {message}")]
    UnsupportedColumn {
        col: String,
        dtype: DataType,
        message: String,
    },

    #[error(
        "column \"{col}\" has insufficient data: need at least {required} non-null rows, got {got}\n  Consider dropping nulls or filtering before charting."
    )]
    InsufficientData {
        col: String,
        required: usize,
        got: usize,
    },

    #[error("DataFrame has {rows} rows which exceeds the {limit} row render limit\n  {message}")]
    DataTooLarge {
        rows: usize,
        limit: usize,
        message: String,
    },

    #[error("SVG render error: {0}")]
    RenderError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// --- CharcoalWarning ---

#[derive(Debug, Clone)]
pub enum CharcoalWarning {
    NullsSkipped { col: String, count: usize },
    RowsSubsampled { original: usize, rendered: usize },
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
