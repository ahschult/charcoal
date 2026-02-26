use polars::datatypes::{DataType, TimeUnit};
use polars::frame::DataFrame;

use crate::dtype::{classify_column, VizDtype};
use crate::error::{CharcoalError, CharcoalWarning};

#[allow(dead_code)]
pub(crate) fn to_f64(
    df: &DataFrame,
    col: &str,
) -> Result<(Vec<Option<f64>>, Vec<CharcoalWarning>), CharcoalError> {
    classify_column(df, col, Some(VizDtype::Numeric))?;

    let series = df
        .column(col)
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;
    let casted = series
        .cast(&DataType::Float64)
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let chunked = casted
        .f64()
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let mut null_count = 0usize;
    let values: Vec<Option<f64>> = chunked
        .into_iter()
        .inspect(|v| {
            if v.is_none() {
                null_count += 1;
            }
        })
        .collect();

    let mut warnings = Vec::new();
    if null_count > 0 {
        warnings.push(CharcoalWarning::NullsSkipped {
            col: col.to_string(),
            count: null_count,
        });
    }

    Ok((values, warnings))
}

#[allow(dead_code)]
pub(crate) fn to_epoch_ms(
    df: &DataFrame,
    col: &str,
) -> Result<(Vec<Option<i64>>, Vec<CharcoalWarning>), CharcoalError> {
    classify_column(df, col, Some(VizDtype::Temporal))?;

    let series = df
        .column(col)
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let as_datetime = series
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let as_i64 = as_datetime
        .cast(&DataType::Int64)
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let chunked = as_i64
        .i64()
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let mut null_count = 0usize;
    let values: Vec<Option<i64>> = chunked
        .into_iter()
        .inspect(|v| {
            if v.is_none() {
                null_count += 1;
            }
        })
        .collect();

    let mut warnings = Vec::new();
    if null_count > 0 {
        warnings.push(CharcoalWarning::NullsSkipped {
            col: col.to_string(),
            count: null_count,
        });
    }

    Ok((values, warnings))
}

#[allow(dead_code)]
pub(crate) fn to_string(
    df: &DataFrame,
    col: &str,
) -> Result<(Vec<Option<String>>, Vec<CharcoalWarning>), CharcoalError> {
    classify_column(df, col, Some(VizDtype::Categorical))?;

    let series = df
        .column(col)
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let casted = series
        .cast(&DataType::String)
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let chunked = casted
        .str()
        .map_err(|e| CharcoalError::RenderError(e.to_string()))?;

    let mut null_count = 0usize;
    let values: Vec<Option<String>> = chunked
        .into_iter()
        .map(|v| {
            if v.is_none() {
                null_count += 1;
            }
            v.map(|s| s.to_string())
        })
        .collect();

    let mut warnings = Vec::new();
    if null_count > 0 {
        warnings.push(CharcoalWarning::NullsSkipped {
            col: col.to_string(),
            count: null_count,
        });
    }

    Ok((values, warnings))
}

#[cfg(test)]
mod tests {
    use polars::datatypes::TimeUnit;
    use polars::prelude::*;

    use super::*;

    // --- to_f64 tests ---

    #[test]
    fn test_to_f64_int32() {
        let df = DataFrame::new(vec![Series::new("x", &[1i32, 2, 3])]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_f64_int64() {
        let df = DataFrame::new(vec![Series::new("x", &[1i64, 2, 3])]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_f64_uint32() {
        let df = DataFrame::new(vec![Series::new("x", &[1u32, 2, 3])]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_f64_uint64() {
        let df = DataFrame::new(vec![Series::new("x", &[1u64, 2, 3])]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_f64_float32() {
        let df = DataFrame::new(vec![Series::new("x", &[1.0f32, 2.0, 3.0])]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert!(values[0].unwrap() - 1.0 < 1e-6);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_f64_float64() {
        let df = DataFrame::new(vec![Series::new("x", &[1.0f64, 2.0, 3.0])]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_f64_nulls_produce_none_and_warning() {
        let s = Series::new("x", &[Some(1.0f64), None, Some(3.0)]);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            CharcoalWarning::NullsSkipped { col, count } => {
                assert_eq!(col, "x");
                assert_eq!(*count, 1);
            }
            _ => panic!("expected NullsSkipped warning"),
        }
    }

    #[test]
    fn test_to_f64_all_nulls_one_warning() {
        let s = Series::new("x", &[None::<f64>, None, None]);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert_eq!(values, vec![None, None, None]);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_to_f64_zero_rows() {
        let s = Series::new_empty("x", &DataType::Float64);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_f64(&df, "x").unwrap();
        assert!(values.is_empty());
        assert!(warnings.is_empty());
    }

    // --- to_epoch_ms tests ---

    #[test]
    fn test_to_epoch_ms_date_known_value() {
        // 2024-01-01 is 19723 days since Unix epoch
        // 19723 * 86400 * 1000 = 1_704_067_200_000 ms
        let s = Series::new("d", &[19723i32]).cast(&DataType::Date).unwrap();
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_epoch_ms(&df, "d").unwrap();
        assert_eq!(values[0], Some(1_704_067_200_000i64));
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_epoch_ms_datetime() {
        let s = Series::new("dt", &[1_704_067_200_000i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_epoch_ms(&df, "dt").unwrap();
        assert_eq!(values[0], Some(1_704_067_200_000i64));
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_epoch_ms_duration() {
        let s = Series::new("dur", &[5_000i64])
            .cast(&DataType::Duration(TimeUnit::Milliseconds))
            .unwrap();
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_epoch_ms(&df, "dur").unwrap();
        assert_eq!(values[0], Some(5_000i64));
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_epoch_ms_rejects_numeric() {
        let df = DataFrame::new(vec![Series::new("x", &[1.0f64, 2.0])]).unwrap();
        let err = to_epoch_ms(&df, "x").unwrap_err();
        match err {
            CharcoalError::UnsupportedColumn { col, .. } => assert_eq!(col, "x"),
            _ => panic!("expected UnsupportedColumn"),
        }
    }

    #[test]
    fn test_to_epoch_ms_zero_rows() {
        let s = Series::new_empty("d", &DataType::Date);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_epoch_ms(&df, "d").unwrap();
        assert!(values.is_empty());
        assert!(warnings.is_empty());
    }

    // --- to_string tests ---

    #[test]
    fn test_to_string_boolean() {
        let df = DataFrame::new(vec![Series::new("b", &[true, false, true])]).unwrap();
        let (values, warnings) = to_string(&df, "b").unwrap();
        assert_eq!(
            values,
            vec![
                Some("true".to_string()),
                Some("false".to_string()),
                Some("true".to_string()),
            ]
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_string_string_column() {
        let df = DataFrame::new(vec![Series::new(
            "s",
            &["setosa", "versicolor", "virginica"],
        )])
        .unwrap();
        let (values, warnings) = to_string(&df, "s").unwrap();
        assert_eq!(
            values,
            vec![
                Some("setosa".to_string()),
                Some("versicolor".to_string()),
                Some("virginica".to_string()),
            ]
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_to_string_nulls_produce_none_and_warning() {
        let s = Series::new("s", &[Some("a"), None, Some("b")]);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_string(&df, "s").unwrap();
        assert_eq!(
            values,
            vec![Some("a".to_string()), None, Some("b".to_string())]
        );
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            CharcoalWarning::NullsSkipped { col, count } => {
                assert_eq!(col, "s");
                assert_eq!(*count, 1);
            }
            _ => panic!("expected NullsSkipped warning"),
        }
    }

    #[test]
    fn test_to_string_all_nulls_one_warning() {
        let s = Series::new("s", &[None::<&str>, None, None]);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_string(&df, "s").unwrap();
        assert_eq!(values, vec![None, None, None]);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_to_string_zero_rows() {
        let s = Series::new_empty("s", &DataType::String);
        let df = DataFrame::new(vec![s]).unwrap();
        let (values, warnings) = to_string(&df, "s").unwrap();
        assert!(values.is_empty());
        assert!(warnings.is_empty());
    }
}
