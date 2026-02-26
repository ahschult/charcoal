use polars::datatypes::DataType;
use polars::frame::DataFrame;
use crate::error::{suggest_column, CharcoalError};

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum VizDtype {
    Numeric,
    Temporal,
    Categorical,
    Unsupported,
}

#[allow(dead_code)]
pub(crate) fn classify(dtype: &DataType) -> VizDtype {
    match dtype {
        // Numeric
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => VizDtype::Numeric,

        // Temporal
        DataType::Date
        | DataType::Datetime(_, _)
        | DataType::Duration(_)
        | DataType::Time => VizDtype::Temporal,

        // Categorical
        DataType::String
        | DataType::Boolean => VizDtype::Categorical,

        // Unsupported
        DataType::List(_)
        | DataType::Binary
        | DataType::BinaryOffset
        | DataType::Null
        | DataType::Unknown => VizDtype::Unsupported,
    }
}

#[allow(dead_code)]
pub(crate) fn classify_column(
    df: &DataFrame,
    col: &str,
    expected: Option<VizDtype>,
) -> Result<VizDtype, CharcoalError> {
    let schema = df.schema();
    let available: Vec<&str> = schema.iter_names().map(|s| s.as_str()).collect();

    let dtype = schema.get(col).ok_or_else(|| CharcoalError::ColumnNotFound {
        name: col.to_string(),
        suggestion: suggest_column(col, &available).to_string(),
        available: available.join(", "),
    })?;

    let viz_dtype = classify(dtype);

    if viz_dtype == VizDtype::Unsupported {
        return Err(CharcoalError::UnsupportedColumn {
            col: col.to_string(),
            dtype: dtype.clone(),
            message: unsupported_message(dtype),
        });
    }

    if let Some(expected_dtype) = expected {
        if viz_dtype != expected_dtype {
            return Err(CharcoalError::UnsupportedColumn {
                col: col.to_string(),
                dtype: dtype.clone(),
                message: format!(
                    "Expected a {} column but got {:?}. \
                     Cast the column to the correct type before charting.",
                    viz_dtype_name(&expected_dtype),
                    dtype
                ),
            });
        }
    }

    Ok(viz_dtype)
}

fn unsupported_message(dtype: &DataType) -> String {
    match dtype {
        DataType::List(_) => {
            "List columns cannot be used as chart axes. \
             Cast to a scalar type first, e.g. series.explode()"
                .to_string()
        }
        DataType::Binary | DataType::BinaryOffset => {
            "Binary columns cannot be used as chart axes. \
             Decode to a string or numeric type first."
                .to_string()
        }
        DataType::Null => {
            "Column contains only nulls and cannot be used as a chart axis.".to_string()
        }
        _ => format!("Column type {dtype:?} is not supported as a chart axis."),
    }
}

fn viz_dtype_name(viz_dtype: &VizDtype) -> &'static str {
    match viz_dtype {
        VizDtype::Numeric => "Numeric",
        VizDtype::Temporal => "Temporal",
        VizDtype::Categorical => "Categorical",
        VizDtype::Unsupported => "Unsupported",
    }
}

