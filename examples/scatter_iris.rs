use charcoal::{Chart, Theme};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::new(vec![
        Series::new("sepal_length", &[5.1f64, 4.9, 4.7, 4.6, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 6.3, 5.8, 7.1, 6.3, 6.5]),
        Series::new("sepal_width",  &[3.5f64, 3.0, 3.2, 3.1, 3.6, 3.2, 3.2, 3.1, 2.3, 2.8, 3.3, 2.7, 3.0, 2.9, 3.0]),
        Series::new("species", &[
            "setosa", "setosa", "setosa", "setosa", "setosa",
            "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
            "virginica", "virginica", "virginica", "virginica", "virginica",
        ]),
    ])?;

    let chart = Chart::scatter(&df)
        .x("sepal_length")
        .y("sepal_width")
        .color_by("species")
        .title("Iris: Sepal Length vs Width")
        .opacity(0.8)
        .theme(Theme::Default)
        .build()?;

    chart.save_html("output.html")?;
    println!("scatter_iris: saved output.html");
    Ok(())
}
