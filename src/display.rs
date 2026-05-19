#![cfg(feature = "notebook")]

use crate::charts::Chart;

/// Renders [`Chart`] inline in an evcxr notebook cell as `image/svg+xml`.
///
/// The SVG string already lives on `self`; `mime_type(..).text(..)` borrows it
/// via `AsRef<str>` and emits the evcxr protocol markers without copying.
impl evcxr_runtime::Display for Chart {
    fn evcxr_display(&self) {
        evcxr_runtime::mime_type("image/svg+xml").text(&self.svg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_chart() -> Chart {
        Chart {
            svg: r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>"#
                .to_string(),
            warnings: vec![],
            title: "Test".to_string(),
            width: 100,
            height: 100,
        }
    }

    #[test]
    fn display_does_not_panic() {
        minimal_chart().display();
    }

    #[test]
    fn svg_content_is_valid_utf8() {
        let chart = minimal_chart();
        assert!(std::str::from_utf8(chart.svg().as_bytes()).is_ok());
    }

    #[test]
    fn svg_content_starts_with_svg_tag() {
        let chart = minimal_chart();
        assert!(chart.svg().starts_with("<svg"));
    }
}
