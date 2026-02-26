#![allow(dead_code)]

pub(crate) fn to_inline_html(svg: &str, title: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: sans-serif;
    background: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
    min-height: 100vh;
  }}
  svg {{
    max-width: 100%;
    height: auto;
  }}
</style>
</head>
<body>
{svg}
</body>
</html>"#,
        title = html_escape(title),
        svg = svg,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><rect width="800" height="600" fill="#fff"/></svg>"##;

    #[test]
    fn output_starts_with_doctype() {
        let html = to_inline_html(SAMPLE_SVG, "Test Chart");
        assert!(html.starts_with("<!DOCTYPE html>"), "should start with DOCTYPE: {}", &html[..40]);
    }

    #[test]
    fn output_contains_svg_verbatim() {
        let html = to_inline_html(SAMPLE_SVG, "Test");
        assert!(html.contains(SAMPLE_SVG), "SVG should appear verbatim (not escaped) in output");
    }

    #[test]
    fn output_contains_title_in_title_tag() {
        let html = to_inline_html(SAMPLE_SVG, "My Scatter Plot");
        assert!(html.contains("<title>My Scatter Plot</title>"), "title tag not found: {html}");
    }

    #[test]
    fn body_closes_before_html() {
        let html = to_inline_html(SAMPLE_SVG, "Test");
        let body_pos = html.rfind("</body>").expect("</body> not found");
        let html_pos = html.rfind("</html>").expect("</html> not found");
        assert!(body_pos < html_pos, "</body> must appear before </html>");
    }

    #[test]
    fn html_is_well_structured() {
        let html = to_inline_html(SAMPLE_SVG, "Test");
        // Basic structure checks
        assert!(html.contains("<html"), "missing <html");
        assert!(html.contains("<head>"), "missing <head>");
        assert!(html.contains("</head>"), "missing </head>");
        assert!(html.contains("<body>"), "missing <body>");
        assert!(html.contains("</body>"), "missing </body>");
        assert!(html.contains("</html>"), "missing </html>");
    }

    #[test]
    fn title_special_chars_are_escaped() {
        let html = to_inline_html(SAMPLE_SVG, "A & B <Chart>");
        // The title in <title> tag should be escaped
        assert!(html.contains("A &amp; B &lt;Chart&gt;"), "title should be HTML-escaped: {html}");
    }

    #[test]
    fn svg_content_is_not_html_escaped() {
        // SVG contains < > characters — they must appear literally, not as &lt; &gt;
        let html = to_inline_html(SAMPLE_SVG, "Test");
        assert!(
            html.contains(r#"<svg xmlns="http://www.w3.org/2000/svg""#),
            "SVG opening tag should not be escaped"
        );
    }

    #[test]
    fn empty_title_produces_empty_title_tag() {
        let html = to_inline_html(SAMPLE_SVG, "");
        assert!(html.contains("<title></title>"), "empty title should produce empty title tag");
    }
}