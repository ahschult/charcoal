#![cfg(feature = "static")]

use crate::error::CharcoalError;

/// Render an SVG string to PNG bytes at the given pixel dimensions.
///
/// # Errors
/// Returns [`CharcoalError::RenderError`] if the SVG is malformed, the pixmap
/// cannot be allocated (zero-dimension input), or PNG encoding fails.
pub(crate) fn render_png(svg: &str, width: u32, height: u32) -> Result<Vec<u8>, CharcoalError> {
    let opts = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_str(svg, &opts)
        .map_err(|e| CharcoalError::RenderError(format!("SVG parse error: {e}")))?;

    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
        CharcoalError::RenderError(format!(
            "cannot allocate pixmap for {width}x{height} — dimensions must be greater than zero"
        ))
    })?;

    resvg::render(&tree, resvg::tiny_skia::Transform::default(), &mut pixmap.as_mut());

    pixmap
        .encode_png()
        .map_err(|e| CharcoalError::RenderError(format!("PNG encode error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_SVG: &str =
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="blue"/></svg>"#;

    const PNG_MAGIC: [u8; 4] = [0x89, 0x50, 0x4E, 0x47];

    #[test]
    fn render_png_returns_ok_bytes() {
        let bytes = render_png(MINIMAL_SVG, 100, 100).expect("render_png failed");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn render_png_starts_with_png_magic() {
        let bytes = render_png(MINIMAL_SVG, 100, 100).unwrap();
        assert_eq!(&bytes[..4], PNG_MAGIC);
    }

    #[test]
    fn render_png_dimensions_match_requested() {
        let bytes = render_png(MINIMAL_SVG, 800, 600).unwrap();
        // PNG IHDR: bytes 16-20 = width (big-endian u32), bytes 20-24 = height
        let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(w, 800);
        assert_eq!(h, 600);
    }

    #[test]
    fn render_png_malformed_svg_returns_error() {
        let result = render_png("this is not svg at all <<<", 100, 100);
        assert!(matches!(result, Err(CharcoalError::RenderError(_))));
    }

    #[test]
    fn render_png_zero_width_returns_error() {
        let result = render_png(MINIMAL_SVG, 0, 100);
        assert!(matches!(result, Err(CharcoalError::RenderError(_))));
    }

    #[test]
    fn render_png_zero_height_returns_error() {
        let result = render_png(MINIMAL_SVG, 100, 0);
        assert!(matches!(result, Err(CharcoalError::RenderError(_))));
    }
}
