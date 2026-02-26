# Contributing to charcoal

## Module Structure

Each source file has a single, clear responsibility:

| File | Responsibility |
|------|---------------|
| `src/error.rs` | `CharcoalError`, `CharcoalWarning`, and the column suggestion helper |
| `src/dtype.rs` | `VizDtype` enum, `classify()`, and `classify_column()` |
| `src/normalize.rs` | `to_f64()`, `to_epoch_ms()`, `to_string()` normalization functions |
| `src/theme.rs` | `Theme`, `ThemeConfig`, `ColorScale` and color stop tables |
| `src/render/geometry.rs` | Atomic SVG element functions — pure data-in, string-out |
| `src/render/axes.rs` | Scale computation, tick generation, axis SVG output |
| `src/render/mod.rs` | `SvgCanvas` — assembles the complete SVG document |
| `src/render/html.rs` | Standalone HTML wrapper (inline SVG, no JS) |
| `src/chart/<name>.rs` | One file per chart type — typestate builder and `.build()` |
| `src/lib.rs` | Public re-exports only — no logic lives here |

### What `pub(crate)` means in practice

`pub(crate)` means the item is accessible anywhere inside the charcoal crate
but is invisible to users. This is the default for everything that is not
explicitly part of the public API documented in `src/lib.rs`.

When in doubt, use `pub(crate)`. Loosening visibility later is not a breaking
change. Tightening it is.

### How to add a new chart type

1. Create `src/chart/<name>.rs`
2. Define the typestate builder chain — use `src/chart/scatter.rs` as the
   reference implementation
3. Define a `<Name>Config` struct to accumulate optional builder fields
4. Implement `.build()` on the final typestate struct, returning
   `Result<Chart, CharcoalError>`
5. Call `classify_column()` for every column reference in `.build()`
6. Call the appropriate `normalize::to_*` function for each column
7. Call into `geometry.rs` for SVG elements and `SvgCanvas::render()` for
   final assembly
8. Add the entry point `Chart::<name>(df: &DataFrame)` to `src/chart/mod.rs`
9. Re-export the builder entry point from `src/lib.rs` if needed
10. Add an example in `examples/<name>.rs`
11. Run `cargo test --all-features` and `cargo clippy --all-features -- -D warnings`

---

## Style Guide

These are concrete, checkable rules — not principles. Apply them mechanically.

### Argument naming

- Always use `col: &str` for column references — never `column`, `name`, or `field`
- Always use `path: &str` for file path arguments — never `filename` or `filepath`

### Method naming

- Verb-first for actions: `save_svg`, `save_png`, `save_html`, `build`
- Noun-first for accessors: `svg()`, `warnings()`, `theme()`
- Builder methods use the field name directly: `.x()`, `.y()`, `.title()`,
  `.color_by()`, `.theme()`

### Return types

- All fallible public methods return `Result<T, CharcoalError>`
- Never return `Option` from a public method — convert to `Result` with an
  appropriate error variant
- Builder methods that advance typestate return the next typestate type, not `Self`
- Builder methods that set optional fields return `Self`

### Visibility

- Everything is `pub(crate)` by default
- Only items explicitly listed in `src/lib.rs` re-exports are `pub`
- No `pub(super)` — use `pub(crate)` instead for consistency

### No unwrap rule

- No `unwrap()` or `expect()` anywhere in `src/` outside of `#[cfg(test)]` blocks
- Use `?` for error propagation
- Use `.map_err(|e| CharcoalError::RenderError(e.to_string()))` when converting
  foreign errors that don't implement `From<_>` for `CharcoalError`
- A panic in library code is a bug, not error handling

### Doc comments

- Every `pub` item requires a doc comment before the PR can merge
- Every public function doc comment requires a `# Errors` section if it returns
  `Result`
- Every public function doc comment requires a `# Examples` block
- Run `cargo doc --no-deps --all-features` locally and confirm zero warnings

### Error messages

Every error message must answer three questions:
1. What went wrong?
2. Where in the user's code?
3. What should the user do next?

Example of a good error message:
```
column "sepal_lenght" not found
  Did you mean: "sepal_length"
  Available: sepal_length, sepal_width, petal_length, petal_width, species
```

### Null handling

Every chart type must document and test its null policy for each column role.
Nulls are never silently dropped — they must either produce a `CharcoalWarning`
or be documented as intentionally excluded (e.g. histogram bin computation).

---

## PR Checklist

Use this checklist on every PR without exception. Every box must be checked
before requesting review.

### Code
- [ ] No `unwrap()` or `expect()` in `src/` outside of `#[cfg(test)]` blocks
- [ ] No panics reachable from the public API
- [ ] No new `pub` items without doc comments and `# Examples` blocks
- [ ] New dependencies justified in PR description (none added without discussion)
- [ ] No `unsafe` blocks anywhere in `src/`

### API changes
- [ ] Any new `pub` item follows the naming conventions in this document
- [ ] Breaking changes noted explicitly in the PR description
- [ ] New public error variants have `Display` output tests

### Tests
- [ ] Happy path tested for each new public function
- [ ] Error cases tested for each new public function
- [ ] Null handling tested explicitly for each new column role
- [ ] `cargo test --all-features` passes locally

### Docs
- [ ] `cargo doc --no-deps --all-features` generates without warnings locally
- [ ] New chart types have an example file in `examples/`
- [ ] New chart type examples run without errors (`cargo run --example <name>`)