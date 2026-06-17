//! Host the packed model image from the `model` flash partition.
//!
//! The partition holds `TOC(16) + tokenizer.bin + model.bin` (see the
//! `tinyllm` `gen_toy_model` example): magic `b"PSM1"`, u32 version, u32 tok_len,
//! u32 model_len, then the two blobs. We never copy the blobs into RAM — each route
//! returns a [`BodySource::Flash`] backed by [`esp_partition_read`], so the body
//! streams off flash in ≤MSS / ≤chunk reads with O(1) RAM, and HTTP Range works (the
//! browser range-requests the multi-MB model). Looked up by label, so the partition
//! subtype is irrelevant.

use std::sync::OnceLock;

use esp_idf_svc::sys;
use tailscale_core::tcp::{BodySource, FlashReader};

/// Resolved layout of the packed image, computed once from the TOC. v2 adds the
/// optional browser blobs (POS/word metadata json + the int8 embedding matrix);
/// they are 0-length for a v1 image.
struct Toc {
    part: usize, // *const esp_partition_t as usize (stable for program lifetime; Send-safe)
    tok_off: usize,
    tok_len: usize,
    model_off: usize,
    model_len: usize,
    wordmeta_off: usize,
    wordmeta_len: usize,
    embed_off: usize,
    embed_len: usize,
}

static TOC: OnceLock<Option<Toc>> = OnceLock::new();

/// Read `buf.len()` bytes at flash `off` within the partition. `esp_partition_read`
/// is internally locked, so it is safe to call from the serving worker threads.
fn part_read(part: usize, off: usize, buf: &mut [u8]) -> bool {
    let p = part as *const sys::esp_partition_t;
    let r = unsafe {
        sys::esp_partition_read(p, off, buf.as_mut_ptr() as *mut core::ffi::c_void, buf.len())
    };
    r == sys::ESP_OK
}

fn load() -> Option<Toc> {
    let label = std::ffi::CString::new("model").unwrap();
    let p = unsafe {
        sys::esp_partition_find_first(
            sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
            sys::esp_partition_subtype_t_ESP_PARTITION_SUBTYPE_ANY,
            label.as_ptr(),
        )
    };
    if p.is_null() {
        println!("model_host: 'model' partition not found (flash the image to 0x810000)");
        return None;
    }
    let part = p as usize;
    let size = unsafe { (*p).size } as usize;

    let mut hdr = [0u8; 24];
    if !part_read(part, 0, &mut hdr) {
        println!("model_host: TOC read failed");
        return None;
    }
    if &hdr[0..4] != b"PSM1" {
        println!("model_host: no model image (bad TOC magic {:02x?})", &hdr[0..4]);
        return None;
    }
    let u = |i: usize| u32::from_le_bytes([hdr[i], hdr[i + 1], hdr[i + 2], hdr[i + 3]]) as usize;
    let version = u(4);
    let tok_len = u(8);
    let model_len = u(12);
    // v2 adds wordmeta + embed; v1 has only tokenizer + model (24- vs 16-byte TOC).
    let (base, wordmeta_len, embed_len) = if version >= 2 { (24, u(16), u(20)) } else { (16, 0, 0) };
    let tok_off = base;
    let model_off = tok_off + tok_len;
    let wordmeta_off = model_off + model_len;
    let embed_off = wordmeta_off + wordmeta_len;
    if embed_off + embed_len > size {
        println!("model_host: image exceeds partition ({size} B)");
        return None;
    }
    println!(
        "model_host: ready (v{version}) — tokenizer {tok_len} B, model {model_len} B, \
         wordmeta {wordmeta_len} B, embed {embed_len} B in {size} B partition"
    );
    Some(Toc {
        part, tok_off, tok_len, model_off, model_len,
        wordmeta_off, wordmeta_len, embed_off, embed_len,
    })
}

fn toc() -> Option<&'static Toc> {
    TOC.get_or_init(load).as_ref()
}

/// A [`FlashReader`] over a `[base, base+len)` window of the partition.
struct PartReader {
    part: usize,
    base: usize,
    len: usize,
}

impl FlashReader for PartReader {
    fn read_at(&self, off: usize, buf: &mut [u8]) -> usize {
        if off >= self.len {
            return 0;
        }
        let n = (self.len - off).min(buf.len());
        if part_read(self.part, self.base + off, &mut buf[..n]) {
            n
        } else {
            0
        }
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// Memory-map the `model.bin` region of the flash partition into the address space and
/// return it as a `&'static [u8]`. The weights are read on demand through the flash
/// cache — they **never enter the heap** (the whole point: the 489 kB model can't fit
/// the ~60 kB free heap). For on-device inference (`device_llm`) over the same weights
/// the browser/WASM path streams. The mapping is created once and kept for the program
/// lifetime (we never munmap). Returns `None` if no image is flashed or mmap fails.
static MODEL_MMAP: OnceLock<Option<&'static [u8]>> = OnceLock::new();
pub fn model_mmap() -> Option<&'static [u8]> {
    *MODEL_MMAP.get_or_init(|| {
        let t = toc()?;
        let p = t.part as *const sys::esp_partition_t;
        let mut ptr: *const core::ffi::c_void = core::ptr::null();
        let mut handle: sys::esp_partition_mmap_handle_t = 0;
        let r = unsafe {
            sys::esp_partition_mmap(
                p,
                t.model_off,
                t.model_len,
                sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
                &mut ptr,
                &mut handle,
            )
        };
        if r != sys::ESP_OK || ptr.is_null() {
            println!("model_host: esp_partition_mmap failed (err {r})");
            return None;
        }
        // SAFETY: the partition is read-only flash mapped for the program lifetime; the
        // region [ptr, ptr+model_len) is valid and immutable. `handle` is intentionally
        // dropped (never munmap'd) so the mapping stays live.
        Some(unsafe { core::slice::from_raw_parts(ptr as *const u8, t.model_len) })
    })
}

/// The tokenizer blob copied into RAM (small, ~6.7 kB) for on-device inference. The
/// browser path streams it instead via [`tokenizer_source`].
pub fn tokenizer_bytes() -> Option<Vec<u8>> {
    let t = toc()?;
    let mut buf = vec![0u8; t.tok_len];
    if part_read(t.part, t.tok_off, &mut buf) {
        Some(buf)
    } else {
        None
    }
}

/// `BodySource` for the model weights blob, or `None` if no image is flashed.
pub fn model_source() -> Option<BodySource> {
    let t = toc()?;
    Some(BodySource::Flash {
        reader: Box::new(PartReader { part: t.part, base: t.model_off, len: t.model_len }),
        len: t.model_len,
    })
}

/// `BodySource` for the tokenizer blob, or `None` if no image is flashed.
pub fn tokenizer_source() -> Option<BodySource> {
    let t = toc()?;
    Some(BodySource::Flash {
        reader: Box::new(PartReader { part: t.part, base: t.tok_off, len: t.tok_len }),
        len: t.tok_len,
    })
}

/// `BodySource` for the word-metadata JSON (tokens + POS types), or `None` if the
/// image has no v2 blobs.
pub fn wordmeta_source() -> Option<BodySource> {
    let t = toc()?;
    if t.wordmeta_len == 0 {
        return None;
    }
    Some(BodySource::Flash {
        reader: Box::new(PartReader { part: t.part, base: t.wordmeta_off, len: t.wordmeta_len }),
        len: t.wordmeta_len,
    })
}

/// `BodySource` for the int8 embedding matrix (`vocab*dim` bytes), or `None`.
pub fn embeddings_source() -> Option<BodySource> {
    let t = toc()?;
    if t.embed_len == 0 {
        return None;
    }
    Some(BodySource::Flash {
        reader: Box::new(PartReader { part: t.part, base: t.embed_off, len: t.embed_len }),
        len: t.embed_len,
    })
}
