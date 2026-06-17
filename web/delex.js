// Delexicalization for the browser/device inference path — the EXACT contract of
// train/delex.py (verified equal by train/parity_delex test). The tiny model never has
// to reproduce a literal: we replace numbers/quoted strings in the English prompt with
// slot markers (#0..#7 numbers, &a..&d strings), let the model emit the template, then
// copy the real literals back into the slots. "The model points, the browser copies."
//
// Contract (must match Python):
//   delexEn(en): strings first ("..."/'...') -> &a.. ; then numbers (-?\d+(\.\d+)?) -> #0..
//   slots assigned in order of first appearance, deduped by value/content, all occurrences
//   replaced. relex(code, nums, strs) substitutes the markers back.
export const NUM_SLOTS = 8, STR_SLOTS = 4
const NUM_PH = Array.from({ length: NUM_SLOTS }, (_, i) => '#' + i)
const STR_PH = Array.from({ length: STR_SLOTS }, (_, i) => '&' + String.fromCharCode(97 + i))

export function delexEn(en) {
  const nums = [], strs = []
  let s = en.replace(/(['"])(.*?)\1/g, (m, q, content) => {
    let i = strs.indexOf(content)
    if (i < 0) { if (strs.length >= STR_SLOTS) return m; strs.push(content); i = strs.length - 1 }
    return q + STR_PH[i] + q
  })
  s = s.replace(/-?\d+(?:\.\d+)?/g, (m) => {
    let i = nums.indexOf(m)
    if (i < 0) { if (nums.length >= NUM_SLOTS) return m; nums.push(m); i = nums.length - 1 }
    return NUM_PH[i]
  })
  return { prompt: s, nums, strs }
}

export function relex(code, nums, strs) {
  // Fill known slots; unfilled markers (the model over-generated, e.g. extra list
  // elements) become \0 and are dropped with an adjacent comma so lists stay valid.
  let s = code.replace(/#[0-7]|&[a-d]/g, (p) => {
    if (p[0] === '#') { const i = +p[1]; return i < nums.length ? nums[i] : '\0' }
    const i = p.charCodeAt(1) - 97; return i < strs.length ? strs[i] : '\0'
  })
  return s.replace(/\s*,\s*\0/g, '').replace(/\0\s*,\s*/g, '').replace(/\0/g, '')
}
