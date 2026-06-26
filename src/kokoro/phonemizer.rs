//! Pure Rust G2P phonemizer for Kokoro TTS.
//!
//! Uses misaki-compatible IPA dictionaries (us_gold.json + us_silver.json).
//! No external dependencies (espeak-ng etc).

use std::collections::HashMap;

pub struct Phonemizer {
    dict: HashMap<String, String>,
    vocab: HashMap<String, usize>,
}

impl Phonemizer {
    pub fn new(gold_json: &str, silver_json: &str, vocab: &HashMap<String, usize>) -> anyhow::Result<Self> {
        let mut dict = HashMap::new();

        // Load gold dictionary (may contain POS-tagged entries as objects)
        let gold: serde_json::Value = serde_json::from_str(gold_json)?;
        if let serde_json::Value::Object(map) = gold {
            for (word, val) in map {
                let ipa = match val {
                    serde_json::Value::String(s) => s,
                    serde_json::Value::Object(obj) => {
                        // Use DEFAULT pronunciation, fall back to first non-null
                        obj.get("DEFAULT")
                            .and_then(|v| v.as_str().map(String::from))
                            .or_else(|| {
                                obj.values()
                                    .find_map(|v| v.as_str().map(String::from))
                            })
                            .unwrap_or_default()
                    }
                    _ => continue,
                };
                if !ipa.is_empty() {
                    dict.insert(word.to_lowercase(), ipa);
                }
            }
        }

        // Load silver dictionary (always plain strings)
        let silver: HashMap<String, String> = serde_json::from_str(silver_json)?;
        for (word, ipa) in silver {
            dict.entry(word.to_lowercase()).or_insert(ipa);
        }

        Ok(Self { dict, vocab: vocab.clone() })
    }

    pub fn phonemize(&self, text: &str) -> Vec<u32> {
        let ipa = self.to_ipa(text);
        tokenize_ipa(&ipa, &self.vocab)
    }

    pub fn to_ipa(&self, text: &str) -> String {
        self.text_to_ipa(text)
    }

    fn text_to_ipa(&self, text: &str) -> String {
        let mut result = String::new();
        let normalized = normalize_text(text);
        let segments = split_segments(&normalized);

        for segment in &segments {
            match segment {
                Segment::Word(w) => {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    let (ipa, _is_acronym) = self.word_to_ipa_tagged(w);
                    result.push_str(&ipa);
                }
                Segment::Punct(ch) => {
                    match ch {
                        '%' => result.push_str("pəɹsˈɛnt"),
                        '\u{00B7}' => result.push_str("pˈYnt"),
                        _ => result.push(*ch),
                    }
                }
                Segment::Space => {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                }
            }
        }
        result
    }

    /// Returns (ipa, is_spelled_acronym).
    fn word_to_ipa_tagged(&self, word: &str) -> (String, bool) {
        let lower = word.to_lowercase();

        // Uppercase letter+digit identifiers are read character-by-character:
        // R2D2, V2, R2, R3000, etc.
        if is_spelled_alphanumeric(word) {
            return (spell_out(word), true);
        }

        // Mixed case: split at case boundaries (e.g. macOS→"mac"+"OS", DBApp→"DB"+"App")
        if word.chars().any(|c| c.is_ascii_uppercase()) && word.chars().any(|c| c.is_ascii_lowercase()) {
            let parts = split_camel_case(word);
            let mut ipa_parts = Vec::new();
            let mut any_spelled = false;
            for part in &parts {
                if part.chars().all(|c| c.is_ascii_uppercase()) {
                    ipa_parts.push(spell_out(part));
                    any_spelled = true;
                } else {
                    ipa_parts.push(self.word_to_ipa_simple(&part.to_lowercase()));
                }
            }
            return (ipa_parts.join(" "), any_spelled);
        }

        // All-uppercase: short (2-3 chars) always spell out as acronym,
        // longer ones check dictionary first (e.g. NASA)
        if word.len() > 1 && word.chars().all(|c| c.is_ascii_uppercase()) {
            if word.len() > 3 {
                if let Some(ipa) = self.dict.get(&lower) {
                    return (ipa.clone(), false);
                }
            }
            return (spell_out(word), true);
        }

        // Direct lookup
        if let Some(ipa) = self.dict.get(&lower) {
            return (ipa.clone(), false);
        }

        // Try stripping common suffixes and rebuilding
        if let Some(ipa) = self.try_suffix_rules(&lower) {
            return (ipa, false);
        }

        // Try compound word split
        if let Some(ipa) = self.try_compound_split(&lower) {
            return (ipa, false);
        }

        // Fallback: emit characters that exist in vocab
        let fallback: String = lower.chars().filter(|c| {
            let s = c.to_string();
            self.vocab.contains_key(&s)
        }).map(|c| c.to_string()).collect::<Vec<_>>().join("");
        (fallback, false)
    }

    /// Simple word→IPA without acronym/compound logic (for sub-parts).
    fn word_to_ipa_simple(&self, word: &str) -> String {
        if let Some(ipa) = self.dict.get(word) {
            return ipa.clone();
        }
        if let Some(ipa) = self.try_suffix_rules(word) {
            return ipa;
        }
        if let Some(ipa) = self.try_compound_split(word) {
            return ipa;
        }
        word.chars().filter(|c| {
            let s = c.to_string();
            self.vocab.contains_key(&s)
        }).map(|c| c.to_string()).collect::<Vec<_>>().join("")
    }

    /// Try splitting word into two known dictionary words (prefer longer suffix).
    fn try_compound_split(&self, word: &str) -> Option<String> {
        for split in 3..word.len().saturating_sub(2) {
            let prefix = &word[..split];
            let suffix = &word[split..];
            let prefix_ipa = self.dict.get(prefix)
                .cloned()
                .or_else(|| self.try_suffix_rules(prefix));
            if let Some(p) = prefix_ipa {
                let suffix_ipa = self.dict.get(suffix)
                    .cloned()
                    .or_else(|| self.try_suffix_rules(suffix));
                if let Some(s) = suffix_ipa {
                    return Some(format!("{} {}", p, s));
                }
            }
        }
        None
    }


    fn try_suffix_rules(&self, word: &str) -> Option<String> {
        // -ing: try base, base+e
        if let Some(base) = word.strip_suffix("ing") {
            if let Some(ipa) = self.dict.get(base) {
                return Some(format!("{}ɪŋ", strip_trailing_stress(ipa)));
            }
            let with_e = format!("{}e", base);
            if let Some(ipa) = self.dict.get(&with_e) {
                return Some(format!("{}ɪŋ", strip_trailing_stress(ipa)));
            }
            // doubled consonant: running -> run
            if base.len() >= 2 {
                let bytes = base.as_bytes();
                if bytes[bytes.len() - 1] == bytes[bytes.len() - 2] {
                    let shorter = &base[..base.len() - 1];
                    if let Some(ipa) = self.dict.get(shorter) {
                        return Some(format!("{}ɪŋ", strip_trailing_stress(ipa)));
                    }
                }
            }
        }

        // -s/-es: try base
        if let Some(base) = word.strip_suffix("es") {
            if let Some(ipa) = self.dict.get(base) {
                return Some(format!("{}ɪz", ipa));
            }
        }
        if let Some(base) = word.strip_suffix('s') {
            if let Some(ipa) = self.dict.get(base) {
                let suffix = if ends_voiced(ipa) { "z" } else { "s" };
                return Some(format!("{}{}", ipa, suffix));
            }
        }

        // -ied (y→i): verified → verify
        if let Some(base) = word.strip_suffix("ied") {
            let with_y = format!("{}y", base);
            if let Some(ipa) = self.dict.get(&with_y) {
                let suffix = if ends_alveolar_stop(ipa) { "ᵻd" } else if ends_voiced(ipa) { "d" } else { "t" };
                return Some(format!("{}{}", ipa, suffix));
            }
        }

        // -ed: try base
        if let Some(base) = word.strip_suffix("ed") {
            if let Some(ipa) = self.dict.get(base) {
                let suffix = if ends_alveolar_stop(ipa) { "ᵻd" } else if ends_voiced(ipa) { "d" } else { "t" };
                return Some(format!("{}{}", ipa, suffix));
            }
            let with_e = format!("{}e", base);
            if let Some(ipa) = self.dict.get(&with_e) {
                let suffix = if ends_alveolar_stop(ipa) { "ᵻd" } else if ends_voiced(ipa) { "d" } else { "t" };
                return Some(format!("{}{}", ipa, suffix));
            }
            // doubled consonant
            if base.len() >= 2 {
                let bytes = base.as_bytes();
                if bytes[bytes.len() - 1] == bytes[bytes.len() - 2] {
                    let shorter = &base[..base.len() - 1];
                    if let Some(ipa) = self.dict.get(shorter) {
                        let suffix = if ends_voiced(ipa) { "d" } else { "t" };
                        return Some(format!("{}{}", ipa, suffix));
                    }
                }
            }
        }

        // -ly
        if let Some(base) = word.strip_suffix("ly") {
            if let Some(ipa) = self.dict.get(base) {
                return Some(format!("{}li", ipa));
            }
        }

        // -er
        if let Some(base) = word.strip_suffix("er") {
            if let Some(ipa) = self.dict.get(base) {
                return Some(format!("{}əɹ", ipa));
            }
            let with_e = format!("{}e", base);
            if let Some(ipa) = self.dict.get(&with_e) {
                return Some(format!("{}əɹ", strip_trailing_stress(ipa)));
            }
        }

        // -est
        if let Some(base) = word.strip_suffix("est") {
            if let Some(ipa) = self.dict.get(base) {
                return Some(format!("{}ɪst", ipa));
            }
        }

        // -ness
        if let Some(base) = word.strip_suffix("ness") {
            if let Some(ipa) = self.dict.get(base) {
                return Some(format!("{}nɪs", ipa));
            }
        }

        None
    }
}

fn spell_out(word: &str) -> String {
    let letters: Vec<&'static str> = word.chars()
        .filter_map(spelled_char_ipa)
        .collect();
    let n = letters.len();
    if n <= 2 {
        return letters.join("-");
    }
    let mut parts = Vec::with_capacity(n);
    for (i, ipa) in letters.iter().enumerate() {
        if i < n - 1 {
            parts.push(ipa.replace('ˈ', "ˌ"));
        } else {
            parts.push(ipa.to_string());
        }
    }
    parts.join("-")
}

fn is_spelled_alphanumeric(word: &str) -> bool {
    let mut has_alpha = false;
    let mut has_digit = false;

    for ch in word.chars() {
        if ch.is_ascii_digit() {
            has_digit = true;
        } else if ch.is_ascii_alphabetic() {
            has_alpha = true;
            if ch.is_ascii_lowercase() {
                return false;
            }
        } else {
            return false;
        }
    }

    has_alpha && has_digit
}

/// Split a mixed-case word at case boundaries.
/// "macOS" → ["mac", "OS"], "DBApp" → ["DB", "App"], "iPhone" → ["i", "Phone"]
fn split_camel_case(word: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();

    for ch in word.chars() {
        if ch.is_ascii_uppercase() {
            if !current.is_empty() && current.chars().last().is_some_and(|c| c.is_ascii_lowercase()) {
                parts.push(std::mem::take(&mut current));
            }
            current.push(ch);
        } else {
            // Lowercase after multiple uppercase: split before last uppercase
            // e.g. "DB" + "a" → keep "DB" separate if current is all-upper and len>1
            if current.len() > 1 && current.chars().all(|c| c.is_ascii_uppercase()) {
                let last = current.pop().unwrap();
                parts.push(std::mem::take(&mut current));
                current.push(last);
            }
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

fn spelled_char_ipa(ch: char) -> Option<&'static str> {
    letter_ipa(ch).or_else(|| digit_ipa(ch))
}

fn letter_ipa(ch: char) -> Option<&'static str> {
    match ch.to_ascii_lowercase() {
        'a' => Some("ˈA"),
        'b' => Some("bˈi"),
        'c' => Some("sˈi"),
        'd' => Some("dˈi"),
        'e' => Some("ˈi"),
        'f' => Some("ˈɛf"),
        'g' => Some("ʤˈi"),
        'h' => Some("ˈAʧ"),
        'i' => Some("ˈI"),
        'j' => Some("ʤˈA"),
        'k' => Some("kˈA"),
        'l' => Some("ˈɛl"),
        'm' => Some("ˈɛm"),
        'n' => Some("ˈɛn"),
        'o' => Some("ˈO"),
        'p' => Some("pˈi"),
        'q' => Some("kjˈu"),
        'r' => Some("ˈɑɹ"),
        's' => Some("ˈɛs"),
        't' => Some("tˈi"),
        'u' => Some("jˈu"),
        'v' => Some("vˈi"),
        'w' => Some("dˈʌbəlˌu"),
        'x' => Some("ˈɛks"),
        'y' => Some("wˈI"),
        'z' => Some("zˈi"),
        _ => None,
    }
}

fn digit_ipa(ch: char) -> Option<&'static str> {
    match ch {
        '0' => Some("zˈɪɹO"),
        '1' => Some("wˈʌn"),
        '2' => Some("tˈu"),
        '3' => Some("θɹˈi"),
        '4' => Some("fˈɔɹ"),
        '5' => Some("fˈIv"),
        '6' => Some("sˈɪks"),
        '7' => Some("sˈɛvən"),
        '8' => Some("ˈAt"),
        '9' => Some("nˈIn"),
        _ => None,
    }
}

fn strip_trailing_stress(ipa: &str) -> &str {
    ipa.trim_end_matches('ˈ').trim_end_matches('ˌ')
}

fn ends_voiced(ipa: &str) -> bool {
    let last = ipa.chars().last().unwrap_or(' ');
    // Vowels and voiced consonants
    matches!(last, 'a' | 'e' | 'i' | 'o' | 'u' | 'æ' | 'ɑ' | 'ɔ' | 'ə' | 'ɛ' | 'ɪ' | 'ʊ' | 'ʌ'
        | 'b' | 'd' | 'g' | 'v' | 'z' | 'ʒ' | 'ð' | 'm' | 'n' | 'ŋ' | 'l' | 'ɹ' | 'w' | 'j'
        | 'A' | 'I' | 'O' | 'W' | 'Y' | 'ᵻ')
}

fn ends_alveolar_stop(ipa: &str) -> bool {
    let last = ipa.chars().last().unwrap_or(' ');
    matches!(last, 't' | 'd')
}

enum Segment {
    Word(String),
    Punct(char),
    Space,
}

fn split_segments(text: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current_word = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            current_word.push(ch);
        } else {
            if !current_word.is_empty() {
                segments.push(Segment::Word(std::mem::take(&mut current_word)));
            }
            if ch.is_whitespace() {
                segments.push(Segment::Space);
            } else {
                segments.push(Segment::Punct(ch));
            }
        }
    }
    if !current_word.is_empty() {
        segments.push(Segment::Word(current_word));
    }
    segments
}

fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut prev_input = None;
    while let Some(ch) = chars.next() {
        match ch {
            '\u{2018}' | '\u{2019}' => {
                out.push('\'');
                prev_input = Some(ch);
            }
            '\u{201C}' | '\u{201D}' => {
                out.push('"');
                prev_input = Some(ch);
            }
            '\u{2014}' => {
                out.push('—');
                prev_input = Some(ch);
            }
            '\u{2026}' => {
                out.push('…');
                prev_input = Some(ch);
            }
            '0'..='9' => {
                let mut num_str = String::new();
                num_str.push(ch);
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() {
                        num_str.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if prev_input.is_some_and(|prev: char| prev.is_ascii_uppercase()) {
                    out.push_str(&num_str);
                    prev_input = num_str.chars().last();
                    continue;
                }
                // Check for decimal point followed by more digits
                if chars.peek() == Some(&'.') {
                    let mut lookahead = chars.clone();
                    lookahead.next(); // consume '.'
                    if lookahead.peek().is_some_and(|c| c.is_ascii_digit()) {
                        chars.next(); // consume '.'
                        let mut frac_str = String::new();
                        while let Some(&next) = chars.peek() {
                            if next.is_ascii_digit() {
                                frac_str.push(next);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        out.push_str(&number_to_words(&num_str));
                        out.push('\u{00B7}'); // middle dot as decimal point marker
                        let frac_words: Vec<String> = frac_str.chars()
                            .map(|d| number_to_words(&d.to_string()))
                            .collect();
                        out.push_str(&frac_words.join(" "));
                    } else {
                        out.push_str(&number_to_words(&num_str));
                    }
                } else {
                    out.push_str(&number_to_words(&num_str));
                }
                // Expand trailing unit suffixes
                if let Some(suffix) = consume_unit_suffix(&mut chars) {
                    out.push_str(&suffix);
                }
                prev_input = num_str.chars().last();
            }
            _ => {
                out.push(ch);
                prev_input = Some(ch);
            }
        }
    }
    out
}

fn consume_unit_suffix(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<String> {
    let next = *chars.peek()?;
    match next {
        '%' => { chars.next(); Some("%".into()) }
        'k' | 'K' => {
            chars.next();
            Some(expand_data_unit(chars, "kilo", "K").into())
        }
        'M' => {
            chars.next();
            Some(expand_data_unit(chars, "mega", "mega").into())
        }
        'G' => {
            chars.next();
            Some(expand_data_unit(chars, "giga", "giga").into())
        }
        'T' => {
            chars.next();
            Some(expand_data_unit(chars, "tera", "tera").into())
        }
        'm' => {
            let mut lookahead = chars.clone();
            lookahead.next();
            if lookahead.peek() == Some(&'s') {
                chars.next(); chars.next();
                Some(" milliseconds".into())
            } else {
                None
            }
        }
        'H' => {
            let mut lookahead = chars.clone();
            lookahead.next();
            if lookahead.peek() == Some(&'z') {
                chars.next(); chars.next();
                Some(" hertz".into())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn expand_data_unit(chars: &mut std::iter::Peekable<std::str::Chars>, prefix: &str, bare: &str) -> String {
    let unit = match chars.peek() {
        Some(&'B') => { chars.next(); format!(" {prefix}byte") }
        Some(&'b') => { chars.next(); format!(" {prefix}bit") }
        Some(&'H') => {
            let mut lookahead = chars.clone();
            lookahead.next();
            if lookahead.peek() == Some(&'z') {
                chars.next(); chars.next();
                return format!(" {prefix}hertz");
            } else {
                return format!(" {bare}");
            }
        }
        _ => return format!(" {bare}"),
    };
    // Check for /s (per second) rate suffix
    if chars.peek() == Some(&'/') {
        let mut lookahead = chars.clone();
        lookahead.next();
        if lookahead.peek() == Some(&'s') {
            chars.next(); chars.next();
            return format!("{unit}s per second");
        }
    }
    format!("{unit}s")
}

fn number_to_words(s: &str) -> String {
    let n: u64 = match s.parse() {
        Ok(v) => v,
        Err(_) => return s.to_string(),
    };
    if n == 0 {
        return "zero".to_string();
    }
    int_to_words(n)
}

fn int_to_words(n: u64) -> String {
    const ONES: &[&str] = &[
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ];
    const TENS: &[&str] = &[
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    if n == 0 {
        return String::new();
    }
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let t = TENS[(n / 10) as usize].to_string();
        let o = n % 10;
        if o > 0 {
            return format!("{} {}", t, ONES[o as usize]);
        }
        return t;
    }
    if n < 1000 {
        let h = format!("{} hundred", ONES[(n / 100) as usize]);
        let rem = n % 100;
        if rem > 0 {
            return format!("{} {}", h, int_to_words(rem));
        }
        return h;
    }
    if n < 1_000_000 {
        let t = format!("{} thousand", int_to_words(n / 1000));
        let rem = n % 1000;
        if rem > 0 {
            return format!("{} {}", t, int_to_words(rem));
        }
        return t;
    }
    if n < 1_000_000_000 {
        let m = format!("{} million", int_to_words(n / 1_000_000));
        let rem = n % 1_000_000;
        if rem > 0 {
            return format!("{} {}", m, int_to_words(rem));
        }
        return m;
    }
    let b = format!("{} billion", int_to_words(n / 1_000_000_000));
    let rem = n % 1_000_000_000;
    if rem > 0 {
        return format!("{} {}", b, int_to_words(rem));
    }
    b
}

/// Tokenize an IPA string into Kokoro vocab token IDs.
pub fn tokenize_ipa(ipa: &str, vocab: &HashMap<String, usize>) -> Vec<u32> {
    let mut tokens = Vec::new();
    for ch in ipa.chars() {
        let key = ch.to_string();
        if let Some(&id) = vocab.get(&key) {
            tokens.push(id as u32);
        }
    }
    tokens
}

/// Legacy entry point — builds a Phonemizer and uses it.
pub fn phonemize(text: &str, vocab: &HashMap<String, usize>) -> Vec<u32> {
    // Without dictionaries, fall back to character-level tokenization
    tokenize_ipa(&text.to_lowercase(), vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_phonemizer() -> Phonemizer {
        Phonemizer {
            dict: HashMap::new(),
            vocab: HashMap::new(),
        }
    }

    #[test]
    fn spells_uppercase_alphanumeric_sequences() {
        let phonemizer = empty_phonemizer();

        assert_eq!(
            phonemizer.to_ipa("R2D2 V2 R2 R3000"),
            "ˌɑɹ-tˌu-dˌi-tˈu vˈi-tˈu ˈɑɹ-tˈu ˌɑɹ-θɹˌi-zˌɪɹO-zˌɪɹO-zˈɪɹO"
        );
    }

    #[test]
    fn normalizer_preserves_uppercase_alphanumeric_digits() {
        assert_eq!(
            normalize_text("R2D2 V2 R2 R3000"),
            "R2D2 V2 R2 R3000"
        );
    }

    #[test]
    fn normalizer_still_expands_plain_numbers_and_units() {
        assert_eq!(
            normalize_text("12 10ms 5GB"),
            "twelve ten milliseconds five gigabytes"
        );
    }
}
