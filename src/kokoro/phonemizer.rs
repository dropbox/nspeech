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

        for segment in split_segments(&normalized) {
            match segment {
                Segment::Word(w) => {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push_str(&self.word_to_ipa(&w));
                }
                Segment::Punct(ch) => {
                    result.push(ch);
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

    fn word_to_ipa(&self, word: &str) -> String {
        let lower = word.to_lowercase();

        // Direct lookup
        if let Some(ipa) = self.dict.get(&lower) {
            return ipa.clone();
        }

        // Try stripping common suffixes and rebuilding
        if let Some(ipa) = self.try_suffix_rules(&lower) {
            return ipa;
        }

        // Fallback: spell out characters that exist in vocab
        lower.chars().filter(|c| {
            let s = c.to_string();
            self.vocab.contains_key(&s)
        }).map(|c| c.to_string()).collect::<Vec<_>>().join("")
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
        if ch.is_alphabetic() || ch == '\'' {
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
    while let Some(ch) = chars.next() {
        match ch {
            '\u{2018}' | '\u{2019}' => out.push('\''),
            '\u{201C}' | '\u{201D}' => out.push('"'),
            '\u{2014}' => out.push('—'),
            '\u{2026}' => out.push('…'),
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
                out.push_str(&number_to_words(&num_str));
            }
            _ => out.push(ch),
        }
    }
    out
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
