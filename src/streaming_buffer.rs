/// Streaming audio buffer with rolling window and overlap management
///
/// This module implements the buffering strategy used in index.html for streaming transcription:
/// - Maintains a rolling buffer of the last N seconds of audio
/// - Commits transcribed text when buffer fills
/// - Keeps overlap samples for context continuity
/// - Can be shared between CLI example and Node.js module

use std::collections::VecDeque;

pub struct StreamingBuffer {
    /// Rolling buffer of audio samples (16kHz mono)
    buffer: VecDeque<f32>,

    /// Maximum buffer size in samples
    max_buffer_samples: usize,

    /// Overlap size in samples (kept after commit for context)
    overlap_samples: usize,

    /// Samples accumulated since last commit
    samples_since_commit: usize,

    /// Committed transcript lines
    pub committed_lines: Vec<String>,

    /// Current line being built (updated on each transcription)
    pub current_line: String,
}

impl StreamingBuffer {
    /// Create a new streaming buffer
    ///
    /// # Arguments
    /// * `max_buffer_secs` - Maximum buffer duration in seconds (e.g., 10.0)
    /// * `overlap_secs` - Overlap duration in seconds (e.g., 0.25)
    /// * `sample_rate` - Audio sample rate (typically 16000)
    pub fn new(max_buffer_secs: f32, overlap_secs: f32, sample_rate: usize) -> Self {
        let max_buffer_samples = (max_buffer_secs * sample_rate as f32) as usize;
        let overlap_samples = (overlap_secs * sample_rate as f32) as usize;

        Self {
            buffer: VecDeque::with_capacity(max_buffer_samples),
            max_buffer_samples,
            overlap_samples,
            samples_since_commit: 0,
            committed_lines: Vec::new(),
            current_line: String::new(),
        }
    }

    /// Add audio samples to the rolling buffer
    ///
    /// Automatically maintains the rolling window by dropping old samples when full.
    /// Returns true if the buffer is full and should trigger a commit.
    pub fn push_samples(&mut self, samples: &[f32]) -> bool {
        // Add samples to buffer, maintaining rolling window
        for &sample in samples {
            if self.buffer.len() >= self.max_buffer_samples {
                // Buffer is full, drop oldest sample
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }

        self.samples_since_commit += samples.len();

        // Check if we should commit (buffer has rolled over)
        self.samples_since_commit >= self.max_buffer_samples
    }

    /// Get the current buffer contents for transcription
    pub fn get_buffer(&self) -> Vec<f32> {
        self.buffer.iter().copied().collect()
    }

    /// Get buffer size in samples
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Get buffer duration in seconds
    pub fn buffer_duration_secs(&self, sample_rate: usize) -> f32 {
        self.buffer.len() as f32 / sample_rate as f32
    }

    /// Update the current line with new transcription
    ///
    /// This is called after each transcription of the rolling buffer.
    pub fn update_current_line(&mut self, text: String) {
        self.current_line = text;
    }

    /// Commit the current line and trim buffer to overlap
    ///
    /// This should be called when `push_samples()` returns true (buffer full).
    /// It commits the current line to the transcript and trims the buffer to keep
    /// only the overlap samples for context continuity.
    pub fn commit_and_trim(&mut self, last_chunk_len: usize) {
        // Commit current line if it has content
        let trimmed = self.current_line.trim();
        if !trimmed.is_empty() {
            self.committed_lines.push(trimmed.to_string());
        }

        // Reset current line
        self.current_line.clear();

        // Trim buffer to keep only (last_chunk_len + overlap)
        let keep_samples = (last_chunk_len + self.overlap_samples).min(self.buffer.len());
        let drop_count = self.buffer.len().saturating_sub(keep_samples);

        for _ in 0..drop_count {
            self.buffer.pop_front();
        }

        // Reset commit counter
        self.samples_since_commit = 0;
    }

    /// Get the full transcript (committed lines + current line)
    pub fn get_full_transcript(&self) -> String {
        let mut lines = self.committed_lines.clone();
        if !self.current_line.is_empty() {
            lines.push(self.current_line.clone());
        }
        lines.join("\n")
    }

    /// Get number of committed lines
    pub fn num_committed_lines(&self) -> usize {
        self.committed_lines.len()
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.samples_since_commit = 0;
        self.committed_lines.clear();
        self.current_line.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_buffer() {
        let mut buf = StreamingBuffer::new(1.0, 0.25, 16000); // 1 second max, 0.25s overlap

        // Add 0.5 seconds of audio
        let chunk1 = vec![0.1f32; 8000];
        assert!(!buf.push_samples(&chunk1)); // Not full yet
        assert_eq!(buf.buffer_len(), 8000);

        // Add another 0.5 seconds
        let chunk2 = vec![0.2f32; 8000];
        assert!(!buf.push_samples(&chunk2)); // Exactly full
        assert_eq!(buf.buffer_len(), 16000);

        // Add more - should trigger commit
        let chunk3 = vec![0.3f32; 100];
        assert!(buf.push_samples(&chunk3)); // Should commit
        assert_eq!(buf.buffer_len(), 16000); // Rolling window maintained
    }

    #[test]
    fn test_commit_and_trim() {
        let mut buf = StreamingBuffer::new(1.0, 0.25, 16000);

        // Fill buffer
        buf.push_samples(&vec![0.1f32; 16000]);
        buf.update_current_line("Hello world".to_string());

        // Commit
        buf.commit_and_trim(1000);

        // Check committed
        assert_eq!(buf.num_committed_lines(), 1);
        assert_eq!(buf.committed_lines[0], "Hello world");
        assert_eq!(buf.current_line, "");

        // Buffer should be trimmed to (last_chunk_len + overlap) = 1000 + 4000 = 5000
        assert!(buf.buffer_len() <= 5000);
    }
}
