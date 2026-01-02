use anyhow::Result;
use candle_core::Device;
use std::env;
use std::path::PathBuf;
use speech::silero::{SileroVad, VadStream};

fn main() -> Result<()> {
    let device = Device::Cpu;

    let assets = PathBuf::from("assets");
    let vad = SileroVad::load(&assets, &device)?;



    // Create streaming wrapper (keeps LSTM (h,c) across pushes)
    let mut stream = VadStream::new(vad, &device)?;

    // Feed audio in small increments (10ms chunks)
    let chunk_len = 160usize; // 10ms at 16kHz
/*
    // Example: simulate “real-time” audio arriving in 10ms chunks @ 16kHz => 160 samples

    // For demo: 5 seconds of audio. Replace this with mic / file reader chunks.
    let total_samples = 5 * 16_000;
    let mut pcm = vec![0.0f32; total_samples];

    // Put a fake “speechy” region in the middle so you see non-zero probs.
    // (Just a simple tone burst; real speech will work better.)
    for i in (16_000..32_000).step_by(1) {
        let t = (i - 16_000) as f32 / 16_000.0;
        pcm[i] = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.2;
    }

    // Stream it through
    let mut idx = 0usize;
    let mut frame_idx = 0usize;

    while idx < pcm.len() {
        let end = (idx + chunk_len).min(pcm.len());
        let probs = stream.push(&pcm[idx..end])?;

        // You'll get ~1 prob per 2048 samples (~128ms) once the internal buffer is warm.
        for p in probs {
            // Convert "frame index" to time:
            // each prob corresponds to 2048 samples at 16kHz
            let t_ms = frame_idx as f32 * (2048.0 / 16_000.0) * 1000.0;
            println!("{:7.1} ms  p_speech={:.3}", t_ms, p);
            frame_idx += 1;
        }

        idx = end;
    }
    */

    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        assert!(false);
        //return Err(anyhow!("expected mono wav, got {} channels", spec.channels));
    }
    const SAMPLE_RATE: u32 = 16000;
    if spec.sample_rate != SAMPLE_RATE {
        assert!(false);
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8_388_608.0))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        _ => [].into()
    };
    if samples.is_empty() {
        assert!(false);
    }

    let pcm = samples;

    // Stream it through
    let mut idx = 0usize;
    let mut total_samples_processed = 0usize;

    while idx < pcm.len() {
        let end = (idx + chunk_len).min(pcm.len());
        let probs = stream.push(&pcm[idx..end])?;

        // Each output probability corresponds to one 512-sample chunk (32ms at 16kHz)
        for p in probs {
            let t_ms = total_samples_processed as f32 * (1000.0 / 16.0);
            println!("{:7.1} ms  p_speech={:.3}", t_ms, p);
            total_samples_processed += 512;
        }

        idx = end;
    }

    return Ok(());


/*
    let args: Vec<String> = env::args().collect();
    let mut wav_path: Option<String> = None;
    let mut hf_repo: Option<String> = None;
    let mut local_dir: Option<String> = None;
    let mut max_chunks: Option<usize> = None; // default: process all chunks; set via --max-chunks
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hf" => {
                if i + 1 < args.len() {
                    hf_repo = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--local" => {
                if i + 1 < args.len() {
                    local_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--max-chunks" => {
                if i + 1 < args.len() {
                    let n: usize = args[i + 1].parse().unwrap_or(0);
                    max_chunks = if n == 0 { None } else { Some(n) };
                    i += 1;
                }
            }
            p if p.starts_with("--") => {}
            p => {
                if wav_path.is_none() {
                    wav_path = Some(p.to_string());
                }
            }
        }
        i += 1;
    }
    let cfg = FastConformerConfig {
        feat_in: 80,
        d_model: 256,
        num_heads: 4,
        ff_mult: 4,
        num_layers: 4,
        conv_kernel_size: 31,
        dropout: 0.1,
        subsampling_channels: 128,
        subsampling_stride: 2,
        subsampling_factor: 8,
        scale_input: true,
        vocab_size: 40, // [blank] + 39 chars
        blank_id: 0,
    };

    // Choose between HF pretrained model (if provided) or random init.
    let (model, id2token_stream, use_pretrained) = if let Some(dir) = local_dir.clone() {
        let m = load_parakeet_ctc_from_local(&dir, &device)?;
        (m, Vec::new(), true)
    } else if let Some(repo) = hf_repo.clone() {
        let m = load_parakeet_ctc_from_hf(&repo, &device)?;
        (m, Vec::new(), true)
    } else {
        let mut id2token = Vec::new();
        id2token.push("<blank>".to_string());
        for c in b'a'..=b'z' {
            id2token.push((c as char).to_string());
        }
        while id2token.len() < cfg.vocab_size {
            id2token.push(format!("<{}>", id2token.len()));
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let m = ParakeetFastConformerCtc::new(cfg.clone(), vb, id2token.clone())?;
        (m, id2token, false)
    };

    // If a wav is provided, load it; otherwise use dummy data.
    if let Some(path) = wav_path {
        let chunk_frames = 200; // ~2s at 10ms hop for lower latency
        let chunks =
            stream_wav_as_feature_chunks(Path::new(&path), cfg.feat_in, chunk_frames, &device)?;
        if chunks.is_empty() {
            println!("No chunks produced from wav.");
            return Ok(());
        }

        let mut tokens: Vec<u32> = Vec::new();
        let mut prev = cfg.blank_id as u32;
        for (idx, chunk) in chunks.iter().enumerate() {
            let logits = model.forward(chunk, false)?;
            let ids = logits.argmax(D::Minus1)?.to_vec2::<u32>()?;
            // assume batch=1
            for &cur in ids[0].iter() {
                if cur == cfg.blank_id as u32 {
                    prev = cur;
                    continue;
                }
                if cur == prev {
                    continue;
                }
                tokens.push(cur);
                prev = cur;
            }
            // Decode incrementally after each chunk for immediate output.
            let text = if use_pretrained {
                model.decode_tokens(&tokens)?
            } else {
                let mut t = String::new();
                for id in &tokens {
                    let idx = *id as usize;
                    if idx < id2token_stream.len() {
                        t.push_str(&id2token_stream[idx]);
                    }
                }
                t
            };
            println!("partial decoded after chunk {}/{}: {}", idx + 1, chunks.len(), text);
            if let Some(limit) = max_chunks {
                if idx + 1 >= limit {
                    println!("stopping after {} chunks (limit)", limit);
                    break;
                }
            }
        }

        let final_text = if use_pretrained {
            model.decode_tokens(&tokens)?
        } else {
            let mut t = String::new();
            for id in tokens {
                let idx = id as usize;
                if idx < id2token_stream.len() {
                    t.push_str(&id2token_stream[idx]);
                }
            }
            t
        };
        println!("decoded (streaming): {}", final_text);
    } else {
        // Dummy batch: B=1, T=320 frames, F=80 log-mel features
        let features = Tensor::zeros((1, 320, cfg.feat_in), DType::F32, &device)?;
        let logits = model.forward(&features, false)?;
        println!("logits shape: {:?}", logits.shape());

        let transcripts = model.greedy_decode(&logits)?;
        println!("decoded: {:?}", transcripts);
    }

    Ok(())
    */
}
