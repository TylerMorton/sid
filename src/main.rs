use std::fs::read;
use std::path::PathBuf;

use whisper_rs::{convert_integer_to_float_audio, FullParams, WhisperContext};

use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    model: Option<PathBuf>,
    #[arg(short, long)]
    file: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    let model_path = cli.model.as_deref().expect("No model supplied");
    let file_path = cli.file.as_deref().expect("No audio file supplied");

    let ctx = WhisperContext::new(model_path.to_str().expect("path not valid utf-8"))
        .expect("failed to load model");
    let mut state = ctx.create_state().expect("failed to create state");

    let mut params = FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(1);
    params.set_translate(true);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // let mut file = File::open(path).expect("failure to open file");
    let audio_data = read(file_path).expect("couldn't read data");
    let audio_data: Vec<i16> = audio_data
        .chunks_exact(2)
        .map(|a| i16::from_ne_bytes([a[0], a[1]]))
        .collect();
    let audio_data = convert_integer_to_float_audio(&audio_data);
    state
        .full(params, &audio_data[..])
        .expect("failed to run model");

    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get segment start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failte to get segment end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
}
