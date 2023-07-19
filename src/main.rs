use async_openai::types::{CreateImageRequestArgs, ImageSize, ResponseFormat};
use async_openai::Client;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample};
use hound::WavWriter;
use std::error::Error;
use std::fs::read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use whisper_rs::{convert_integer_to_float_audio, FullParams, WhisperContext};

use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    model: Option<PathBuf>,
    #[arg(short, long)]
    file: Option<PathBuf>,
}

fn stt(model_path: &Path, file_path: &Path) -> Vec<String> {
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
    let mut phrase: Vec<String> = Vec::new();
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
        phrase.push(segment);
    }
    phrase
}

async fn tti(prompt: String) -> Result<(), Box<dyn Error>> {
    let client = Client::new();

    let request = CreateImageRequestArgs::default()
        .prompt(prompt)
        .n(1)
        .response_format(ResponseFormat::Url)
        .size(ImageSize::S256x256)
        .user("async-openai")
        .build()
        .expect("request error");

    let response = client.images().create(request).await?;
    let paths = response.save("./data").await?;
    paths
        .iter()
        .for_each(|path| println!("Image file path: {}", path.display()));
    Ok(())
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_float() {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    }
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate().0 as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}

type WavWriterHandle = Arc<Mutex<Option<WavWriter<std::io::BufWriter<std::fs::File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}

fn recording(
    writer: Arc<Mutex<Option<WavWriter<std::io::BufWriter<std::fs::File>>>>>,
    config: cpal::SupportedStreamConfig,
    device: cpal::Device,
) -> Result<(), anyhow::Error> {
    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<i8, i8>(data, &writer),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<i16, i16>(data, &writer),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<i32, i32>(data, &writer),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| write_input_data::<f32, f32>(data, &writer),
            err_fn,
            None,
        )?,
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            )))
        }
    };
    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs(3));
    drop(stream);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("failed to find input device");
    println!(
        "Input device: {}",
        device.name().expect("failed to located device name")
    );
    let config = device
        .default_input_config()
        .expect("failed to get default input config");
    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recoded.wav");
    let spec = wav_spec_from_config(&config);
    let writer = hound::WavWriter::create(PATH, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_2: Arc<Mutex<Option<WavWriter<std::io::BufWriter<std::fs::File>>>>> = writer.clone();

    recording(writer_2, config, device)?;
    writer.lock().unwrap().take().unwrap().finalize()?;
    println!("Recording {} complete!", PATH);

    // let model_path = cli.model.as_deref().expect("No model supplied");
    // let file_path = cli.file.as_deref().expect("No audio file supplied");
    // let text = stt(model_path, file_path);
    // let prompt = text.iter().fold("".to_string(), |acc, x| acc + x);
    // comment out if you don't need or else you will get charged!!!
    // tti(prompt);
    Ok(())
}
