use prettytable::Table;
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[macro_use]
extern crate prettytable;

fn estimate_pi_portion(n_points: usize, seed: u64) -> usize {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut inside_circle = 0;

    for _ in 0..n_points {
        let x: f64 = rng.gen_range(0.0..1.0);
        let y: f64 = rng.gen_range(0.0..1.0);
        if x * x + y * y <= 1.0 {
            inside_circle += 1;
        }
    }
    inside_circle
}

fn estimate_pi_serial(total_points: usize, seed: u64) -> f64 {
    let inside_circle = estimate_pi_portion(total_points, seed);
    4.0 * (inside_circle as f64) / (total_points as f64)
}

fn estimate_pi_parallel(total_points: usize, chunk_size: usize, base_seed: u64) -> f64 {
    let n_chunks = (total_points + chunk_size - 1) / chunk_size;
    let points_per_chunk = total_points / n_chunks;
    let remainder = total_points % n_chunks;

    let inside_circle: usize = (0..n_chunks)
        .into_par_iter()
        .map(|i| {
            let chunk_points = if i < remainder {
                points_per_chunk + 1
            } else {
                points_per_chunk
            };
            // Use base_seed and chunk index for unique seeds
            estimate_pi_portion(chunk_points, base_seed.wrapping_add(i as u64))
        })
        .sum();

    4.0 * (inside_circle as f64) / (total_points as f64)
}

fn calculate_stats(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() as f64);
    let stddev = variance.sqrt();
    (mean, stddev)
}

fn calculate_error(estimated_pi: f64) -> f64 {
    let true_pi = std::f64::consts::PI;
    ((estimated_pi - true_pi) / true_pi).abs() * 100.0
}

fn benchmark<F, R>(label: &str, f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_secs_f64();
    println!("{} took {:.4} seconds", label, duration);
    (result, duration)
}

fn run_with_threads<F>(num_threads: usize, f: F)
where
    F: FnOnce() + Send,
{
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    pool.install(f);
}

fn main() {
    let total_points = 1_000_000_000;
    let chunk_size = 1_000_000;
    let trials = 5;

    println!("Estimating π using Monte Carlo method");
    println!("Total points per trial: {}", total_points);
    println!("Expected π: {:.10}", std::f64::consts::PI);
    println!("Running {} trials for each configuration\n", trials);

    let mut csv_file = File::create("results.csv").expect("Failed to create CSV file");
    writeln!(
        csv_file,
        "mode,threads,trial,pi_estimate,time_seconds,error_percent"
    )
    .expect("Failed to write CSV header");

    let mut rng = rand::thread_rng();

    let mut serial_pis = Vec::with_capacity(trials);
    let mut serial_times = Vec::with_capacity(trials);
    let mut serial_errors = Vec::with_capacity(trials);

    println!("Running serial trials...");
    for trial in 0..trials {
        let seed: u64 = rng.gen();
        let (pi, time) = benchmark(&format!("Serial trial {}", trial + 1), || {
            estimate_pi_serial(total_points, seed)
        });

        let error = calculate_error(pi);

        writeln!(
            csv_file,
            "serial,1,{},{:.10},{:.4},{:.10}",
            trial + 1,
            pi,
            time,
            error
        )
        .expect("Failed to write to CSV");

        serial_pis.push(pi);
        serial_times.push(time);
        serial_errors.push(error);
    }

    let (serial_mean, serial_stddev) = calculate_stats(&serial_pis);
    let (serial_time_mean, serial_time_stddev) = calculate_stats(&serial_times);
    let (serial_error_mean, serial_error_stddev) = calculate_stats(&serial_errors);
    println!("\nSerial Results:");
    println!(
        "π = {:.10} ± {:.10} (error: {:.10}%)",
        serial_mean, serial_stddev, serial_error_mean
    );
    println!(
        "Time: {:.4} ± {:.4} seconds\n",
        serial_time_mean, serial_time_stddev
    );

    let thread_counts = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let mut parallel_results = vec![];

    for &num_threads in &thread_counts {
        println!("Running parallel trials with {} threads...", num_threads);
        let mut parallel_pis = Vec::with_capacity(trials);
        let mut parallel_times = Vec::with_capacity(trials);
        let mut parallel_errors = Vec::with_capacity(trials);

        for trial in 0..trials {
            let seed: u64 = rng.gen();
            run_with_threads(num_threads, || {
                let (pi, time) = benchmark(
                    &format!("Parallel trial {} ({} threads)", trial + 1, num_threads),
                    || estimate_pi_parallel(total_points, chunk_size, seed),
                );

                let error = calculate_error(pi);

                writeln!(
                    csv_file,
                    "parallel,{},{},{:.10},{:.4},{:.10}",
                    num_threads,
                    trial + 1,
                    pi,
                    time,
                    error
                )
                .expect("Failed to write to CSV");

                parallel_pis.push(pi);
                parallel_times.push(time);
                parallel_errors.push(error);
            });
        }

        let (par_mean, par_stddev) = calculate_stats(&parallel_pis);
        let (par_time_mean, par_time_stddev) = calculate_stats(&parallel_times);
        let (par_error_mean, par_error_stddev) = calculate_stats(&parallel_errors);
        println!("\nResults for {} threads:", num_threads);
        println!(
            "π = {:.10} ± {:.10} (error: {:.10}%)",
            par_mean, par_stddev, par_error_mean
        );
        println!(
            "Time: {:.4} ± {:.4} seconds",
            par_time_mean, par_time_stddev
        );

        parallel_results.push((
            num_threads,
            par_mean,
            par_stddev,
            par_time_mean,
            par_time_stddev,
            par_error_mean,
            par_error_stddev,
        ));
    }

    let mut table = Table::new();
    table.add_row(row![
        "Configuration",
        "π Estimate",
        "π Std Dev",
        "Error %",
        "Error Std Dev",
        "Time (s)",
        "Time Std Dev",
        "Speedup"
    ]);

    table.add_row(row![
        "Serial",
        format!("{:.10}", serial_mean),
        format!("{:.10}", serial_stddev),
        format!("{:.10}%", serial_error_mean),
        format!("{:.10}%", serial_error_stddev),
        format!("{:.4}", serial_time_mean),
        format!("{:.4}", serial_time_stddev),
        "1.00x"
    ]);

    for &(threads, pi_mean, pi_stddev, time_mean, time_stddev, error_mean, error_stddev) in
        &parallel_results
    {
        let speedup = serial_time_mean / time_mean;
        table.add_row(row![
            format!("{} threads", threads),
            format!("{:.10}", pi_mean),
            format!("{:.10}", pi_stddev),
            format!("{:.10}%", error_mean),
            format!("{:.10}%", error_stddev),
            format!("{:.4}", time_mean),
            format!("{:.4}", time_stddev),
            format!("{:.2}x", speedup)
        ]);
    }

    println!("\nPerformance Summary:");
    table.printstd();
}
