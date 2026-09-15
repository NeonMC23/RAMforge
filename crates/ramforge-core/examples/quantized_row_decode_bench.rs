//! Dependency-free synthetic microbenchmark for quantized row decoders.
//!
//! Q4_0 is measured only in its retained production form. Q8_0 compares its
//! provisional direct-destination path with the former decode-and-copy path.
//!
//! Run with:
//! cargo run --release -p ramforge-core --example quantized_row_decode_bench

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ramforge_core::quant::{
    dequantize_row_q2_k, dequantize_row_q3_k, dequantize_row_q4_0, dequantize_row_q4_k,
    dequantize_row_q5_k, dequantize_row_q6_k, dequantize_row_q8_0, dequantize_row_q8_k,
    BlockQ8_0, BLOCK_SIZE_Q2_K, BLOCK_SIZE_Q3_K, BLOCK_SIZE_Q4_0,
    BLOCK_SIZE_Q4_K, BLOCK_SIZE_Q5_K, BLOCK_SIZE_Q6_K, BLOCK_SIZE_Q8_0, BLOCK_SIZE_Q8_K,
    QK4_0, QK8_0, QK_K,
};
use ramforge_core::DataSourceError;

const ELEMENTS_PER_ROW: usize = 2_048;
const ROWS: usize = 8;
const WARMUP_ITERATIONS: usize = 10;
const ITERATIONS_PER_SAMPLE: usize = 100;
const SAMPLES: usize = 5;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this allocator delegates the request unchanged to `System`.
        let pointer = unsafe { System.alloc(layout) };
        if TRACK_ALLOCATIONS.load(Ordering::SeqCst) && !pointer.is_null() {
            ALLOCATION_CALLS.fetch_add(1, Ordering::SeqCst);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::SeqCst);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this allocator delegates the request unchanged to `System`.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if TRACK_ALLOCATIONS.load(Ordering::SeqCst) && !pointer.is_null() {
            ALLOCATION_CALLS.fetch_add(1, Ordering::SeqCst);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::SeqCst);
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the original pointer/layout and requested size are forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if TRACK_ALLOCATIONS.load(Ordering::SeqCst) && !new_pointer.is_null() {
            REALLOCATION_CALLS.fetch_add(1, Ordering::SeqCst);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::SeqCst);
        }
        new_pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if TRACK_ALLOCATIONS.load(Ordering::SeqCst) {
            DEALLOCATION_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: this allocator returns the pointer/layout pair to `System` unchanged.
        unsafe { System.dealloc(pointer, layout) };
    }
}

type Decoder = fn(&[u8], usize, &mut [f32]) -> Result<(), DataSourceError>;

struct Workload {
    bytes: Vec<u8>,
    row_bytes: usize,
    blocks_per_row: usize,
}

#[derive(Clone, Copy)]
struct Sample {
    elapsed: Duration,
    allocation_calls: u64,
    reallocation_calls: u64,
    deallocation_calls: u64,
    allocated_bytes: u64,
}

struct BenchResult {
    median: Duration,
    allocation_calls: u64,
    reallocation_calls: u64,
}

fn main() {
    assert!(
        !cfg!(debug_assertions),
        "run this microbenchmark with cargo run --release"
    );

    let q4_0 = build_workload(QK4_0, BLOCK_SIZE_Q4_0, make_q4_0_block);
    let q8_0 = build_workload(QK8_0, BLOCK_SIZE_Q8_0, make_q8_0_block);
    let q4_k = build_workload(QK_K, BLOCK_SIZE_Q4_K, make_q4_k_block);
    let q5_k = build_workload(QK_K, BLOCK_SIZE_Q5_K, make_q5_k_block);
    let q6_k = build_workload(QK_K, BLOCK_SIZE_Q6_K, make_q6_k_block);
    let q2_k = build_workload(QK_K, BLOCK_SIZE_Q2_K, make_q2_k_block);
    let q3_k = build_workload(QK_K, BLOCK_SIZE_Q3_K, make_q3_k_block);
    let q8_k = build_workload(QK_K, BLOCK_SIZE_Q8_K, make_q8_k_block);

    assert_reference_parity(
        "Q8_0",
        dequantize_row_q8_0_reference,
        dequantize_row_q8_0,
        &q8_0,
    );

    let baselines: [(&str, Decoder, &Workload); 6] = [
        ("Q4_K", dequantize_row_q4_k, &q4_k),
        ("Q5_K", dequantize_row_q5_k, &q5_k),
        ("Q6_K", dequantize_row_q6_k, &q6_k),
        ("Q2_K", dequantize_row_q2_k, &q2_k),
        ("Q3_K", dequantize_row_q3_k, &q3_k),
        ("Q8_K", dequantize_row_q8_k, &q8_k),
    ];
    for (name, decoder, workload) in baselines {
        validate_decoder(name, decoder, workload);
    }

    println!("RAMforge synthetic quantized row-decode benchmark");
    println!("  rows per iteration:       {ROWS}");
    println!("  decoded elements per row: {ELEMENTS_PER_ROW}");
    println!("  warmup iterations:        {WARMUP_ITERATIONS}");
    println!("  iterations per sample:    {ITERATIONS_PER_SAMPLE}");
    println!("  measured samples:         {SAMPLES}");
    println!(
        "  decoded elements/sample: {}",
        ROWS * ELEMENTS_PER_ROW * ITERATIONS_PER_SAMPLE
    );
    println!("  parity: Q8_0 production direct path matches its reference exactly");
    println!();
    println!(
        "{:<15} {:>9} {:>19} {:>14} {:>14} {:>14} {:>17} {:>12}",
        "decoder",
        "blocks/row",
        "median ms [min,max]",
        "M elem/s",
        "input MiB/s",
        "output GiB/s",
        "alloc/realloc/free",
        "alloc KiB"
    );

    let q4_production = benchmark("Q4_0 production", dequantize_row_q4_0, &q4_0);
    let q8_reference = benchmark(
        "Q8_0 reference",
        dequantize_row_q8_0_reference,
        &q8_0,
    );
    let q8_direct = benchmark("Q8_0 direct", dequantize_row_q8_0, &q8_0);

    for (name, result) in [
        ("Q4_0 production", &q4_production),
        ("Q8_0 reference", &q8_reference),
        ("Q8_0 direct", &q8_direct),
    ] {
        assert_eq!(result.allocation_calls, 0, "{name} allocated");
        assert_eq!(result.reallocation_calls, 0, "{name} reallocated");
    }

    for (name, decoder, workload) in baselines {
        benchmark(name, decoder, workload);
    }

    println!();
    print_relative_difference("Q8_0", q8_reference.median, q8_direct.median);
    println!(
        "Allocation counters use separate decode loops; timed samples disable counting."
    );
    println!("Inputs and output buffers are preallocated outside the measured regions.");
}

fn build_workload(
    elements_per_block: usize,
    block_size: usize,
    mut make_block: impl FnMut(usize) -> Vec<u8>,
) -> Workload {
    assert_eq!(ELEMENTS_PER_ROW % elements_per_block, 0);
    let blocks_per_row = ELEMENTS_PER_ROW / elements_per_block;
    let row_bytes = blocks_per_row * block_size;
    let mut bytes = Vec::with_capacity(ROWS * row_bytes);
    for row in 0..ROWS {
        for block in 0..blocks_per_row {
            let encoded = make_block(row * blocks_per_row + block);
            assert_eq!(encoded.len(), block_size);
            bytes.extend_from_slice(&encoded);
        }
    }
    Workload {
        bytes,
        row_bytes,
        blocks_per_row,
    }
}

fn assert_reference_parity(
    name: &str,
    reference: Decoder,
    direct: Decoder,
    workload: &Workload,
) {
    let mut expected = vec![0.0f32; ELEMENTS_PER_ROW];
    let mut actual = vec![f32::NAN; ELEMENTS_PER_ROW];
    for row in workload.bytes.chunks_exact(workload.row_bytes) {
        reference(row, ELEMENTS_PER_ROW, &mut expected).unwrap();
        direct(row, ELEMENTS_PER_ROW, &mut actual).unwrap();
        for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "{name} parity mismatch at element {index}"
            );
        }
        assert!(actual.iter().all(|value| !value.is_nan()));
        actual.fill(f32::NAN);
    }
}

fn validate_decoder(name: &str, decoder: Decoder, workload: &Workload) {
    let mut output = vec![f32::NAN; ELEMENTS_PER_ROW];
    for row in workload.bytes.chunks_exact(workload.row_bytes) {
        decoder(row, ELEMENTS_PER_ROW, &mut output).unwrap();
        assert!(
            output.iter().all(|value| value.is_finite()),
            "{name} produced a non-finite value"
        );
        black_box(&output);
    }
}

fn benchmark(name: &str, decoder: Decoder, workload: &Workload) -> BenchResult {
    let mut output = vec![0.0f32; ELEMENTS_PER_ROW];
    let _ = run_sample(
        decoder,
        workload,
        WARMUP_ITERATIONS,
        &mut output,
        false,
    );

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        samples.push(run_sample(
            decoder,
            workload,
            ITERATIONS_PER_SAMPLE,
            &mut output,
            false,
        ));
    }
    samples.sort_by_key(|sample| sample.elapsed);
    let median = samples[SAMPLES / 2];
    let min = samples.first().unwrap().elapsed;
    let max = samples.last().unwrap().elapsed;
    let allocations = run_sample(
        decoder,
        workload,
        ITERATIONS_PER_SAMPLE,
        &mut output,
        true,
    );

    let decoded_elements = (ROWS * ELEMENTS_PER_ROW * ITERATIONS_PER_SAMPLE) as f64;
    let input_bytes = (workload.bytes.len() * ITERATIONS_PER_SAMPLE) as f64;
    let output_bytes = decoded_elements * std::mem::size_of::<f32>() as f64;
    let seconds = median.elapsed.as_secs_f64();

    println!(
        "{:<15} {:>9} {:>6.3} [{:>6.3},{:>6.3}] {:>14.3} {:>14.3} {:>14.3} {:>5}/{:>5}/{:>5} {:>12.3}",
        name,
        workload.blocks_per_row,
        median.elapsed.as_secs_f64() * 1_000.0,
        min.as_secs_f64() * 1_000.0,
        max.as_secs_f64() * 1_000.0,
        decoded_elements / seconds / 1_000_000.0,
        input_bytes / seconds / (1024.0 * 1024.0),
        output_bytes / seconds / (1024.0 * 1024.0 * 1024.0),
        allocations.allocation_calls,
        allocations.reallocation_calls,
        allocations.deallocation_calls,
        allocations.allocated_bytes as f64 / 1024.0,
    );
    black_box(&output);

    BenchResult {
        median: median.elapsed,
        allocation_calls: allocations.allocation_calls,
        reallocation_calls: allocations.reallocation_calls,
    }
}

fn run_sample(
    decoder: Decoder,
    workload: &Workload,
    iterations: usize,
    output: &mut [f32],
    track_allocations: bool,
) -> Sample {
    reset_allocation_counters();
    let started = Instant::now();
    if track_allocations {
        TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    }
    for _ in 0..iterations {
        for row in workload.bytes.chunks_exact(workload.row_bytes) {
            decoder(black_box(row), ELEMENTS_PER_ROW, black_box(&mut *output)).unwrap();
            black_box(&*output);
        }
    }
    if track_allocations {
        TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    }
    let elapsed = started.elapsed();
    Sample {
        elapsed,
        allocation_calls: ALLOCATION_CALLS.load(Ordering::SeqCst),
        reallocation_calls: REALLOCATION_CALLS.load(Ordering::SeqCst),
        deallocation_calls: DEALLOCATION_CALLS.load(Ordering::SeqCst),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::SeqCst),
    }
}

fn reset_allocation_counters() {
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    ALLOCATION_CALLS.store(0, Ordering::SeqCst);
    REALLOCATION_CALLS.store(0, Ordering::SeqCst);
    DEALLOCATION_CALLS.store(0, Ordering::SeqCst);
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
}

fn print_relative_difference(name: &str, reference: Duration, direct: Duration) {
    let relative = (direct.as_secs_f64() / reference.as_secs_f64() - 1.0) * 100.0;
    println!(
        "{name} direct median time relative to reference: {relative:+.2}% (negative is faster)"
    );
}

fn dequantize_row_q8_0_reference(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
    if n_elements % QK8_0 != 0 {
        return Err(DataSourceError::General(format!(
            "Q8_0 row size {} not divisible by block size {}",
            n_elements, QK8_0
        )));
    }
    let n_blocks = n_elements / QK8_0;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q8_0 {
        return Err(DataSourceError::General(format!(
            "Q8_0 row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q8_0,
            bytes.len()
        )));
    }
    for (index, chunk) in bytes
        .chunks(BLOCK_SIZE_Q8_0)
        .enumerate()
        .take(n_blocks)
    {
        let block = BlockQ8_0::from_bytes(chunk)?;
        let mut decoded = [0.0f32; QK8_0];
        block.dequantize(&mut decoded);
        out[index * QK8_0..(index + 1) * QK8_0].copy_from_slice(&decoded);
    }
    Ok(())
}

const F16_SCALES: [u16; 4] = [0x3400, 0x3800, 0x3A00, 0x3C00];
const F16_MINS: [u16; 4] = [0x2C00, 0x3000, 0x3200, 0x3400];

fn pattern(seed: usize, index: usize, salt: usize) -> u8 {
    seed
        .wrapping_mul(37 + salt)
        .wrapping_add(index.wrapping_mul(29 + 2 * salt))
        .wrapping_add(17 * salt) as u8
}

fn make_q4_0_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q4_0);
    bytes.extend_from_slice(&F16_SCALES[seed % F16_SCALES.len()].to_le_bytes());
    for index in 0..16 {
        bytes.push(pattern(seed, index, 1));
    }
    bytes
}

fn make_q8_0_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q8_0);
    bytes.extend_from_slice(&F16_SCALES[seed % F16_SCALES.len()].to_le_bytes());
    for index in 0..32 {
        let value = ((seed * 17 + index * 23) % 255) as i16 - 127;
        bytes.push(value as i8 as u8);
    }
    bytes
}

fn make_q4_k_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q4_K);
    bytes.extend_from_slice(&F16_SCALES[seed % 4].to_le_bytes());
    bytes.extend_from_slice(&F16_MINS[(seed + 1) % 4].to_le_bytes());
    for index in 0..12 {
        bytes.push(pattern(seed, index, 2));
    }
    for index in 0..128 {
        bytes.push(pattern(seed, index, 3));
    }
    bytes
}

fn make_q5_k_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q5_K);
    bytes.extend_from_slice(&F16_SCALES[seed % 4].to_le_bytes());
    bytes.extend_from_slice(&F16_MINS[(seed + 1) % 4].to_le_bytes());
    for index in 0..12 {
        bytes.push(pattern(seed, index, 4));
    }
    for index in 0..32 {
        bytes.push(pattern(seed, index, 5));
    }
    for index in 0..128 {
        bytes.push(pattern(seed, index, 6));
    }
    bytes
}

fn make_q6_k_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q6_K);
    for index in 0..128 {
        bytes.push(pattern(seed, index, 7));
    }
    for index in 0..64 {
        bytes.push(pattern(seed, index, 8));
    }
    for index in 0..16 {
        let scale = ((seed * 11 + index * 7) % 31) as i8 - 15;
        bytes.push(scale as u8);
    }
    bytes.extend_from_slice(&F16_SCALES[seed % 4].to_le_bytes());
    bytes
}

fn make_q2_k_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q2_K);
    for index in 0..16 {
        bytes.push(pattern(seed, index, 9));
    }
    for index in 0..64 {
        bytes.push(pattern(seed, index, 10));
    }
    bytes.extend_from_slice(&F16_SCALES[seed % 4].to_le_bytes());
    bytes.extend_from_slice(&F16_MINS[(seed + 1) % 4].to_le_bytes());
    bytes
}

fn make_q3_k_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q3_K);
    for index in 0..32 {
        bytes.push(pattern(seed, index, 11));
    }
    for index in 0..64 {
        bytes.push(pattern(seed, index, 12));
    }
    for index in 0..12 {
        bytes.push(pattern(seed, index, 13));
    }
    bytes.extend_from_slice(&F16_SCALES[seed % 4].to_le_bytes());
    bytes
}

fn make_q8_k_block(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q8_K);
    let scale = 0.015625f32 * (1 + seed % 4) as f32;
    bytes.extend_from_slice(&scale.to_le_bytes());
    let mut quants = [0i8; QK_K];
    for (index, quant) in quants.iter_mut().enumerate() {
        *quant = (((seed * 19 + index * 13) % 127) as i16 - 63) as i8;
        bytes.push(*quant as u8);
    }
    for group in quants.chunks_exact(16) {
        let sum: i16 = group.iter().map(|value| *value as i16).sum();
        bytes.extend_from_slice(&sum.to_le_bytes());
    }
    bytes
}
