# Parallel Monte Carlo Pi Estimation

This project demonstrates parallel computation by estimating π using the Monte Carlo method, comparing performance across different numbers of threads.

## Prerequisites

1. **Python** (3.8 or newer)

2. **Rust** (latest stable)

## Running the Program

1. **Clone the repository**

   ```bash
   git clone https://github.com/nexxeln/parallel-pi
   cd parallel-pi
   ```

2. **Run the Rust program**
   ```bash
   cargo run --release
   ```
   This will generate `results.csv` with the benchmark data.

## Visualizing Results

1. **Create and activate Python virtual environment**

   ```bash
   python -m venv venv

   # On macOS/Linux
   source venv/bin/activate

   # On Windows
   .\venv\Scripts\activate
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the plotting script**
   ```bash
   python plot.py
   ```
