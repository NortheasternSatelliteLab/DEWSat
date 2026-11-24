# DEWSat on the Jetson Orin Nano: tables for the paper

Measured on 2026-11-21 (evening) and overnight into 2026-11-22, with the 7W benchmarks rerun on the morning of 2026-11-22. Two models ran on the same board:

- **v2, the paper's model:** the plain Bi-LSTM, ten-seed ensemble (seeds 42-51, 11,617 parameters each), job 1.
  - `fused10` is the ten seeds in one graph. This is the version to quote.
  - `seed42` is one copy. Its speed and power are measured; its accuracy is not reported.
  - `ensemble10` is ten separate copies, used for the INT8 tests.
- **Prithvi, the Prithvi pilot's fresh-test model:** the frozen Prithvi-EO-2.0-tiny encoder (5.67 M parameters) and O3's GRU head (5 seeds x 67,588 parameters). Its ESI output passed gate 1 of the fresh test (scored once on 2026-11-20, D90). The kit also times D91's Bi-LSTM head, which is exploratory and has no verdict.

**Board:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, 8 GB.
- Software: JetPack 6.2.1 (L4T 36.4.4), TensorRT 10.3.0, CUDA 12.6, ONNX Runtime 1.23.2, Python 3.10.
- Setup: the developer-kit carrier board in room air with the stock fan.

## Files

| File | What it holds |
|---|---|
| `B0_budget_sheet.csv` | **Start here.** One line per budget item: power, energy, time, heat, memory, storage, data and reliability, each with its conditions and source table |
| `J0_v2_power_budget_by_mode.csv` | One row per power mode and clock setting: idle power (at the benchmark's start, how long after boot, at rest, and the one to use for the budget), the paper model's latency and energy per window, region and tile, one copy vs ten |
| `J1_v2_speed_power_energy.csv` | The paper model and one copy, per window, region (25 windows) and tile (100 windows), every mode |
| `J2_v2_heat_soaks.csv` | The long sustained-load tests: hottest sensor, margin to 70 C, power, errors, latency drift, over-current events, and why a test is not PASS |
| `J3_v2_board_parity.csv` | Whether the board gives the paper's answers: AUC, Bi-LSTM minus LR, Brier, ECE and probability differences for every runtime and engine |
| `J4_boot_time.csv` | Each reboot: command to SSH login (measured on the Mac) and systemd's boot time |
| `J5_v2_all_benchmark_rows.csv` | Every v2 benchmark row: all runtimes, engines, batches, precisions and modes |
| `J6_v2_tensorrt_engines.csv` | The TensorRT engines built on the board: size, build time, and the precision TensorRT actually used per layer |
| `J7_paper_claims_check.csv` | The manuscript's power, heat and speed statements next to what the board measured |
| `J8_v2_soak_timeseries.csv` | The soaks in 10-second steps: power, every temperature sensor, fan, GPU clock, latency, memory. This is the source data for the heat figure |
| `P1_prithvi_speed_power_energy.csv` | Prithvi model, every mode run: TensorRT (encoder per chip and per 100-chip region pass, GRU head per window and per region-month) and the CPU (ONNX Runtime). **Each row says whether its output matched the reference on the board. Quote only rows that say yes.** |
| `P2_prithvi_board_parity.csv` | Prithvi model on the board against the stored fresh-test predictions (no labels on the board) |
| `P3_prithvi_all_benchmark_rows.csv` | Every Prithvi benchmark row, with engine size, memory and the same match column |
| `P4_prithvi_per_cell_and_region.csv` | Prithvi model per cell and per region (encoder plus head, one pass a month; each extra pass adds the encoder cost). Two paths: **the CPU, whose outputs match**, and TensorRT, whose outputs did not (timing only) |
| `figures/jetson_heat_power_soaks.png/.pdf` | Hottest sensor and module power over time in the three soak tests F10, F11 and F12. Drawn from J8's F10, F11 and F12 rows (J8 also holds the Nov 21 soak, which the figure leaves out) |
| `figures/jetson_power_by_mode.png/.pdf` | Idle power vs power while running the paper model, every power mode. Drawn from J0 |
| `figures/jetson_heat_power_soaks_data.csv`, `figures/jetson_power_by_mode_data.csv` | The exact values each figure plots, copied from J8 and J0 without re-rounding, for re-making the charts in Excel (see the section below). `make_jetson_figures.py` writes them with the figures |
| `jetson_tables.html` | Every table on one page: select one, copy it, paste it into Google Docs or Word |
| `CROSSCHECK.txt` | Independent checks: every trtexec number re-read from its raw log, the soak summaries recomputed from their 10-second records |
| `make_jetson_tables.py`, `make_jetson_figures.py`, `crosscheck_jetson.py` | The scripts that rebuild everything (commands at the end) |

## How the numbers were measured

- **Power is module input power (VDD_IN).** It comes from the Orin Nano module's own INA3221 sensor, sampled about 10 times a second and averaged over each timed run. For trtexec rows only the middle 60% of the run is averaged.
  - It is the module only. It leaves out the carrier board, NVMe SSD, fan, USB, any radio or camera, and power-supply losses.
- **Idle** is module power with nothing running, averaged over 10 s. There are two kinds of reading:
  - **At the start of each benchmark.** Readings taken about a minute after a reboot came out about 0.5 W high. At 15W with dynamic clocks the reading was 4.99 W, against 4.44-4.50 W at rest. At MAXN_SUPER with pinned clocks it was 7.14 W, against 6.67 W.
  - **At rest**, with no load for the 4-10 min before. These are each soak's baseline (taken after a 10-min cool-down), F11's 240-s rest phases, and the benchmarks that began after a logged 5-min rest: the 7W ones and the Prithvi 15W and 7W ones.
  - J0's **Idle for the budget** column uses the rested reading where one exists and says which reading it used. The budget sheet and the claims check use the same value.
  - **Energy above idle** in J1, J5, P1 and P3 subtracts the idle read at the start of the same benchmark. J0 also gives each tile's energy above the rested idle.
  - The stock Ubuntu desktop's login screen (GDM) was running. A headless flight image would idle lower; that was not measured.
- **Over-current events** are the module's own over-current protection counters (`/sys/class/hwmon/hwmon*/oc*_event_cnt`), read before and after each run. The protection throttles the clocks when the input current passes its limit.
- **Energy per unit** = module power ÷ throughput × units per batch.
- **Energy above idle** = (module power − idle power) ÷ throughput: the energy the inference itself adds.
- **Clocks:**
  - **dynamic (DVFS)** is the board's default: clocks rise with load. This is the flight-like setting.
  - **pinned** uses `jetson_clocks` to hold the CPU, GPU and memory clocks at the mode's maximum. This gives the best-case latency and the worst-case idle power.
- **Power modes** are NVIDIA's nvpmodel presets: 7W, 15W, 25W and MAXN_SUPER (the Super kit's maximum).
  - 7W on this board, read from `nvpmodel -q --verbose`: CPU cores 0-3 online (4-5 off) at 729.6-960 MHz, GPU up to 408 MHz with part of the GPU power-gated, memory clock capped at 2,133 MHz.
  - Switching to 7W or to 25W made nvpmodel reboot the board; 15W and MAXN_SUPER did not.
- **Timing:**
  - **trtexec** is NVIDIA's own tool. It measures synchronous inference with the host-to-GPU and GPU-to-host copies included. The tables quote these.
  - **From Python** is the same engine called from a Python loop, as a Python flight program would call it.
  - **CUDA graph** rows (in J5 and P3) remove the kernel-launch overhead.
- **Units:**
  - **v2:** a window is one cell's 12 monthly steps; a region is 25 windows (5 x 5 cells); a tile is 100 windows, about one 110 km HLS tile (the kit's definition).
  - **Prithvi:** a chip is a 6-band 224 x 224 HLS crop; a cell-pass is 4 chips; a region pass is 100 chips; a region-month is 25 windows. The encoder runs once per satellite pass and the GRU head once per cell-month.
- **Precision TensorRT actually ran** is read from TensorRT's own layer report.
  - Engines built with FP16 allowed did not always use it. Only the fused ten-copy engine at batches above 1 ran in FP16.
  - The batch-1 engines and the one-copy engines ran in FP32. They time almost the same as the FP32 builds.
- **Heat** is the hottest of the CPU, GPU, SoC and junction sensors.
  - Room temperature was not recorded.
  - The 70 C line is the board's own passive (throttling) trip on those sensors. The critical shutdown is at 104.5 C.

## What these tests do not show (say this in the paper)

1. **Module power only.** The carrier board, SSD, fan, radio, camera and power conversion are not included.
2. **Room air with a fan, not vacuum.** In orbit the heat must leave by conduction. These temperatures do not show that a flight unit stays below 70 C. No thermal-vacuum or radiation test was done.
3. **One board, one night.** There are no repeats across boards or days.
4. **The image preprocessing that makes the windows and chips is not timed.**
5. **The manuscript's 10-25 W and 1-2U figures were written for the 12U design.** The mission is now 6U, so the budget needs the 6U power system's numbers, which were not measured here.
6. **Accuracy was not re-measured.**
   - The board's outputs were checked against the result the Mac scored once (v2: 2026-11-09; Prithvi: 2026-11-20).
   - Nothing was rescored or chosen on the board.
   - The Prithvi kit carries no labels.
   - A single seed's accuracy was never computed.
7. **ONNX Runtime (CPU) could not run in the 7W mode.**
   - Version 1.23.2 aborts as soon as it is imported (exit 134, an index assertion in its CPU detection), because 7W takes CPU cores 4-5 offline.
   - The abort also ended the overnight 7W benchmarks before their TensorRT rows, so the 7W benchmarks were rerun with ONNX Runtime left out.
   - The 7W rows are TensorRT only. The flight path is TensorRT in every mode.
8. **Two runs hit the module's over-current limit.**
   - At MAXN_SUPER with pinned clocks and continuous 4,096-window batches (F12), the module's over-current protection fired 18 times in 60 min. There were no errors, and latency did not change (p50 drift 0.05%). That is the only reason F12 is marked WARN.
   - In the Prithvi benchmark at MAXN_SUPER it fired 13,900 times. The encoder on TensorRT drew about 22 W there.
   - It fired in no other soak or benchmark, and never at 15W or 7W.
9. **TensorRT did not reproduce the Prithvi model on this board, so its TensorRT speed and energy are timing only.**
   - **Encoder:** all four engines (FP32 and FP16, batch 1 and variable batch) gave wrong features. The worst chip's cosine similarity to the Mac's PyTorch features was 0.67, and the mean error (0.46) was larger than the mean feature (0.29). The error was the same in FP32 and FP16 and on every rerun, so it is not rounding. trtexec, run independently, gave the same wrong answer.
   - **GRU head:** the variable-batch engines gave wrong outputs (ESI rank agreement 0.69), and two runs of the same engine differed. The batch-1 engines matched to 4e-7.
   - **Bi-LSTM head (exploratory):** every engine matched.
   - **CPU (ONNX Runtime):** the same ONNX files matched on the board's CPU. The encoder matched to 8.8e-6 with cosine 1.0, and the head to 5e-7.
   - **What can be quoted:** for Prithvi, only the CPU rows and the GRU head's batch-1 TensorRT rows.
   - **The cause is not known.** The build logs show no warning. Finding it needs diagnostic builds on the board, which were not run.

## Suggested figure captions

The limits go in the captions, not on the figures.

- **`jetson_heat_power_soaks`:** Hottest on-chip sensor (top) and module input power (bottom) during three sustained-load tests of the paper model on the Jetson Orin Nano Super:
  - F10: the 15 W mode with dynamic clocks, 100-window batches back to back for 120 min.
  - F11: the same load, 60 s on and 240 s off, for 60 min.
  - F12: the worst case, MAXN_SUPER with pinned clocks and 4,096-window batches, for 60 min.

  The dashed line is the board's 70 °C passive (throttling) trip. Room air with the stock fan, on the developer-kit carrier board. Power is the module's input (VDD_IN) only, without the carrier board, SSD or fan. This is not a thermal-vacuum test.
- **`jetson_power_by_mode`:** Module input power at idle (grey) and while running the paper model on 100-window tiles back to back (blue), for each power mode and clock setting, with the median time per tile (TensorRT FP16, host-device copies included).
  - Idle is read with no load for the 4-10 min before, except where starred.
  - Module power only.

## Re-making the figures in Excel

- **`figures/jetson_heat_power_soaks_data.csv`:** one block of columns per test, side by side.
  - F10 runs 120 min (719 rows); F11 and F12 run 60 min (360 rows), so their columns are blank below that.
  - Chart: Insert > X Y (Scatter) > Scatter with Straight Lines. For each series, X is a test's minutes column and Y is its hottest-sensor column (top panel) or power column (bottom panel).
  - The 70 °C line is a two-point series: (0, 70) and (120, 70).
  - The figure's labels are the column maximum (hottest) and the column mean (power). For F11, average the `on` and `off` rows separately (7.745 W and 4.50 W; J2 rounds the first to 7.74).
- **`figures/jetson_power_by_mode_data.csv`:** one row per mode and clock setting, top to bottom as in the figure.
  - Chart: a clustered bar chart, or a scatter of idle and running power against the row.
  - The figure stars an idle value when the third column does not start with "at rest".

## Rebuild

```bash
python3 make_jetson_tables.py
~/Desktop/DEWSat_BiLSTM_v2/.venv/bin/python make_jetson_figures.py
python3 crosscheck_jetson.py
```

**Sources:** the files the board wrote, copied back with hash checks.
- v2: `~/Desktop/DEWSat_BiLSTM_v2/jetson_kit/results_from_jetson/` (2026-11-21, 2026-11-22, and 2026-11-22_7W for the 7W rerun; each fetch repeats the earlier files, and the scripts use each file once)
- Prithvi: `~/Desktop/DEWSat_Prithvi_pilot/jetson_ft/results_from_jetson/`
