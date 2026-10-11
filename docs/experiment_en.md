# Experiment notebook — Soil moisture forecasting and GPU acceleration

French editorial source: [experiment_fr.md](experiment_fr.md).

This notebook reconstructs the experiment in its logical order, from the
agronomic data through the GPU optimizations. It therefore does not merely
describe the final system: it explains why each experiment was conducted, what
it showed, and which decision followed from it.

The values presented here come first from the frozen machine-readable results
in `results/`, then from the final benchmark files and current documentation.
Older results remain useful for tracing the history, but they do not replace
final measurements when an experiment was repeated with a more complete
protocol.

## 1. Objective and research question

Precision irrigation seeks to deliver water in the right place at the right
time. A soil-moisture forecast can help anticipate drying, avoid an unnecessary
application, or plan water usage more effectively. In this project, the
predictive objective is to estimate future moisture at several horizons, from
1 to 24 hours, using the previous 24 hours of measurements.

The project also started from an interest in **DeepONet**, a neural-network
architecture designed to learn operators: instead of simply mapping one vector
to one output, it represents a historical function and queries it at a point,
here the forecast horizon. There was, however, no prior assumption that this
architecture would automatically be more accurate than a simpler network.

Two distinct questions therefore guided the work:

1. **Predictive quality:** does DeepONet improve soil-moisture forecasting over
   a simple rule and a multilayer perceptron?
2. **Computational performance:** under what conditions does GPU inference
   become useful for this DeepONet?

This distinction is essential. A model can be interesting for studying an
operator architecture and its GPU deployment without being the experiment's
best predictor.

## 2. Arnesano dataset

### 2.1 Selected data

The project uses version 2 of the *Soil Moisture, Irrigation Actuator and
Weather Dataset from a Multi-Sector Precision-Irrigation Field Trial*,
published on Mendeley Data under
[DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2). It describes
an instrumented farm in Arnesano, Apulia, Italy, with soil measurements,
irrigation telemetry, and weather data.

Sector 4 was selected. Under the same conservative cleaning policy, it provided
the largest number of valid origins for every horizon among sectors 1, 2, and
4. Sector 3 provided virtually no complete 24-hour historical windows because
of a long flat signal tail; sector 5 was excluded because of a documented
inconsistency in water volumes.

Each point on the grid contains five variables, in this order:

1. soil moisture;
2. applied water volume;
3. rain;
4. air temperature;
5. air humidity.

The grid is regular, with a 10-minute interval. A 24-hour history therefore
contains 144 observations and forms a `[144, 5]` tensor. The studied horizons
are 1, 3, 6, 12, and 24 hours. The **target**, sometimes called the *ground
truth*, is the moisture measurement actually recorded at the relevant horizon;
it is an experimental reference supplied by the sensor, not a perfect measure
of all water physically present in the soil.

### 2.2 Cleaning and prevention of temporal leakage

The author-provided “merged” table was preferred over the already preprocessed
table because it preserves the complete 10-minute grid. The main decisions were:

- check the physical moisture range, clip at 100, and remove obvious spikes;
- interpolate only when a complete interior gap is no longer than 20 minutes;
- reject every window that touches a longer gap;
- remove series tails corresponding to a stalled sensor;
- lag applied water by one step, so the input at time `t` contains only a fully
  completed irrigation interval;
- include no future weather or irrigation in the inputs.

The train, validation, and test splits are chronological and contiguous, using
70/15/15 time proportions. Histories or targets that cross a boundary are
purged. This produces 24,156 training examples, 4,235 validation examples, and
5,097 test examples. Every normalization statistic is fitted only on rows used
by training histories.

The distinction is as follows: **training** adjusts the weights;
**validation** selects model variants; **test** is consulted only after the
configuration is frozen. This chronology prevents information from the future
from indirectly influencing modeling decisions.

### 2.3 A difficult series despite its apparent stability

The series contains many plateaus. In the test set, the target is exactly
unchanged from the prediction origin in 94.75% of cases at 1 hour, 87.49% at
3 hours, 82.06% at 6 hours, 75.52% at 12 hours, and 66.45% at 24 hours. A
forecast that never changes is therefore already highly competitive at short
horizons.

At the same time, there is a strong chronological shift: the target mean is
78.15 in training, 60.51 in validation, and 55.68 in test. The experiment thus
combines two difficulties: many zero changes, but an absolute moisture
distribution that moves substantially over time.

## 3. Baseline: persistence

**Persistence** is the simplest forecasting rule: it assumes that future
moisture will equal current moisture.

```text
ŷ(t + h) = y(t)
```

This baseline is scientifically important. If the soil changes slowly, a neural
network must contribute more than a small random correction to beat it. It also
prevents a complex model that performs worse than “do nothing” from being
presented as useful.

Three metrics are used:

- **MAE** is the mean absolute error;
- **RMSE** is the root mean squared error and penalizes large errors more
  strongly;
- **R²** equals 1 for a perfect prediction, 0 for a mean-based reference, and
  can become negative when the model is worse than that mean reference.

| Split      |    MAE |   RMSE |      R² |
| ---------- | -----: | -----: | ------: |
| Validation | 0.2916 | 1.2174 |  0.2472 |
| Test       | 0.2743 | 1.0489 | -0.1073 |

Persistence's low MAE is consistent with the plateaus. On the 4,164 test
samples considered “nearly unchanged,” its error is exactly zero. By contrast,
its RMSE reaches 2.4516 on the 933 samples that genuinely change. Overall RMSE
is therefore a better guide than MAE alone for evaluating rare but important
transitions.

## 4. First MLP experiments

An **MLP** (*multilayer perceptron*) is a network made of linear layers and
nonlinear functions. The first model flattened the 144 × 5 historical values,
appended the normalized horizon, and applied this architecture:

```text
721 → 128 → ReLU → 64 → ReLU → 1
```

`ReLU` keeps positive values and replaces negative values with zero; this
nonlinearity allows the network to represent more than a simple linear
regression. This first MLP predicted future absolute moisture directly.

| Direct model |    MAE |   RMSE |      R² |
| ------------ | -----: | -----: | ------: |
| Validation   | 0.6994 | 1.1110 |  0.3731 |
| Test         | 1.7122 | 1.9633 | -2.8795 |

Validation looked promising in terms of RMSE, but test performance collapsed.
An audit of the pipeline found no denormalization error, incorrect target index,
or future-feature leakage. Instead, the diagnosis highlighted the temporal
shift in absolute values. Mean bias, defined as `prediction - target`, moved
from -0.2333 in validation to -1.6207 in test and reached -2.0143 at the 24-hour
horizon.

**Observation:** the model learned an absolute value strongly tied to the
moisture regime of the training period.

**Hypothesis:** asking it to predict only the future change should be more stable
when the absolute level moves.

## 5. Residual prediction and final MLP

### 5.1 Changing the learned question

The residual target is the change between the origin and the horizon:

```text
Δy = y(t + h) - y(t)
ŷ(t + h) = y(t) + Δŷ
```

With this formulation, predicting `Δŷ = 0` is exactly persistence. The network
therefore learns the more focused question “by how much will moisture change?”
rather than “what will its absolute value be?”

An initial residual normalization subtracted the training mean. A zero network
output then corresponded to a physical change of -0.336 rather than persistence.
The final formulation divides only by the training residual standard deviation,
4.435837, without subtracting the mean:

```text
network target = Δy / 4.435837
```

The last layer is initialized to zero. At the start of training, the model
therefore reproduces persistence exactly. This small decision provides a
physically consistent prior without changing the inputs.

### 5.2 Successive experiments

The first mean-centered residual MLP reached a validation RMSE of 1.1755.
Switching to persistence-centered scaling with MSE—the mean squared error—
lowered it to 0.9460. A slightly wider `256 → 128 → 64` architecture reached
0.9347.

Two additional hypotheses were tested and rejected:

- Huber loss, which is more robust to large deviations, yielded 1.1796;
- moderate weighting of changes during training yielded 1.0449.

These results indicated that large training changes should not receive still
more emphasis. Validation and test remained natural: their temporal
distributions were never modified.

The final pass evaluated simple temporal features calculated exclusively at or
before the prediction origin. Three moisture trends were retained: current
moisture minus moisture 1, 3, and 6 hours earlier. They improved validation RMSE
from 0.9347 to 0.8975, approximately 4%. Recent irrigation and rain sums, alone
or combined, did not provide a comparable gain.

### 5.3 Frozen MLP

The final MLP receives 720 historical values, one horizon, and three trends:

```text
724 → 256 → 128 → 64 → 1
```

It uses ReLU, MSE, Adam, a `1e-3` learning rate, batches of 512, and seed 42. A
seed fixes pseudo-random initialization and makes it possible to test whether a
result depends on a lucky draw. Selection used validation only. Three seeds
gave a mean validation RMSE of 0.9244 ± 0.0193; seed 42, already selected by
validation, reaches 0.8975.

On the final test, this MLP reaches an RMSE of 0.9478 and an R² of 0.0958. An
earlier MLP without trends happened to score 0.9310 on test, but only 0.9347 on
validation. It did not replace the official model: choosing after reading the
test would have constituted methodological leakage.

## 6. DeepONet experiment

### 6.1 Principle

DeepONet is an architecture proposed for learning mappings between functions,
known as operators. In our case, the 24-hour history plays the role of the input
function and the horizon is the point at which that function is queried. The
formulation follows the original DeepONet principle described by
[Lu et al.](https://doi.org/10.1038/s42256-021-00302-5).

```text
24 h history + trends
          ↓
    Branch network
          ↓
   64-element vector
          \
           dot product → predicted change → future moisture
          /
   64-element vector
          ↑
     Trunk network
          ↑
     horizon / 24
```

The **Branch** network encodes the historical state. The **Trunk** network
encodes the query coordinate, here the horizon. Their elementwise product is
summed to produce a single normalized residual. This structure, rather than a
concatenation followed by another large network, defines the tested DeepONet.

### 6.2 Bounded search and selected model

The initial raw-history model obtained a validation RMSE of 1.1143. Reducing
the learning rate or Trunk size did not improve it. A single ablation adding
the MLP's three historical trends reduced RMSE to 1.0292; this approximately
7.6% gain justified retaining them.

The frozen model contains:

- Branch: `723 → 256 → 128 → 64`;
- Trunk: `1 → 64 → 64 → 64`;
- latent dimension: 64;
- dot product and operator bias;
- residual reconstruction with standard deviation 4.435837;
- 234,945 trainable parameters.

Six DeepONet training runs were performed in total. The final configuration was
selected on validation with seed 42. Its RMSE across three seeds is
1.1202 ± 0.1025, which reveals non-negligible sensitivity to initialization.

### 6.3 Scientific result

DeepONet reaches an RMSE of 1.0292 in validation and 1.2112 on test. The frozen
MLP reaches 0.8975 and 0.9478, respectively. The important result is therefore
unambiguous: **the final MLP predicts better than the tested DeepONet, overall
and at every one of the five test horizons**.

DeepONet was nevertheless kept as the deployment target. It is the operator
model at the center of the technological question and makes it possible to
study the costs of a Branch/Trunk structure separately on CPU and GPU. Keeping
this model is not a claim that it is more accurate.

## 7. Prediction comparison

| Model                    | Validation MAE | Validation RMSE | Validation R² | Test MAE |  Test RMSE |    Test R² |
| ------------------------ | -------------: | --------------: | ------------: | -------: | ---------: | ---------: |
| Persistence              |         0.2916 |          1.2174 |        0.2472 |   0.2743 |     1.0489 |    -0.1073 |
| Frozen residual MLP      |         0.3598 |      **0.8975** |    **0.5909** |   0.5678 | **0.9478** | **0.0958** |
| Frozen residual DeepONet |         0.5304 |          1.0292 |        0.4620 |   0.8390 |     1.2112 |    -0.4767 |

Persistence retains the best MAE because it is perfect on plateaus. RMSE tells
a different story: the MLP reduces large errors during changes more effectively.
DeepONet remains better than persistence on test at 1 and 3 hours, but becomes
slightly worse at 6 hours and substantially worse at 12 and 24 hours.

| Test horizon | Persistence RMSE |   MLP RMSE | DeepONet RMSE |
| -----------: | ---------------: | ---------: | ------------: |
|          1 h |           0.9557 | **0.7717** |        0.8377 |
|          3 h |           1.0007 | **0.8337** |        0.8877 |
|          6 h |           1.0427 | **0.9028** |        1.0488 |
|         12 h |           1.0920 | **1.0023** |        1.4396 |
|         24 h |       **1.1577** |     1.2052 |        1.6930 |

The following figure shows the 24-hour horizon. Ground truth is the future
measurement; persistence carries forward the current measurement; both
networks add their predicted change. It makes the long flat areas and rare
transitions that dominate RMSE visible.

![Final predictions at the 24-hour horizon](../results/deeponet_final_test_predictions_24h.png)

## 8. Freezing the deployment experiment

After validation selection and the single final evaluation, the
`models/deeponet_reference.pt` checkpoint was frozen. A `.pt` checkpoint
contains the PyTorch weights and metadata needed to reconstruct the model
exactly. Freezing prevents the deployment phase from drifting into another
phase of predictive tuning.

The model was then exported to **ONNX**, a portable format that describes a
computation graph independently of PyTorch. The graph receives:

- `branch_input [batch, 723]`;
- `horizon [batch, 1]`;
- `current_moisture [batch, 1]`;

and directly returns `prediction [batch, 1]` in physical units. The
Branch/Trunk product, residual rescaling, and addition of current moisture are
included in ONNX.

Ten deterministic cases—two for each of the five horizons—serve as **golden
vectors**. They fix the inputs and expected outputs for comparing inference
engines. The maximum PyTorch/ONNX Runtime difference is `7.6294 × 10⁻⁶`, below
the FP32 tolerance of `10⁻⁴`.

The following command reconstructs and verifies these artifacts:

```bash
uv sync --project python
uv run --project python setup-onnx
```

`setup-onnx` **never trains the model**. It starts from the frozen checkpoint,
checks the contract, re-exports ONNX, verifies numerical equivalence, and
refreshes the Python and C++ golden vectors. Because the ONNX model itself is
versioned, C++ users need neither Python, PyTorch, nor the dataset to run
inference.

## 9. C++ and ONNX Runtime

The deployment chain is deliberately short:

```text
Frozen DeepONet
      ↓
     ONNX
      ↓
ONNX Runtime C++
   ↙       ↘
 CPU       CUDA
```

**Inference** is the use of an already trained network to produce a prediction
without modifying its weights. **ONNX Runtime** is an engine that loads the ONNX
graph, optimizes this inference, and delegates its operations to an execution
provider. `CPUExecutionProvider` runs on the processor.
`CUDAExecutionProvider` places compatible operations on an NVIDIA GPU through
CUDA, NVIDIA's parallel-computing platform.

The C++ project uses ONNX Runtime 1.30.0, obtained as an official CUDA 13 binary
archive by CMake. At load time it validates the names, types, and shapes of all
four tensors. Batches are dynamic: the `batch` dimension can change without
modifying the model.

The same golden vectors are run on CPU and CUDA. The maximum difference between
the C++ FP32 runtime and the expected values is `3.8147 × 10⁻⁶`. This step checks
numerical correctness before any speed measurement; a fast engine that differs
from the frozen model would be experimentally useless.

## 10. First CPU versus CUDA benchmarks

### 10.1 Protocol

A **batch** is the number of samples processed by one model call. A batch of 1
represents an isolated request. A batch of 4096 processes 4096 forecast samples
together and can fill the GPU's parallel compute units more effectively.

**Latency** is the time needed to complete one call; lower is better.
**Throughput** is the number of samples processed per second; it often increases
with batch size even when total batch latency rises. These criteria correspond
to different uses: interactive response for latency, bulk processing for
throughput.

Final measurements were made on an NVIDIA GeForce RTX 2060 and an Intel Core
i5-9600K, with CUDA 13.4, ONNX Runtime 1.30.0, and a Release build. Every
combination uses 100 warmup iterations followed by 1000 measured iterations,
repeated three times. The ten real golden inputs are repeated cyclically to form
large batches. Model loading and session creation are outside timing.

The normal CUDA path starts with data in CPU memory and returns a CPU-visible
result; it therefore includes transfers managed by ONNX Runtime.

| Batch |    CPU, ms |   CUDA, ms | CPU throughput, samples/s | CUDA throughput, samples/s | CUDA/CPU speedup |
| ----: | ---------: | ---------: | ------------------------: | -------------------------: | ---------------: |
|     1 | **0.0181** |     0.1201 |                **55,175** |                      8,326 |           0.151× |
|     8 | **0.0259** |     0.1166 |               **309,043** |                     68,625 |           0.222× |
|    32 | **0.0586** |     0.1278 |               **546,108** |                    250,455 |           0.459× |
|   128 |     0.1552 | **0.1449** |                   824,815 |                **883,223** |           1.071× |
|   256 |     0.2689 | **0.2048** |                   952,148 |              **1,249,784** |           1.313× |
|   512 |     0.4859 | **0.3012** |                 1,053,681 |              **1,700,110** |           1.613× |
|  1024 |     0.9388 | **0.4848** |                 1,090,780 |              **2,112,079** |           1.936× |
|  2048 |     1.9312 | **0.8553** |                 1,060,499 |              **2,394,545** |           2.258× |
|  4096 |     4.1299 | **1.6125** |                   991,782 |              **2,540,168** |       **2.561×** |

The GPU is therefore not “always faster.” CPU clearly wins at batches 1, 8,
and 32. The observed crossover lies between 32 and 128; the tested points do
not support assigning it a more precise value. At 4096, the large matrix
multiplications use enough of the GPU to reach 2.56 times the CPU speed in
batch latency.

## 11. I/O Binding experiment

The CPU's main memory, often called **host** memory, is separate from the GPU's
memory, or **device memory/VRAM**. An input must generally be copied from host to
GPU (**H2D**), and the result from GPU to host (**D2H**).

ONNX Runtime **I/O Binding** makes it possible to bind buffers explicitly to a
session's inputs and outputs. This can avoid some implicit allocations or
copies, but it does not magically remove transfers required by the application's
contract.

In the first experiment, inputs still originated on CPU and the output had to
return there. H2D and D2H copies therefore remained included. This path reaches
0.1221 ms versus 0.1201 ms for normal CUDA at batch 1, 0.1749 versus 0.1449 ms
at batch 128, and 1.6433 versus 1.6125 ms at batch 4096. It is slower at every
one of the nine tested batches.

**Conclusion:** host-to-host I/O Binding did not provide a stable benefit. It
must not be described as a generic acceleration switch.

## 12. Profiling with Nsight Systems

The benchmark answers “how long?”, but not “why?”. A **profiler** observes CPU
calls, copies, and GPU kernels to locate costs. Nsight Systems, NVIDIA's
[system profiler](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
was used for short traces at batches 1 and 4096.

A CUDA **kernel** is a function launched on the GPU. Launching it carries a
fixed CPU/GPU cost even if its internal computation is tiny. Normal CUDA
DeepONet triggers approximately 21 kernels per inference. At small batch sizes,
many of these kernels are very short: their orchestration weighs more heavily
than useful computation. At batch 4096, the large **GEMMs**—general matrix
multiplications that implement the linear layers—account for more of the GPU
time. H2D transfers also become visible in total cost.

Nsight timings are not used as benchmark results. Trace collection perturbs
execution; here it serves only as diagnostic evidence for choosing the next
experiments.

## 13. Optimization 1 — CUDA Graphs

### 13.1 Hypothesis

Because small workloads are dominated by repeated launches, grouping an
already-known sequence should reduce orchestration. A **CUDA Graph** captures a
stable sequence of CUDA operations and replays it with a single graph launch.
ONNX Runtime's constraints require stable shapes and addresses, so one session
is created for each tested batch. Capture occurs before the measured loop, in
accordance with CUDA Graph support in the
[CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview).

### 13.2 Results

| Batch | Normal CUDA, ms | CUDA Graph, ms | Speedup over normal CUDA |
| ----: | --------------: | -------------: | -----------------------: |
|     1 |          0.1201 |     **0.0535** |               **2.244×** |
|     8 |          0.1166 |     **0.0607** |               **1.921×** |
|    32 |          0.1278 |     **0.0861** |               **1.483×** |
|   128 |          0.1449 |     **0.1373** |                   1.056× |
|   256 |      **0.2048** |         0.2083 |                   0.983× |
|   512 |          0.3012 |     **0.3006** |                   1.002× |
|  4096 |          1.6125 |     **1.5939** |                   1.012× |

The result matches the hypothesis: the gain is strong when launch cost
dominates, then becomes neutral when large GEMMs dominate. The trace contains
one `cudaGraphLaunch` per replay instead of repeated individual launches. The
copies remain, which explains why graph replay does not solve the transfer cost
of large batches.

At batch 1, CUDA Graph improves CUDA by 2.24×, but remains at 0.0535 ms versus
0.0181 ms for CPU. Reducing GPU overhead is therefore not sufficient to
guarantee the best absolute latency for an isolated request.

## 14. Optimization 2 — GPU-resident I/O Binding

This experiment deliberately changes the measured contract.

```text
Host-to-host contract: CPU → GPU → inference → CPU

GPU-resident contract: data already on GPU
                              ↓
                          inference
                              ↓
                       result stays on GPU
```

Buffers are allocated and inputs transferred before timing. During measured
iterations there is no H2D copy, D2H copy, or allocation. A single final copy
is used for correctness. Nsight traces confirm that observed copies belong to
setup and final readback, not the measured iterations.

At batch 4096, this path reaches **0.5734 ms**, or **7.14 million samples per
second**. It is **2.812× faster** than normal CUDA at 1.6125 ms. The gain grows
with the batch: 1.19× at batch 1, 1.31× at 128, 1.95× at 512, and 2.81× at
4096.

This result does not represent the latency of an ordinary application that
receives and returns CPU data. It measures the model's potential when integrated
into a larger pipeline in which the preceding stage already produces GPU inputs
and the following stage consumes the output on GPU.

## 15. Optimization 3 — FP16 and Tensor Cores

The canonical model uses **FP32** numbers encoded on 32 bits. Experimental
**FP16** uses 16 bits for internal weights and operations while keeping FP32
inputs and output. The representation is less precise, but moves less data and
can activate **Tensor Cores**, specialized units for reduced-precision matrix
multiplication.

The canonical FP32 model was not modified and no retraining took place. A
second ONNX file is produced by deterministic conversion.

| Batch | CUDA FP32, ms | CUDA FP16, ms | FP16 speedup |
| ----: | ------------: | ------------: | -----------: |
|     1 |    **0.1201** |        0.1269 |       0.947× |
|     8 |    **0.1166** |        0.1336 |       0.872× |
|    32 |    **0.1278** |        0.1411 |       0.905× |
|   128 |    **0.1449** |        0.1502 |       0.965× |
|   256 |        0.2048 |    **0.1786** |       1.147× |
|   512 |        0.3012 |    **0.2762** |       1.090× |
|  1024 |        0.4848 |    **0.4372** |       1.109× |
|  2048 |        0.8553 |    **0.7762** |       1.102× |
|  4096 |        1.6125 |    **1.4400** |   **1.120×** |

FP16 is therefore harmful at small batches, where conversions and fixed costs
are not amortized. From 256 onward, the measured gain is 9–15%. At batch 4096,
throughput rises from 2.54 to 2.84 million samples per second.

The maximum absolute difference from the FP32 reference is 0.013729 moisture
unit; the mean difference at batch 4096 is 0.007320. These values come from the
repeated golden cases and do not constitute a new predictive evaluation over
the complete test set. Nsight identifies `tensorop_f16_s1688` kernels, evidence
that Tensor-Core-eligible GEMMs were selected. FP16 therefore remains an
experimental option, useful at large batch when this numerical difference is
acceptable, rather than a new default model.

## 16. Why no custom CUDA kernel?

After profiling, custom CUDA fusion was considered. The FP32 ONNX graph contains
six `Gemm`, four `Relu`, two `Mul`, two `Add`, one `ReduceSum`, and one constant.
Biases are already included in the `Gemm` operations, and the important matrix
multiplications are delegated to optimized GPU kernels. The remaining ReLU and
elementwise operations are visible as separate small kernels.

Writing matrix multiplication by hand would have duplicated highly optimized
work. Fusing only the small operations would have required a custom ONNX
Runtime operator or a parallel inference path, with disproportionate
maintenance and validation cost. More importantly, CUDA Graphs had already
removed most launch overhead at small batch sizes.

The experimental decision was therefore **not to write a custom CUDA kernel**.
This is an evidence-based engineering conclusion, not unfinished work.

## 17. Final synthesis

| Experiment               | Targeted problem                           | Observed result                                | Decision                          |
| ------------------------ | ------------------------------------------ | ---------------------------------------------- | --------------------------------- |
| Persistence              | Establish a minimum reference              | Very strong on plateaus; test RMSE 1.0489      | Retained baseline                 |
| Direct MLP               | Learn absolute moisture                    | Test RMSE 1.9633 and strong temporal bias      | Rejected                          |
| Residual MLP             | Resist absolute-level shift                | Test RMSE 0.9478 for validation-selected model | Best tested predictor             |
| DeepONet                 | Test a multi-horizon operator formulation  | Test RMSE 1.2112, worse than MLP               | Kept as GPU study target          |
| Normal CUDA              | Accelerate inference                       | Slower at small batch, 2.561× CPU at 4096      | Useful for grouped workloads      |
| Host-to-host I/O Binding | Control buffers and transfers              | Slower at every batch                          | Negative result retained          |
| CUDA Graphs              | Reduce launch cost                         | 2.244× CUDA at batch 1; neutral at large batch | Useful for small GPU batches      |
| GPU-resident I/O         | Remove transfers from measured path        | 0.5734 ms and 7.14 M samples/s at 4096         | Useful only in a GPU pipeline     |
| FP16                     | Exploit reduced precision and Tensor Cores | Slower through 128; 1.120× at 4096             | Large-batch option, not canonical |
| Custom fusion            | Reduce remaining small kernels             | Opportunity judged too small for cost          | No kernel added                   |

### Predictive conclusion

The residual MLP is the best tested predictor. Its simpler structure and
explicit trends fit this chronological dataset better than the evaluated
DeepONet. DeepONet is better neither than the MLP nor than overall persistence
on test.

### GPU conclusion

GPU usefulness depends on granularity. CPU minimizes latency for small batches.
GPU overtakes it between tested batches 32 and 128, after which throughput grows
through the large batches. Data location matters as much as computation:
removing transfers in a genuinely GPU-resident pipeline offers a much larger
gain than host-to-host I/O Binding.

### Engineering conclusion

Profiling was more useful than applying optimizations blindly. It justified
CUDA Graphs for small batch, showed the potential of a GPU-resident pipeline,
limited FP16's usefulness to large batches, and avoided custom CUDA development
without demonstrated benefit.

## 18. Limitations

- **Agronomic scope:** the main study covers one sector and a relatively short
  period of usable support. It does not demonstrate generalization to other
  soils, crops, or years.
- **Temporal shift:** mean moisture changes substantially between the three
  periods. Residual prediction reduces this problem without eliminating it.
- **Plateaus:** the high proportion of unchanged targets favors persistence and
  can make MAE and RMSE appear contradictory.
- **Scope of the DeepONet result:** six bounded training runs were conducted.
  The result does not prove that all neural operators underperform an MLP; it
  describes this architecture and dataset.
- **Random stability:** DeepONet displays greater seed-to-seed dispersion than
  desired.
- **Hardware:** performance was measured on one main configuration, an RTX 2060
  and i5-9600K. It does not automatically transfer to other generations.
- **Artificially enlarged batches:** beyond ten examples, the benchmark repeats
  the golden cases cyclically. The compute cost is valid for measuring the
  runtime, but does not represent a natural arrival of 4096 independent
  measurements.
- **GPU-resident contract:** its figures intentionally exclude transfers and
  are not comparable to complete host-to-host application latency.
- **Reduced precision:** FP16 changes predictions slightly. Its effect was not
  re-evaluated over the entire predictive test set.
- **Historical reproducibility:** scripts and selection logs are present, but
  not every intermediate checkpoint is part of a single command replaying the
  history. The frozen deployment experiment itself is reproducible from a
  clean clone.

## 19. Reproducibility

The main deployment path starts directly from the versioned ONNX model:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/deeponet_golden_test --cpu
./build/deeponet_golden_test --cuda
./build/deeponet_golden_test --cuda-iobinding
./build/deeponet_benchmark --provider all --warmup 100 --iterations 1000 --repetitions 3
```

Rebuilding ONNX artifacts from the frozen checkpoint is separate:

```bash
uv sync --project python
uv run --project python setup-onnx
```

It neither downloads the dataset nor trains anything. Details, prerequisites,
and hashes are documented in the
[reproducibility note](reproducibility.md), the
[ONNX contract](onnx_deployment.md), the
[C++ runtime document](onnx_runtime_cpp.md), and the
[GPU optimization report](gpu_optimization_experiments.md).

## 20. Conclusion

This experiment leads neither to “DeepONet is better” nor to “GPU is always
better.” The most convincing predictive model is the residual MLP, with a test
RMSE of 0.9478 versus 1.2112 for DeepONet. DeepONet nevertheless remains a
relevant target for studying deployment of a multi-horizon neural operator.

For this small network, CPU provides the best latency at low batch sizes. GPU
becomes useful when enough predictions are grouped: normal CUDA is 2.56× faster
than CPU at batch 4096. CUDA Graphs reduce orchestration overhead and accelerate
CUDA by 2.24× at batch 1, without beating CPU there. A genuinely GPU-resident
pipeline reveals potential throughput of 7.14 million samples per second at
batch 4096, while FP16 supplies a more modest 1.12× gain with a small numerical
difference.

The most general conclusion is methodological: the residual formulation was
chosen from the behavior of the data, then GPU optimizations were chosen from
profiling. This approach produced measured gains and also made it possible to
reject ideas cleanly—host-to-host I/O Binding and a custom CUDA kernel—when they
were not justified.

## Essential references

- Arnesano v2 dataset: [Mendeley Data, DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2).
- Lu et al., *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, [Nature Machine Intelligence, 2021](https://doi.org/10.1038/s42256-021-00302-5).
- [ONNX Runtime — CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- [ONNX Runtime — I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).
- [NVIDIA Nsight Systems — User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- [NVIDIA CUDA C++ Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).
