# Real Application Evidence Runbook

This runbook is the promotion path from the current checksum-valid control
stages to a real dependent application result. The proposed system name is
**QUIET**; generated control MLP runs remain exploratory.

## Required Artifacts

1. A newline-complete dataset manifest. Each row must contain exactly
   `schema_version`, `sample_id`, `input_sha256`, and `expected_label`.
2. A serialized producer TensorRT engine and a trained downstream TensorRT
   engine whose input tensor accepts the producer activation shape.
3. Reference and candidate JSONL traces validated by
   `analysis/verify_application_accuracy.py`.
4. The same arrival trace, placement, MIG inventory, engine hashes, wall
   deadline lock, and inline correctness mode for every comparator.

For image/audio experiments, create the dataset manifest from the raw samples
and an external label map first:

```bash
python3 scripts/build_application_dataset_manifest.py \
  --root /path/to/samples --labels /path/to/labels.json \
  --pattern '*.jpg' --output results/application-dataset.jsonl
```

The command rejects missing or unused labels and writes a separate provenance
record. It intentionally does not infer labels from filenames or predictions.

Bind those labeled samples to the measured iteration numbers before collecting
production traces:

```bash
python3 scripts/build_application_request_manifest.py \
  --dataset-manifest results/application-dataset.jsonl \
  --sample-list results/resnet10-labelled-subset/samples.jsonl \
  --warmup 10 --requests 100 \
  --request-id-prefix resnet-head \
  --output results/application-requests.jsonl
```

The request manifest uses iterations `10..109` for the 100 measured requests,
keeps arrival sequence `0..99`, and refuses to reuse a labeled sample. Its
provenance records the dataset-manifest SHA; a producer trace with different
input payload hashes is rejected by the accuracy gate rather than silently
treated as the same workload. Explicit duplicate bytes are allowed only when
their externally supplied labels agree; conflicting labels fail closed.

The producer must consume the same preprocessed tensor bytes used to create
the input hashes. Pack those tensors into the benchmark's fixed-size `JDGINT1`
trace; the producer then copies each record into its TensorRT input buffer and
adds the bound hash to `pipeline.csv`:

For the vendor ResNet10 detector, `scripts/prepare_resnet10_labeled_samples.py`
can create those tensors and expected detection labels from an externally
annotated COCO-style subset. It fixes the Jetson preprocessing contract
(`640x368`, BGR-to-RGB, float32 NCHW, scale `1/255`) and requires an explicit
category-ID map; the vendor parser emits only `Car`, `RoadSign`, and
`TwoWheeler` (it excludes the final `labels.txt` entry), so
annotations outside those three output classes must be listed explicitly with
`--ignore-category`. Missing, unsupported, or unlisted annotations fail
closed, and filenames or model predictions are never used as labels:

For KITTI-style detection data, convert labels explicitly before preparing the
sample list. The converter rejects every unmapped class and rejects target
labels that the vendor parser cannot emit; it never silently maps `Person` or
other parser-excluded classes:

```bash
python3 scripts/convert_kitti_to_coco.py \
  --image-dir /path/to/kitti/training/image_2 \
  --label-dir /path/to/kitti/training/label_2 \
  --map Car=Car \
  --map Van=Car \
  --map Cyclist=TwoWheeler \
  --ignore Pedestrian \
  --ignore Tram \
  --ignore Misc \
  --ignore DontCare \
  --output results/kitti-resnet10/annotations.json \
  --category-map-output results/kitti-resnet10/category-map.json
```

The resulting COCO annotations must still be checked against the external
dataset policy (for example, whether `Van` is scientifically merged into
`Car`). That merge is recorded in `source.class_mapping` and is part of the
dataset-manifest hash used by every arm.

```bash
python3 scripts/prepare_resnet10_labeled_samples.py \
  --image-root /path/to/images \
  --annotations /path/to/instances.json \
  --category-map /path/to/resnet10-category-map.json \
  --ignore-category 16 --ignore-category 17 \
  --output-dir results/resnet10-labelled-subset
```

The generated `samples.jsonl` and `dataset-manifest.jsonl` are consumed by the
commands below. The provenance records image, annotation, category-map, and
vendor preprocessing hashes.

```bash
python3 scripts/build_producer_input_trace.py \
  --sample-list results/resnet10-labelled-subset/samples.jsonl \
  --output results/resnet10-labelled-subset/inputs.bin \
  --provenance results/resnet10-labelled-subset/inputs.json
```

Each sample-list row must contain exactly `iteration`, `sample_id`, `path`,
and `input_sha256`; files must be distinct, fixed-size, and already in the
TensorRT engine's input layout. Pass the resulting binary to every arm as
`--producer-input-trace` and use `--require-input-binding` when building the
application prediction trace. Without this flag the legacy four-pattern smoke
remains available for diagnostics but cannot prove a real input workload.

For an independent/dependent causal pair, capture the producer's actual
activation bytes before either measured arm. The capture is producer-only and
does not contribute to the production wall:

```bash
build-r39/jdg-mig-trt-pipeline \
  --producer-engine models/engines/mig-1g-q100/resnet10-detection.engine \
  --producer "$JDG_MIG_SMALL_UUID" \
  --producer-mps-pipe "$JDG_MPS_PIPE_DIRECTORY" \
  --producer-input-trace results/resnet10-labelled-subset/inputs.bin \
  --capture-activation-trace results/resnet10-labelled-subset/activations.bin
```

Both arms must consume that same `JDGACT1` trace with
`--activation-replay-trace`. The dependent arm keeps the live producer-to-
consumer precedence edge; the independent arm removes only that precedence
and directly binds the request-indexed registered replay slot. The independent
arm no longer accepts a synthetic CPU `memset` input. The capture trace
validator is `scripts/verify_activation_replay_trace.py`; request-level output
traces must also match before the pair is treated as a causal contract.

Build and verify the operational `JDGARR1` release schedule before launching
either arm:

```bash
python3 scripts/build_operational_arrival_trace.py \
  --request-trace results/application-requests.jsonl \
  --period-us 2000 \
  --output results/application-arrivals.bin
python3 scripts/verify_operational_arrival_trace.py \
  results/application-arrivals.bin
```

Pass it as `--operational-arrival-trace` to the runner (which forwards
`--arrival-trace` to the C++ pipeline) and use
`--require-operational-arrival-trace` for a fail-closed run. The scheduler
anchors declared offsets to one measurement epoch; late one-buffer releases
are recorded as queue delay rather than shifting the next declared arrival.
Each arm emits `events.csv` with pause completion, publication, resume
observation, validation timestamps, and total gate hold.

The repository includes a real learned dependent DAG derived from the vendor
ResNet10 detector. `scripts/prepare_resnet10_real_dag.py` splits the graph at
`Layer6_relu_Y` and preserves the original learned detection head outputs
(`Layer7_cov`, `Layer7_bbox`) instead of substituting the generated control
MLP. The cross-MIG payload is `1x512x23x40` (`1,884,160` bytes). Build it on
the matching MIG slices with the `onnx` Python package and record the generated
manifest before running the benchmark.

For a learned split, run `scripts/verify_split_dag_equivalence.py` on a
representative float32 input before starting a TensorRT campaign. The verifier
infers static spatial dimensions from the ONNX graph (including dynamic batch
size); use `--input-shape N,C,H,W` only when spatial dimensions are dynamic. A
passed record proves full-model versus producer/head graph equivalence only; it
is not a task-accuracy result.

## Collection

The production runner accepts the downstream artifact directly:

Before launching any arm, build one shared contract and pass the same file to
QUIET and every comparator. The request manifest is the canonical arrival/input
sequence; the dataset manifest is the external label authority.

```bash
python3 scripts/build_common_workload_contract.py \
  --workload-id resnet-detection-head \
  --topology fixed-2g+1g \
  --placement fixed-1g-producer-2g-consumer \
  --input-tensor Layer6_relu_Y \
  --payload-bytes 1884160 \
  --arrival-trace results/application-requests.jsonl \
  --dataset-manifest results/application-dataset.jsonl \
  --producer-input-trace results/preprocessed-resnet10-inputs.bin \
  --output results/common-workload.json
```

```bash
python3 scripts/run_p9_dependent_stress_smoke.py \
  --workload resnet-detection-head \
  --consumer-engine /path/to/trained-downstream.engine \
  --consumer-input-tensor Layer6_relu_Y \
  --common-workload-contract results/common-workload.json \
  --require-common-workload \
  --producer-input-trace results/preprocessed-resnet10-inputs.bin \
  --checksum-mode inline \
  --application-output-trace-dir results/p9-application-outputs \
  --deadline-lock /path/to/common-deadline-lock.json \
  --result-dir results/p9-real-application-quiet
```

The resulting pipeline JSON records `consumer_engine_mode` as
`external-trained-engine`; omission of `--consumer-engine` records
`generated-control-policy` and cannot satisfy the accuracy gate.
For a short directional smoke whose measured gate exceeds the selected plan,
add `--allow-plan-diagnostic`. The run is explicitly tagged
`diagnostic-only-plan-violation` and must never enter the formal frontier; the
default path remains fail-closed.
When `--quiet-plan` is supplied, the runner materializes a pre-launch
`quiet-execution.json` manifest and derives the producer/consumer MIG UUIDs,
registered/direct transport, producer/background/consumer quotas, protection
scope, and dependent admission from that plan before starting any CUDA
process. A CLI value that differs from the selected plan is rejected. Runs
without a plan remain explicitly exploratory CLI characterization and cannot
be treated as planner-actuated evidence.
For the split detector use `--workload resnet-detection-head`; the runner
defaults the external input tensor to `Layer6_relu_Y` for that workload.

After collection, join the post-completion output bytes to the measured wall
CSV and the externally owned labels. This creates the exact trace schema
consumed by the accuracy gate; it does not infer labels from model outputs:

```bash
python3 analysis/build_application_prediction_trace.py \
  --output-trace results/p9-application-outputs/quiet/outputs.bin \
  --pipeline-csv results/p9-real-application-quiet/quiet/pipeline.csv \
  --request-manifest results/application-requests.jsonl \
  --class-map /path/to/imagenet-class-map.json \
  --warmup 10 --deadline-us FROZEN_DEADLINE_US --require-input-binding \
  --output results/p9-real-application-quiet/quiet/application-trace.jsonl
```

The output trace binds prediction, raw output digest, request input digest,
expected label, and production wall latency for every measured request. Build
one trace per comparator from the same request manifest before invoking the
accuracy gate.

For the learned ResNet10 detector, do not use classifier argmax.  The
production output container has `Layer7_cov` and `Layer7_bbox`; use the vendor
stride/normalization/NMS decoder and an externally labelled detection manifest:

```bash
python3 analysis/build_application_prediction_trace.py \
  --output-trace results/p9-application-outputs/quiet/outputs.bin \
  --pipeline-csv results/p9-real-application-quiet/quiet/pipeline.csv \
  --request-manifest results/application-requests.jsonl \
  --class-map models/resnet10-output-class-map.json \
  --prediction-mode resnet10-detection \
  --warmup 10 --deadline-us FROZEN_DEADLINE_US --require-input-binding \
  --output results/p9-real-application-quiet/quiet/application-trace.jsonl
```

The detector mode is only a decoder: it never creates labels or turns the
current four-byte-pattern smoke inputs into an accuracy result.  The accuracy
gate remains pending until the manifest contains real image hashes and
ground-truth detections shared by QUIET and every comparator.

For the split ResNet-50 classification DAG, use the ImageNet WNID contract
below.  The preparation tool requires an explicit map from every WNID in the
image directory to the model output index, human-readable class, and label
source; a filename is only a join key and is never treated as a label oracle:

```bash
python3 scripts/prepare_resnet50_imagenet_samples.py \
  --image-root /path/to/extracted/imagenet_data \
  --source-archive /path/to/imagenet_data.tar \
  --synset-map docs/p9-resnet50-imagenet-mini-synset-map.json \
  --limit-per-synset 10 \
  --output-dir results/resnet50-imagenet-labelled-subset

python3 scripts/build_producer_input_trace.py \
  --sample-list results/resnet50-imagenet-labelled-subset/samples.jsonl \
  --output results/resnet50-imagenet-labelled-subset/inputs.bin \
  --provenance results/resnet50-imagenet-labelled-subset/inputs.json
```

The preprocessing is the ONNX Model Zoo ResNet-50 v2 contract: RGB, resize
short side to 256, center crop 224, float32 NCHW, and ImageNet mean/std
normalization.  The generated `provenance.json` binds the archive, images,
synset map, tensor hashes, and manifests.  The earlier six-synset 59-request
run remains a negative diagnostic (`0.711864`, below the frozen `0.80`
threshold); its latency is not a numeric application claim.  The promoted
learned-head gate uses the standard ImageNette validation split and the
explicit map `docs/p9-resnet50-imagenette-synset-map.json`.  Its 100-sample
labelled subset has two warmups and 90 measured requests; the CPU composed
reference and cross-MIG registered/direct split both reach `0.8333` accuracy
with zero delta, zero deadline misses, input-bound wall CSVs, and independent
post-completion `JDGOUT1` traces.  The locked artifact is
`results/p9-resnet50-imagenette-gate100-20260811/accuracy-gate.json`.

For an audio/ASR DAG, the request manifest's `expected_label` is the externally
owned reference transcript. The production decoder must emit a JSON object
mapping each post-completion output SHA-256 to its transcript; the mapping is
decoder output metadata, not a source of labels:

```bash
python3 analysis/build_application_prediction_trace.py \
  --output-trace results/p9-application-outputs/quiet/outputs.bin \
  --pipeline-csv results/p9-real-application-quiet/quiet/pipeline.csv \
  --request-manifest results/audio-requests.jsonl \
  --prediction-mode asr \
  --transcript-map results/p9-real-application-quiet/quiet/asr-transcripts.json \
  --warmup 10 --deadline-us FROZEN_DEADLINE_US --require-input-binding \
  --output results/p9-real-application-quiet/quiet/asr-application-trace.jsonl
```

The ASR accuracy gate recomputes request correctness from normalized word
error rate (WER), then requires both arms to satisfy `--asr-max-wer` and the
candidate/reference mean-WER tolerance. A Whisper encoder-only activation or
synthetic transcript is not an ASR result; a real audio dataset, decoder, and
shared external transcripts must be present before this gate can promote an
audio workload.

The checked-in real Whisper smoke uses the official LibriSpeech `dev-clean`
transcript sidecars, canonical relative-path order, and a frozen 12-sample
prefix (two warmups plus ten measured requests). Prepare the mel features and
JDGINT1 inputs as follows; the generated provenance records the dataset and
selection rule:

```bash
python3 scripts/prepare_whisper_asr_samples.py \
  --dataset-root results/p9-real-asr-dataset/dev_clean \
  --mel-model models/cache/whisper-tiny-mel.onnx \
  --output-dir results/p9-real-whisper-asr-lex12-20260811 \
  --pattern '*.flac' --limit 12
python3 scripts/build_producer_input_trace.py \
  --sample-list results/p9-real-whisper-asr-lex12-20260811/samples.jsonl \
  --output results/p9-real-whisper-asr-lex12-20260811/inputs.bin \
  --provenance results/p9-real-whisper-asr-lex12-20260811/provenance.json
python3 scripts/build_application_request_manifest.py \
  --dataset-manifest results/p9-real-whisper-asr-lex12-20260811/dataset-manifest.jsonl \
  --sample-list results/p9-real-whisper-asr-lex12-20260811/samples.jsonl \
  --warmup 2 --requests 10 --request-id-prefix whisper-asr \
  --output results/p9-real-whisper-asr-lex12-20260811/requests.jsonl
```

Build the FP32 encoder on the 1g producer and FP32 initial/with-past decoders
on the 2g consumer with `scripts/prepare_whisper_asr_models.sh`. The resident
MPS daemon is scoped to the 1g producer; the 2g consumer intentionally clears
the MPS pipe and uses its direct MIG context. Run the split application with
`jdg-mig-whisper-asr`, then decode the post-completion token trace with the
pinned tokenizer:

```bash
build-r39/jdg-mig-whisper-asr \
  --encoder-engine models/engines/mig-1g-q100/whisper-tiny-encoder-fp32.engine \
  --decoder-initial-engine models/engines/mig-2g-q100/whisper-tiny-decoder-initial-4-fp32.engine \
  --decoder-with-past-engine models/engines/mig-2g-q100/whisper-tiny-decoder-with-past-fp32.engine \
  --input-trace results/p9-real-whisper-asr-lex12-20260811/inputs.bin \
  --output-trace results/p9-real-whisper-asr-lex12-20260811/asr-output.bin \
  --trace-csv results/p9-real-whisper-asr-lex12-20260811/pipeline.csv \
  --warmup 2 --iterations 10 --max-tokens 128 --deadline-us 1000000 \
  --producer "$JDG_MIG_SMALL_UUID" --consumer "$JDG_MIG_BIG_UUID" \
  --mps-pipe "$JDG_MPS_PIPE_DIRECTORY"
python3 analysis/decode_whisper_token_trace.py \
  --trace results/p9-real-whisper-asr-lex12-20260811/asr-output.bin \
  --tokenizer models/cache/whisper-tiny-tokenizer.json \
  --output results/p9-real-whisper-asr-lex12-20260811/transcript-map.json
python3 analysis/build_application_prediction_trace.py \
  --output-trace results/p9-real-whisper-asr-lex12-20260811/asr-output.bin \
  --pipeline-csv results/p9-real-whisper-asr-lex12-20260811/pipeline.csv \
  --request-manifest results/p9-real-whisper-asr-lex12-20260811/requests.jsonl \
  --prediction-mode asr \
  --transcript-map results/p9-real-whisper-asr-lex12-20260811/transcript-map.json \
  --warmup 2 --deadline-us 1000000 --asr-max-wer 0.20 \
  --require-input-binding \
  --output results/p9-real-whisper-asr-lex12-20260811/application-trace.jsonl
```

The CPU ONNX reference is `scripts/run_whisper_onnx_reference.py`; run it on
the same sample list and invoke `analysis/verify_application_accuracy.py` with
both raw `JDGASR1` traces, both input-bound timing CSVs, and
`--require-output-traces`. The formal gate in
`results/p9-real-whisper-asr-lex12-20260811/accuracy-gate.json` passes at
minimum accuracy `0.90`, maximum mean/request WER `0.20`, and WER delta `0.02`.

The `--class-map` passed to detection decoding is the model-output map from
the vendor `labels.txt`: indices `0..3` must be `Car`, `RoadSign`,
`TwoWheeler`, `Person` in that order. The final `Person` slot is present in
the tensor but ignored by the vendor parser. Do not pass the 1-based COCO
`category-map.json` generated by the KITTI converter as `--class-map`.

The rejected COCO8 smoke is retained only as a negative-control artifact. Its
person-only labels target the parser-excluded final class and therefore yielded
zero valid task matches; do not lower the accuracy threshold or reuse its
timing rows as application evidence.

For native XSched controls, set `APPLICATION_OUTPUT_TRACE` when invoking
`scripts/run_p9_xsched_resnet_control_smoke.sh` or
`scripts/run_p9_xsched_dependent_smoke.sh`. The runner passes
`--application-output-trace` to the production binary and the verifier binds
its SHA with `capture_boundary=post-completion`; omitting it is allowed only
for exploratory checksum-only runs.

For the executable SOTA sequence, pass the same contract to the Williams
runner. Omitting it is allowed only for historical smoke replay and keeps the
aggregate non-promoting:

```bash
python3 scripts/run_p9_common_sota_williams.py \
  --active-only --sequence-index 0 --requests 100 \
  --workload resnet-detection-head \
  --consumer-engine results/p9-real-resnet-head-artifacts-20260810/resnet10-detection-head.engine \
  --common-workload-contract results/common-workload.json \
  --deadline-lock results/deadline-lock.json \
  --quiet-plan results/quiet-selection.json \
  --result-dir results/p9-williams-seq0
```

Convert an XSched trace with the same request manifest used by QUIET:

```bash
python3 analysis/build_application_prediction_trace.py \
  --output-trace results/xsched/application-outputs.bin \
  --pipeline-csv results/xsched/pipeline.csv \
  --request-manifest results/application-requests.jsonl \
  --class-map /path/to/imagenet-class-map.json \
  --warmup 10 --deadline-us FROZEN_DEADLINE_US \
  --output results/xsched/application-trace.jsonl
```

For a quick hardware sanity check without waiting for formal locks, use the
dedicated learned-head pair wrapper. It runs 20 production-wall requests for
QUIET and NVIDIA MPS with inline correctness and post-completion output traces:

```bash
DEADLINE_US=3127.355209 \
  scripts/run_p9_learned_head_fast_pair.sh
```

The resulting pair is explicitly exploratory: it has no common arrival
contract, task-accuracy gate, or thermal normalization and cannot enter a
paper frontier.

For the native XSched comparator, run the same learned ResNet10
backbone-to-detection-head workload through the XQueue path:

```bash
RESULT_DIR=results/xsched-resnet-head-current \
DEADLINE_LOCK=results/current-resnet-head-deadline/deadline-lock.json \
WORKLOAD=resnet-detection-head \
CONSUMER_ENGINE=results/p9-real-resnet-head-artifacts-20260810/resnet10-detection-head.engine \
CONSUMER_INPUT_TENSOR=Layer6_relu_Y \
CRITICAL_REQUESTS=100 WARMUP=5 BE_REQUESTS=5000 \
APPLICATION_OUTPUT_TRACE=results/xsched-resnet-head-current/application-outputs/xsched/outputs.bin \
bash scripts/run_p9_xsched_dependent_smoke.sh
```

This runner checks the native XQueue suspend/resume path, learned consumer
engine and tensor, production-wall completion, and inline payload checks. The
deadline lock must be regenerated with the current binary/source hashes; a
stale-lock attempt is diagnostic only and cannot become a comparator row.

## Promotion

Before starting any Williams sequence, run the fail-fast preflight.  It only
reads files and `nvidia-smi`; it never fabricates an arrival trace, label, or
thermal lock.  A nonzero exit is expected until the external dataset/label
manifest, common workload contract, accuracy gate, and current thermal and
deadline locks are all present:

```bash
python3 analysis/preflight_p9_campaign.py \
  --mig-env /tmp/jdg-mps-1g/mig.env \
  --common-workload-contract results/common-workload.json \
  --thermal-lock results/thermal-lock.json \
  --deadline-lock results/deadline-lock.json \
  --accuracy-gate results/application-accuracy-gate.json \
  --output results/p9-campaign-preflight.json
```

The report distinguishes `exploratory_ready` from `formal_ready` and records
the SHA-256 of every input it accepted.  Do not use an old thermal lock merely
because it exists: the verifier must accept the lock under the current active
boundary protocol.

Run the accuracy gate on traces from the same request set:

```bash
python3 analysis/verify_application_accuracy.py \
  --reference-trace reference.jsonl \
  --candidate-trace quiet.jsonl \
  --dataset dataset-manifest.jsonl \
  --reference-engine reference.engine \
  --candidate-engine trained-downstream.engine \
  --workload resnet-control \
  --task classification \
  --deadline-us FROZEN_DEADLINE_US \
  --minimum-accuracy 0.90 \
  --reference-pipeline-csv reference/pipeline.csv \
  --candidate-pipeline-csv quiet/pipeline.csv \
  --pipeline-warmup 10 \
  --reference-output-trace reference.outputs.bin \
  --candidate-output-trace quiet.outputs.bin \
  --output-trace-warmup 10 \
  --output-trace-index 0 \
  --require-output-traces \
  --require-input-binding \
  --output application-accuracy-gate.json
```

The two production CSVs are independently checked for dense post-warmup
iterations, input SHA equality, wall latency equality, and deadline
classification. `--minimum-accuracy` is an absolute gate for both reference
and candidate; equal predictions with equally poor accuracy are rejected.
For `--task object-detection`, correctness is recomputed from the external
labels using class-aware one-to-one IoU matching (default threshold 0.5), not
canonical JSON string equality. Only a passed, byte-bound gate may be attached
to a formal frontier artifact.
When summarizing a common MPS/XSched/QUIET load sweep, pass
`--require-application-accuracy --require-output-traces`; exploratory
checksum-only points must remain
descriptive and cannot receive a numeric frontier label.
The repository contains learned ResNet10 and ResNet50 split-DAG preparation
tools.  The ResNet10 detector remains graph-equivalence-only pending external
box annotations, while the ResNet50/ImageNette classifier now has a passed
dataset-label reference/candidate gate.  Keep these task boundaries explicit;
the classifier gate does not promote the detector or any comparator.
