# Multimodal TensorRT workloads

The evaluation uses one batch-1 model per modality rather than treating several
vision networks as a multimodal workload:

| Model | Modality | Fixed evaluation shape | Source |
|---|---|---|---|
| ResNet-50 v2 | Vision | `1x3x224x224` | ONNX Model Zoo |
| ResNet10 detector | Vision | `data:1x3x368x640` | Jetson Multimedia API |
| DistilBERT SST-2 | Language | sequence length 128 | Hugging Face DistilBERT |
| Whisper-tiny encoder/decoder | Audio | 3,000 log-Mel frames, 128-token greedy decode | Hugging Face export |

`manifest.json` pins the download locations and SHA-256 digests. Model weights
and generated TensorRT engines are deliberately ignored by Git. Run
`scripts/prepare_models.sh` to verify/download the ONNX files and build FP16
TensorRT engines for the active CUDA device. `ENGINE_TAG` identifies the exact
MIG/MPS resource view used for tactic selection. The experiment scripts build
separate engines for `2g` and every `1g`/`2g` MPS quota; reusing a 12-SM plan
when MPS exposes only 2--8 SMs is invalid and TensorRT warns that it can
deadlock. Engines are platform artifacts and must be rebuilt after a TensorRT,
CUDA, GPU architecture, MIG profile, or MPS quota change.

The real ASR gate uses the complete Whisper graph rather than the historical
encoder-only projection. Run `scripts/prepare_whisper_asr_models.sh` with
`JDG_MIG_SMALL_UUID` and `JDG_MIG_BIG_UUID` set to download the pinned mel,
encoder, initial-decoder, autoregressive-decoder, and ByteLevel tokenizer
artifacts and build the split FP32 engines. FP32 is intentional: the FP16
decoder changes greedy token choices on the frozen labelled LibriSpeech
contract. The split runner binds the encoder's `last_hidden_state` directly
to the registered shared system-memory edge and binds that same activation as
the decoder's `encoder_hidden_states` input.

The historical timing smoke uses synthetic inputs because the evaluated graphs
have input-independent control flow. That mode is not an application-accuracy
claim; the real dependent split below preserves the learned model path.

## Real dependent split

For the application-DAG gate, use
`scripts/prepare_resnet10_real_dag.py` rather than the generated control MLP.
It splits the learned ResNet10 detector at `Layer6_relu_Y`: the 1g producer
runs the learned backbone and the 2g consumer runs the original learned
`Layer7_cov`/`Layer7_bbox` detection head. The payload is `1x512x23x40`; the
source-model, ONNX, and engine hashes are recorded in the generated manifest.
A dataset-label accuracy gate is still required before formal promotion of
this ResNet10 detector. The separate ResNet50/ImageNette classifier has a
passed labelled application gate, but that result does not promote this
detector.
The vendor parser decodes `Car`, `RoadSign`, and `TwoWheeler`; it excludes the
final `labels.txt` slot (`Person`). The raw output tensor still has all four
slots, so accuracy decoding must use the exact output map
`[Car,RoadSign,TwoWheeler,Person]` and ignore only the final slot. Label
preparation must bind only the three emitted classes and fail closed for
unsupported categories.

For a learned classification DAG, `scripts/prepare_resnet50_real_dag.py`
extracts the ONNX Model Zoo ResNet-50 v2 graph at
`gpu_0/res4_5_branch2c_bn_2`. The tool repairs the older export's
initializer/input declarations before TensorRT validation and records both
split shapes and hashes in its manifest. It does not create labels or an
accuracy claim; build the external label manifest with
`scripts/build_application_dataset_manifest.py` and run the reference/candidate
gate separately.
