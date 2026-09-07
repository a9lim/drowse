FROM ubuntu:22.04@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc

ARG DEBIAN_FRONTEND=noninteractive
ARG TVM_COMMIT=68ba2b31c2a6d202fcd44e5780ef10ce08721dd5
ARG TVM_FFI_COMMIT=12dbf053b3d9ba4ebd9da3123b1aeca79cf74229
ARG MLC_COMMIT=9fa644f54b04983adea4d0168f49fc6af4a893ba
ARG EMSDK_COMMIT=e10826f9196bb0e704014af2252ef58fb4ba261b
ARG EMCC_VERSION=3.1.56
ARG TARGETARCH
ARG LLVM_ARCHIVE_AMD64=clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
ARG LLVM_SHA256_AMD64=54ec30358afcc9fb8aa74307db3046f5187f9fb89fb37064cdde906e062ebf36
ARG LLVM_ARCHIVE_ARM64=clang+llvm-18.1.8-aarch64-linux-gnu.tar.xz
ARG LLVM_SHA256_ARM64=dcaa1bebbfbb86953fdfbdc7f938800229f75ad26c5c9375ef242edad737d999

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git libtinfo-dev libtinfo5 ninja-build \
    pkg-config python3 python3-dev python3-pip python3-venv xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN COMPILER_ARCH="${TARGETARCH:-$(dpkg --print-architecture)}" \
    && case "${COMPILER_ARCH}" in \
      amd64) LLVM_ARCHIVE="${LLVM_ARCHIVE_AMD64}"; LLVM_SHA256="${LLVM_SHA256_AMD64}" ;; \
      arm64) LLVM_ARCHIVE="${LLVM_ARCHIVE_ARM64}"; LLVM_SHA256="${LLVM_SHA256_ARM64}" ;; \
      *) echo "unsupported compiler architecture: ${COMPILER_ARCH}" >&2; exit 1 ;; \
    esac \
    && curl --fail --location --retry 5 \
      "https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/${LLVM_ARCHIVE}" \
      --output /tmp/llvm.tar.xz \
    && echo "${LLVM_SHA256}  /tmp/llvm.tar.xz" | sha256sum --check --strict \
    && mkdir /opt/llvm \
    && tar --extract --xz --file /tmp/llvm.tar.xz --strip-components=1 --directory /opt/llvm \
    && rm /tmp/llvm.tar.xz \
    && test "$(/opt/llvm/bin/llvm-config --version)" = "18.1.8"

RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
      dpkg --add-architecture amd64 \
      && sed -i 's|^deb http://ports|deb [arch=arm64] http://ports|' /etc/apt/sources.list \
      && printf '%s\n' \
        'deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy main universe' \
        'deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy-updates main universe' \
        'deb [arch=amd64] http://security.ubuntu.com/ubuntu jammy-security main universe' \
        > /etc/apt/sources.list.d/amd64.list \
      && apt-get update \
      && apt-get install -y --no-install-recommends \
        libc6:amd64 libstdc++6:amd64 libtinfo5:amd64 zlib1g:amd64 \
      && rm -rf /var/lib/apt/lists/*; \
    fi

ENV EMSDK_ARCH=x86_64

RUN git clone --filter=blob:none https://github.com/emscripten-core/emsdk.git /opt/emsdk \
    && git -C /opt/emsdk checkout "${EMSDK_COMMIT}" \
    && /opt/emsdk/emsdk install "${EMCC_VERSION}" \
    && /opt/emsdk/emsdk activate "${EMCC_VERSION}"

ENV PATH=/opt/llvm/bin:/opt/emsdk:/opt/emsdk/upstream/emscripten:/opt/venv/bin:${PATH}
ENV TVM_SOURCE_DIR=/opt/tvm
ENV TVM_LIBRARY_PATH=/opt/tvm/build
ENV TVM_DEVICE_BACKEND_AUTOLOAD=0
ENV SKIP_LOADING_MLCLLM_SO=1
ENV PYTHONPATH=/opt/tvm/python:/opt/mlc-llm/python

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 \
    && pip install --no-cache-dir \
      cloudpickle==3.1.1 cmake==3.31.6 ml_dtypes==0.5.1 ninja==1.11.1.4 \
      numpy==2.2.6 packaging==25.0 \
      psutil==7.0.0 pydantic==2.11.5 safetensors==0.5.3 \
      sentencepiece==0.2.0 shortuuid==1.0.13 tiktoken==0.9.0 \
      tqdm==4.67.1 transformers==4.52.4

RUN git clone --filter=blob:none https://github.com/apache/tvm.git /opt/tvm \
    && git -C /opt/tvm checkout "${TVM_COMMIT}" \
    && git -C /opt/tvm submodule update --init --recursive \
    && test "$(git -C /opt/tvm/3rdparty/tvm-ffi rev-parse HEAD)" = "${TVM_FFI_COMMIT}"

RUN git clone --filter=blob:none https://github.com/mlc-ai/mlc-llm.git /opt/mlc-llm \
    && git -C /opt/mlc-llm checkout "${MLC_COMMIT}" \
    && git -C /opt/mlc-llm submodule update --init --recursive -- \
      3rdparty/argparse 3rdparty/googletest 3rdparty/stb \
      3rdparty/tokenizers-cpp 3rdparty/xgrammar

COPY forks/tvm-webgpu-readonly.patch /opt/overlays/tvm-webgpu-readonly.patch
RUN git -C /opt/tvm apply --check /opt/overlays/tvm-webgpu-readonly.patch \
    && git -C /opt/tvm apply /opt/overlays/tvm-webgpu-readonly.patch

ENV CMAKE_BUILD_PARALLEL_LEVEL=8

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list \
    && find /etc/apt/sources.list.d -type f -name '*.list' -exec sed -i 's|http://|https://|g' {} + \
    && apt-get update \
    && apt-get install -y --no-install-recommends libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      cython==3.3.0 exceptiongroup==1.3.1 pathspec==1.1.1 \
      packaging==26.3 scikit-build-core==1.0.3 setuptools==84.0.0 setuptools-scm==10.2.1 \
      tomli==2.4.1 vcs-versioning==2.3.1

RUN pip install --no-cache-dir --no-build-isolation /opt/tvm/3rdparty/tvm-ffi \
    && cmake -S /opt/tvm -B /opt/tvm/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_LLVM=/opt/llvm/bin/llvm-config \
      -DUSE_CUDA=OFF -DUSE_OPENCL=OFF -DUSE_VULKAN=OFF \
      -DUSE_METAL=OFF -DUSE_ROCM=OFF -DUSE_RPC=OFF -DBUILD_TESTING=OFF \
    && cmake --build /opt/tvm/build --parallel 8

COPY forks/mlc-llm-drowse.patch /opt/overlays/mlc-llm-drowse.patch
RUN git -C /opt/mlc-llm apply --check /opt/overlays/mlc-llm-drowse.patch \
    && git -C /opt/mlc-llm apply /opt/overlays/mlc-llm-drowse.patch

RUN make -C /opt/mlc-llm/web clean \
    && make -C /opt/tvm/web clean \
    && make -C /opt/mlc-llm/web \
    && TVM_HOME=/opt/tvm make -C /opt/tvm/web

COPY fixtures /opt/drowse/fixtures
COPY forks/verify-mlc-hook.py /opt/drowse/verify-mlc-hook.py
COPY forks/compile-tiny-webgpu.py /opt/drowse/compile-tiny-webgpu.py
COPY forks/create-tiny-webgpu-model.py /opt/drowse/create-tiny-webgpu-model.py
COPY forks/verify-tvm-attention.py /opt/drowse/verify-tvm-attention.py

RUN python /opt/drowse/verify-tvm-attention.py \
    && python /opt/drowse/verify-mlc-hook.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --fixture /opt/drowse/fixtures/post-block-rank-one-v2.json \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture qwen3 --output /tmp/drowse-tiny-qwen3.wasm \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture llama --output /tmp/drowse-tiny-llama.wasm \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture gemma3_text --output /tmp/drowse-tiny-gemma3.wasm \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture qwen3 --quantization q4f16_1 \
      --output /tmp/drowse-tiny-qwen3-q4f16.wasm \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture llama --quantization q4f16_1 \
      --output /tmp/drowse-tiny-llama-q4f16.wasm \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture gemma3_text --quantization q4f16_1 \
      --output /tmp/drowse-tiny-gemma3-q4f16.wasm \
    && python /opt/drowse/compile-tiny-webgpu.py \
      --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm \
      --architecture gemma3_text --quantization q4f32_1 \
      --output /tmp/drowse-tiny-gemma3-q4f32.wasm \
    && rm /tmp/drowse-tiny-qwen3.wasm /tmp/drowse-tiny-llama.wasm \
      /tmp/drowse-tiny-gemma3.wasm /tmp/drowse-tiny-qwen3-q4f16.wasm \
      /tmp/drowse-tiny-llama-q4f16.wasm /tmp/drowse-tiny-gemma3-q4f16.wasm \
      /tmp/drowse-tiny-gemma3-q4f32.wasm

COPY forks/licenses /opt/drowse/licenses
COPY forks/build-production-webgpu.py /opt/drowse/build-production-webgpu.py
COPY forks/base_model_adapters.py /opt/drowse/base_model_adapters.py
COPY forks/qwen35_adapter.py /opt/drowse/qwen35_adapter.py
COPY forks/hybrid_state_kernels.py /opt/drowse/hybrid_state_kernels.py

WORKDIR /work
ENTRYPOINT ["/bin/bash", "-lc"]
