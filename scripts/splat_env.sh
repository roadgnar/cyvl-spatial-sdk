# Source this to run nerfstudio (ns-train / ns-export / ns-render) on this box.
#   source scripts/splat_env.sh
# gsplat JIT-compiles CUDA kernels against CUDA_HOME; we pin 12.8 to match the
# torch cu128 build (the default /usr/local/cuda -> 13.2 is newer than torch's
# toolchain and can fail the nvcc compile). ninja must be on PATH for the build.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HERE/.venv-ns/bin:/usr/local/cuda-12.8/bin:$PATH"
