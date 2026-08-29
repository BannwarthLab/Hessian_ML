import sys
from typing import Dict, Tuple

# Configuration mappings for each component
OS_CONFIG = {
    "rl9": "rockylinux:9",
    "rl10": "rockylinux/rockylinux:10",
}

COMPILER_CONFIG = {
    "gnu": {
        "packages": "gcc-gfortran",
        "env": {"CC": "gcc", "CXX": "g++", "FC": "gfortran", "F77": "gfortran", "F90": "gfortran"},
    },
    "intel2023.2.0": {
        "repo": True,
        "packages": "intel-oneapi-compiler-dpcpp-cpp-2023.2.0 intel-oneapi-compiler-fortran-2023.2.0",
        "env": {
            "CC": "icx", "CXX": "icpx", "FC": "ifx", "F77": "ifx", "F90": "ifx",
            "PATH": "/opt/intel/oneapi/compiler/2023.2.0/linux/bin:$PATH",
            "TBBROOT": "/opt/intel/oneapi/tbb/2021.10.0",
            "COMPILERROOT": "/opt/intel/oneapi/compiler/2023.2.0",
        },
    },
    "intel2025.3.0": {
        "repo": True,
        "packages": "intel-oneapi-compiler-dpcpp-cpp-2025.3.0 intel-oneapi-compiler-fortran-2025.3.0",
        "env": {
            "CC": "icx", "CXX": "icpx", "FC": "ifx", "F77": "ifx", "F90": "ifx",
            "PATH": "/opt/intel/oneapi/compiler/2025.3.0/linux/bin:$PATH",
            "TBBROOT": "/opt/intel/oneapi/tbb/2022.3",
            "COMPILERROOT": "/opt/intel/oneapi/compiler/2025.3",
        },
    },
}

LAPACK_CONFIG = {
    "openblas": {
        "packages": "openblas-devel lapack-devel",
    },
    "mkl2023.2.0": {
        "repo": True,
        "packages": "intel-oneapi-mkl-devel-2023.2.0",
        "env": {"MKLROOT": "/opt/intel/oneapi/mkl/2023.2.0"},
    },
    "mkl2025.3.0": {
        "repo": True,
        "packages": "intel-oneapi-mkl-devel-2025.3.0",
        "env": {"MKLROOT": "/opt/intel/oneapi/mkl/2025.3"},
    },
}

GPU_CONFIG = {
    "none": {},
    "cuda11.8": {
        "packages": "cuda-compiler-11-8 cuda-cudart-devel-11-8 libcusolver-devel-11-8 libcublas-devel-11-8 libcusparse-devel-11-8",
        "env": {
            "PATH": "/usr/local/cuda-11.8/bin${PATH:+:${PATH}}",
            "CPATH": "/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH",
            "LD_LIBRARY_PATH": "/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
            "LIBRARY_PATH": "/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LIBRARY_PATH",
            "CUDA_HOME": "/usr/local/cuda-11.8",
        },
    },
    "cuda12.5": {
        "packages": "cuda-compiler-12-5 cuda-cudart-devel-12-5 libcusolver-devel-12-5 libcublas-devel-12-5 libcusparse-devel-12-5 libnvjitlink-devel-12-5",
        "env": {
            "PATH": "/usr/local/cuda-12.5/bin${PATH:+:${PATH}}",
            "CPATH": "/usr/local/cuda-12.5/targets/x86_64-linux/include:$CPATH",
            "LD_LIBRARY_PATH": "/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
            "LIBRARY_PATH": "/usr/local/cuda-12.5/lib64:/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LIBRARY_PATH",
            "CUDA_HOME": "/usr/local/cuda-12.5",
        },
    },
    "cuda13.0": {
        "packages": "cuda-compiler-13-0 cuda-cudart-devel-13-0 libcusolver-devel-13-0 libcublas-devel-13-0 libcusparse-devel-13-0 libnvjitlink-devel-13-0",
        "env": {
            "PATH": "/usr/local/cuda-13.0/bin${PATH:+:${PATH}}",
            "CPATH": "/usr/local/cuda-13.0/targets/x86_64-linux/include:$CPATH",
            "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
            "LIBRARY_PATH": "/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LIBRARY_PATH",
            "CUDA_HOME": "/usr/local/cuda-13.0",
        },
    },
}

def get_cuda_repo(os_name: str, gpu: str) -> str:
    """Get the appropriate CUDA repo URL based on OS and GPU version."""
    if gpu == "none":
        return None
    rhel_version = "9" if os_name == "rl9" else "10"
    return f"https://developer.download.nvidia.com/compute/cuda/repos/rhel{rhel_version}/x86_64/cuda-rhel{rhel_version}.repo"

def parse_recipe_name(name: str) -> Tuple[str, str, str, str]:
    """Parse recipe name in format: OS-COMPILER-LAPACK-GPU"""
    parts = name.lower().split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid recipe name format. Expected: OS-COMPILER-LAPACK-GPU, got: {name}")
    return tuple(parts)


def validate_config(os_name: str, compiler: str, lapack: str, gpu: str) -> None:
    """Validate that all configuration values are supported."""
    if os_name not in OS_CONFIG:
        raise ValueError(f"Unknown OS: {os_name} (supported: {', '.join(OS_CONFIG.keys())})")
    if compiler not in COMPILER_CONFIG:
        raise ValueError(f"Unknown compiler: {compiler} (supported: {', '.join(COMPILER_CONFIG.keys())})")
    if lapack not in LAPACK_CONFIG:
        raise ValueError(f"Unknown lapack: {lapack} (supported: {', '.join(LAPACK_CONFIG.keys())})")
    if gpu not in GPU_CONFIG:
        raise ValueError(f"Unknown gpu: {gpu} (supported: {', '.join(GPU_CONFIG.keys())})")


def generate_recipe(name: str) -> str:
    """Generate the Singularity recipe file content."""
    os_name, compiler, lapack, gpu = parse_recipe_name(name)
    validate_config(os_name, compiler, lapack, gpu)

    lines = []
    
    # Bootstrap section
    lines.append("Bootstrap: docker")
    lines.append(f"From: {OS_CONFIG[os_name]}\n")
    
    # Post section
    lines.append("%post")
    lines.append("    # Update system and install necessary packages")
    lines.append("    export DEBIAN_FRONTEND=noninteractive")
    lines.append("    export TZ=Europe/Berlin")
    lines.append("    dnf -y update")
    lines.append("    dnf groupinstall -y \"Development Tools\"")
    lines.append("    dnf install -y wget git python3-pip python3.11 python3.11-pip")
    lines.append("    dnf install -y epel-release")
    lines.append("    dnf config-manager --set-enabled crb")
    lines.append("    dnf install -y cmake")
    lines.append("    pip3 install meson ninja")
    lines.append("")
    
    # Intel OneAPI repo if needed (for compiler or MKL)
    needs_oneapi_repo = COMPILER_CONFIG[compiler].get("repo") or LAPACK_CONFIG[lapack].get("repo")
    if needs_oneapi_repo:
        lines.append("    # Intel OneAPI Repository")
        lines.append("    dnf install -y procps-ng")
        lines.append("    tee > /tmp/oneAPI.repo << 'EOF'")
        lines.append("[oneAPI]")
        lines.append("name=Intel® oneAPI repository")
        lines.append("baseurl=https://yum.repos.intel.com/oneapi")
        lines.append("enabled=1")
        lines.append("gpgcheck=1")
        lines.append("repo_gpgcheck=1")
        lines.append("gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB")
        lines.append("EOF")
        lines.append("    mv /tmp/oneAPI.repo /etc/yum.repos.d")
        lines.append("")
    
    # Compiler installation
    lines.append(f"    # {compiler.title()} Compiler")
    lines.append(f"    dnf install -y {COMPILER_CONFIG[compiler]['packages']}")
    lines.append("")
    
    # LAPACK/BLAS installation
    lines.append(f"    # {lapack.upper()} Installation")
    lines.append(f"    dnf install -y {LAPACK_CONFIG[lapack]['packages']}")
    lines.append("")
    
    # GPU installation
    if gpu != "none":
        cuda_repo = get_cuda_repo(os_name, gpu)
        lines.append("    # CUDA Repository")
        lines.append(f"    dnf config-manager --add-repo {cuda_repo}")
        lines.append("")
        lines.append(f"    # {gpu.upper()} Installation")
        lines.append(f"    dnf install -y {GPU_CONFIG[gpu]['packages']}")
        lines.append("")
    
    # Cleanup
    lines.append("    # Clean up cache to reduce image size")
    lines.append("    dnf clean all")
    lines.append("    rm -rf /var/cache/dnf\n")
    
    # Environment section
    lines.append("%environment")
    
    # Collect all environment variables
    env_vars = {}
    for config in [COMPILER_CONFIG[compiler], LAPACK_CONFIG[lapack], GPU_CONFIG[gpu]]:
        env_vars.update(config.get("env", {}))
    
    for key, value in env_vars.items():
        lines.append(f"    export {key}={value}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_recipe.py <recipe-name>")
        print("Format: OS-COMPILER-LAPACK-GPU")
        print(f"Supported OS: {', '.join(OS_CONFIG.keys())}")
        print(f"Supported Compilers: {', '.join(COMPILER_CONFIG.keys())}")
        print(f"Supported LAPACK: {', '.join(LAPACK_CONFIG.keys())}")
        print(f"Supported GPU: {', '.join(GPU_CONFIG.keys())}")
        sys.exit(1)
    
    name = sys.argv[1]
    print(f"Making recipe for {name}")
    
    try:
        content = generate_recipe(name)
        with open(f"{name}.def", "w") as f:
            f.write(content)
        print(f"Successfully created {name}.def")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)