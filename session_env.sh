#!/usr/bin/env bash
# Source this file before every Day7 command.

DAY7_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAY7_RUNTIME_ROOT="$DAY7_ROOT/runtime"
DAY7_PREVIOUS_NAME="day""6"
DAY7_PREVIOUS_ROOT="$(dirname "$DAY7_ROOT")/$DAY7_PREVIOUS_NAME"

if [[ "${CONDA_SHLVL:-0}" != "0" || -n "${CONDA_PREFIX:-}" ]]; then
  echo "检测到Anaconda环境。请先执行：conda deactivate" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if declare -F deactivate >/dev/null 2>&1; then
    deactivate
  else
    unset VIRTUAL_ENV
  fi
fi

DAY7_CLEAN_PATH=""
IFS=: read -ra DAY7_PATH_PARTS <<<"$PATH"
for DAY7_PATH_PART in "${DAY7_PATH_PARTS[@]}"; do
  [[ "$DAY7_PATH_PART" == *anaconda* || "$DAY7_PATH_PART" == *conda* ]] && continue
  [[ "$DAY7_PATH_PART" == "$DAY7_PREVIOUS_ROOT"* ]] && continue
  [[ "$DAY7_PATH_PART" =~ /day[0-6](/|$|-) ]] && continue
  DAY7_CLEAN_PATH="${DAY7_CLEAN_PATH:+$DAY7_CLEAN_PATH:}$DAY7_PATH_PART"
done
export PATH="/usr/bin:$DAY7_CLEAN_PATH"
unset DAY7_PATH_PART DAY7_PATH_PARTS DAY7_CLEAN_PATH
unset PYTHONHOME
unset PYTHONNOUSERSITE

source /opt/ros/humble/setup.bash
source "$DAY7_ROOT/../install/setup.bash"

DAY7_CLEAN_PYTHONPATH=""
IFS=: read -ra DAY7_PYTHONPATH_PARTS <<<"${PYTHONPATH:-}"
for DAY7_PYTHONPATH_PART in "${DAY7_PYTHONPATH_PARTS[@]}"; do
  [[ "$DAY7_PYTHONPATH_PART" == "$DAY7_PREVIOUS_ROOT"* ]] && continue
  [[ "$DAY7_PYTHONPATH_PART" =~ /day[0-6](/|$|-) ]] && continue
  DAY7_CLEAN_PYTHONPATH="${DAY7_CLEAN_PYTHONPATH:+$DAY7_CLEAN_PYTHONPATH:}$DAY7_PYTHONPATH_PART"
done
export PYTHONPATH="$DAY7_ROOT/vendor:$DAY7_ROOT${DAY7_CLEAN_PYTHONPATH:+:$DAY7_CLEAN_PYTHONPATH}"
unset DAY7_PYTHONPATH_PART DAY7_PYTHONPATH_PARTS DAY7_CLEAN_PYTHONPATH
unset DAY7_PREVIOUS_NAME DAY7_PREVIOUS_ROOT

mkdir -p \
  "$DAY7_ROOT/results" \
  "$DAY7_RUNTIME_ROOT/ros-home" \
  "$DAY7_RUNTIME_ROOT/logs/ros" \
  "$DAY7_RUNTIME_ROOT/logs/check" \
  "$DAY7_RUNTIME_ROOT/ignition/log" \
  "$DAY7_RUNTIME_ROOT/ignition/fuel" \
  "$DAY7_RUNTIME_ROOT/pycache" \
  "$DAY7_RUNTIME_ROOT/xdg/cache" \
  "$DAY7_RUNTIME_ROOT/xdg/config" \
  "$DAY7_RUNTIME_ROOT/xdg/data" \
  "$DAY7_RUNTIME_ROOT/matplotlib" \
  "$DAY7_RUNTIME_ROOT/qml-cache" \
  "$DAY7_RUNTIME_ROOT/torch" \
  "$DAY7_RUNTIME_ROOT/cuda-cache" \
  "$DAY7_RUNTIME_ROOT/tmp"

export DAY7_ROOT DAY7_RUNTIME_ROOT
export DAY7_PYTHON=/usr/bin/python3
export CAMERA_TYPE=GEMINI
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export ROS_DOMAIN_ID="${DAY7_ROS_DOMAIN_ID:-89}"
export IGN_PARTITION="${DAY7_IGN_PARTITION:-day7_${USER}}"
# All Day7 simulator and CLI processes are local; avoid VM NIC discovery.
export IGN_IP="${DAY7_IGN_IP:-127.0.0.1}"
export ROS_HOME="$DAY7_RUNTIME_ROOT/ros-home"
export ROS_LOG_DIR="$DAY7_RUNTIME_ROOT/logs/ros"
export IGN_LOG_PATH="$DAY7_RUNTIME_ROOT/ignition/log"
export IGN_FUEL_CACHE_PATH="$DAY7_RUNTIME_ROOT/ignition/fuel"
export PYTHONPYCACHEPREFIX="$DAY7_RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$DAY7_RUNTIME_ROOT/xdg/cache"
export XDG_CONFIG_HOME="$DAY7_RUNTIME_ROOT/xdg/config"
export XDG_DATA_HOME="$DAY7_RUNTIME_ROOT/xdg/data"
export MPLCONFIGDIR="$DAY7_RUNTIME_ROOT/matplotlib"
export QML_DISK_CACHE_PATH="$DAY7_RUNTIME_ROOT/qml-cache"
export TORCH_HOME="$DAY7_RUNTIME_ROOT/torch"
export CUDA_CACHE_PATH="$DAY7_RUNTIME_ROOT/cuda-cache"
export TMPDIR="$DAY7_RUNTIME_ROOT/tmp"

export XDG_RUNTIME_DIR="$DAY7_RUNTIME_ROOT/xdg/run"
mkdir -p "$XDG_RUNTIME_DIR" "$DAY7_RUNTIME_ROOT/home"
chmod 700 "$XDG_RUNTIME_DIR"
export DAY7_APPLICATION_HOME="$DAY7_RUNTIME_ROOT/home"
# A Ruby shim is required because ros_gz_sim invokes `ruby <ign executable>`.
# Give Ignition a real private application home without repurposing shell HOME.
mkdir -p "$DAY7_RUNTIME_ROOT/bin"
DAY7_IGN_SHIM_TMP="$(mktemp "$DAY7_RUNTIME_ROOT/bin/.ign.XXXXXX")"
cat > "$DAY7_IGN_SHIM_TMP" <<'DAY7_IGN_RUBY'
#!/usr/bin/ruby
ENV["HOME"] = ENV.fetch("DAY7_APPLICATION_HOME")
load "/usr/bin/ign"
DAY7_IGN_RUBY
chmod 755 "$DAY7_IGN_SHIM_TMP"
mv -f "$DAY7_IGN_SHIM_TMP" "$DAY7_RUNTIME_ROOT/bin/ign"
unset DAY7_IGN_SHIM_TMP
export PATH="$DAY7_RUNTIME_ROOT/bin:$PATH"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export IGN_HOMEDIR="$DAY7_RUNTIME_ROOT/home"
export GZ_HOMEDIR="$DAY7_RUNTIME_ROOT/home"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$DAY7_RUNTIME_ROOT/numba"
export TRITON_CACHE_DIR="$DAY7_RUNTIME_ROOT/triton"
export __GL_SHADER_DISK_CACHE_PATH="$DAY7_RUNTIME_ROOT/gl-cache"
cd "$DAY7_ROOT"

echo "DAY7_ENV=READY"
echo "ROOT=$DAY7_ROOT"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "IGN_PARTITION=$IGN_PARTITION"
echo "PYTHON=$DAY7_PYTHON"
