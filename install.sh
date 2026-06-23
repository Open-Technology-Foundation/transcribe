#!/usr/bin/env bash
# install.sh - Install the transcribe package (local .venv or system-wide).
#
#   ./install.sh                 # local install into ./.venv (default)
#   ./install.sh --local         # same as above
#   sudo ./install.sh --system   # system-wide under $PREFIX/share/transcribe
#   PREFIX=/opt sudo ./install.sh --system   # custom install prefix
#
set -euo pipefail
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 2) )) \
  || { >&2 echo "${0##*/}: requires Bash >= 5.2 (have ${BASH_VERSION:-unknown})"; exit 2; }
shopt -s inherit_errexit

#shellcheck disable=SC2155
declare -r SCRIPT_PATH="$(realpath -- "$0")"
declare -r SCRIPT_DIR="${SCRIPT_PATH%/*}"
declare -r SCRIPT_NAME="${SCRIPT_PATH##*/}"
declare -r VERSION=0.1.0
declare -r PREFIX="${PREFIX:-/usr/local}"
declare -r BIN_DIR="$PREFIX"/bin
declare -r SYSTEM_DIR="$PREFIX"/share/transcribe
# Console-script commands created by `pip install -e` (see setup.py entry_points).
declare -ra COMMANDS=(transcribe clean-transcript create-sentences language-codes)

declare -i VERBOSE=1
declare -- INSTALL_TYPE=local

# --- messaging (all to stderr; stdout is reserved for data) ------------------
#shellcheck disable=SC2059  # prefix/icon are known-safe constants (BCS0305 canonical form)
_msg()    { >&2 printf "$SCRIPT_NAME: $1 %s\n" "${@:2}"; }
info()    { ((VERBOSE)) || return 0; _msg '◉' "$@"; }
success() { ((VERBOSE)) || return 0; _msg '✓' "$@"; }
warn()    { _msg '▲' "$@"; }
error()   { _msg '✗' "$@"; }
die()     { (($# < 2)) || error "${@:2}"; exit "${1:-0}"; }

show_help() {
  cat <<USAGE
$SCRIPT_NAME - Install the transcribe package

Usage: $SCRIPT_NAME [OPTIONS]

Options:
  --local    Install locally into a project .venv (default)
  --system   Install system-wide under $SYSTEM_DIR (requires sudo)
  --version  Print version and exit
  --help     Display this help message and exit

Environment:
  PREFIX     Install prefix for --system (default: /usr/local)
USAGE
}

# Create a venv (if absent) and install the package + dependencies into it.
# Uses the venv's pip by absolute path rather than sourcing activate.
# $1 - target directory containing requirements.txt and setup.py
setup_venv() {
  local -- target="$1"
  local -- venv="$target"/.venv
  local -- pip="$venv"/bin/pip
  command -v python3 >/dev/null || die 18 'python3 is required (e.g. apt install python3)'
  if [[ -d $venv ]]; then
    info 'Python virtual environment already exists.'
  else
    info 'Creating Python virtual environment...'
    python3 -m venv "$venv" || die 1 "Failed to create venv at ${venv@Q}"
  fi
  info 'Upgrading pip...'
  "$pip" install --upgrade pip || die 1 'Failed to upgrade pip'
  info 'Installing dependencies...'
  "$pip" install -r "$target"/requirements.txt || die 1 'Failed to install dependencies'
  info 'Installing transcribe package (editable)...'
  "$pip" install -e "$target" || die 1 'Failed to install transcribe package'
}

# Advise on API-key configuration when no .env is present.
# $1 - directory to check for a .env file
check_env() {
  local -- target="$1"
  if [[ -f $target/.env ]]; then
    return 0
  fi
  warn 'No .env file found. transcribe reads API keys from the environment:'
  warn '  OPENAI_API_KEY    - required for Whisper audio transcription'
  warn '  ANTHROPIC_API_KEY - default for text post-processing (Claude)'
  warn '  (OpenAI / Gemini / Ollama are also supported for post-processing)'
  warn "  Configure via $target/.env (KEY=value lines) or export in your shell."
}

# Write an executable wrapper in BIN_DIR that execs a venv console script.
# $1 - command name (exists at $SYSTEM_DIR/.venv/bin/$1 after pip install -e)
write_wrapper() {
  local -- name="$1"
  cat > "$BIN_DIR/$name" <<WRAPPER || die 1 "Failed to write wrapper ${name@Q}"
#!/usr/bin/env bash
# Wrapper for '$name' (transcribe system install).
set -euo pipefail
exec "$SYSTEM_DIR/.venv/bin/$name" "\$@"
WRAPPER
  chmod +x -- "$BIN_DIR/$name" || die 1 "Failed to make wrapper ${name@Q} executable"
}

install_local() {
  local -- cmd
  info 'Installing transcribe locally...'
  setup_venv "$SCRIPT_DIR"
  info 'Making project wrapper scripts executable...'
  for cmd in "${COMMANDS[@]}"; do
    if [[ -f $SCRIPT_DIR/$cmd ]]; then
      chmod +x -- "$SCRIPT_DIR/$cmd" || die 1 "Failed to make ${cmd@Q} executable"
    fi
  done
  check_env "$SCRIPT_DIR"
  success 'Local installation complete.'
  >&2 cat <<NEXT_STEPS

Activate the virtual environment, then run any command:
  source $SCRIPT_DIR/.venv/bin/activate
  transcribe audio_file.mp3
  clean-transcript raw.txt -o clean.txt
NEXT_STEPS
}

install_system() {
  local -- cmd
  info 'Installing transcribe system-wide...'
  ((EUID == 0)) || die 13 'System-wide installation requires root. Run: sudo ./install.sh --system'

  # Harden the environment before running privileged external commands: sudo does
  # not reset PATH by default, and loader variables can hijack interpreters.
  # (SHELLOPTS/BASHOPTS are read-only in Bash and cannot be unset.)
  local -rx PATH=/usr/local/bin:/usr/bin:/bin
  unset -v LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT PYTHONPATH PERL5LIB \
           RUBYLIB NODE_PATH BASH_ENV ENV

  info "Copying files to $SYSTEM_DIR..."
  mkdir -p -- "$SYSTEM_DIR" || die 1 "Failed to create ${SYSTEM_DIR@Q}"
  # The glob skips dotfiles, so the local .venv/.env/.git are not copied.
  cp -R -- "$SCRIPT_DIR"/* "$SYSTEM_DIR/" || die 1 "Failed to copy files to ${SYSTEM_DIR@Q}"

  setup_venv "$SYSTEM_DIR"

  info "Creating command wrappers in $BIN_DIR..."
  for cmd in "${COMMANDS[@]}"; do
    write_wrapper "$cmd"
  done

  # transcribe-parallel: convenience wrapper that auto-sizes the worker pools.
  cat > "$BIN_DIR/transcribe-parallel" <<PARALLEL_WRAPPER || die 1 'Failed to write transcribe-parallel wrapper'
#!/usr/bin/env bash
# Convenience wrapper: transcribe with auto-sized parallel worker pools.
set -euo pipefail
shopt -s inherit_errexit
declare -i cpus workers
cpus=\$(nproc)
workers=\$(( cpus / 2 > 2 ? cpus / 2 : 2 ))
>&2 echo "Using \$workers workers for transcription and post-processing"
exec "$SYSTEM_DIR/.venv/bin/transcribe" --parallel \\
  --max-workers "\$workers" --max-parallel-workers "\$workers" "\$@"
PARALLEL_WRAPPER
  chmod +x -- "$BIN_DIR/transcribe-parallel" || die 1 'Failed to make transcribe-parallel executable'

  check_env "$SYSTEM_DIR"
  success 'System-wide installation complete.'
  >&2 cat <<'NEXT_STEPS'

You can now run, for example:
  transcribe audio_file.mp3
  transcribe-parallel audio_file.mp3
NEXT_STEPS
}

main() {
  while (($#)); do case $1 in
    --local)      INSTALL_TYPE=local ;;
    --system)     INSTALL_TYPE=system ;;
    --version|-V) printf '%s %s\n' "$SCRIPT_NAME" "$VERSION"; exit 0 ;;
    --help|-h)    show_help; exit 0 ;;
    *)            die 22 "Unknown option ${1@Q} (try --help)" ;;
  esac; shift; done
  readonly INSTALL_TYPE

  case $INSTALL_TYPE in
    local)  install_local ;;
    system) install_system ;;
    *)      die 22 "Unknown install type ${INSTALL_TYPE@Q}" ;;
  esac
}

main "$@"
#fin
