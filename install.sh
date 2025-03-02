#!/bin/bash
set -euo pipefail

# Function to display help message
show_help() {
  echo "Transcribe Installation Script"
  echo "------------------------------"
  echo "Usage: ./install.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --local    Install locally in the current directory (default)"
  echo "  --system   Install system-wide (requires sudo)"
  echo "  --help     Display this help message and exit"
  echo ""
}

# Default installation type
INSTALL_TYPE="local"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      INSTALL_TYPE="local"
      shift
      ;;
    --system)
      INSTALL_TYPE="system"
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'"
      show_help
      exit 1
      ;;
  esac
done

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Perform local installation
install_local() {
  echo "🔧 Installing transcribe locally..."
  
  # Create virtual environment if it doesn't exist
  if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/.venv"
  else
    echo "📦 Python virtual environment already exists."
  fi
  
  # Activate virtual environment
  echo "🚀 Activating virtual environment..."
  source "${SCRIPT_DIR}/.venv/bin/activate"
  
  # Install dependencies
  echo "📥 Installing dependencies..."
  pip install -r "${SCRIPT_DIR}/requirements.txt"
  
  # Make scripts executable
  echo "🔑 Making scripts executable..."
  chmod +x "${SCRIPT_DIR}/transcribe"
  chmod +x "${SCRIPT_DIR}/clean-transcript" 2>/dev/null || echo "⚠️ Warning: clean-transcript not found"
  chmod +x "${SCRIPT_DIR}/create-sentences" 2>/dev/null || echo "⚠️ Warning: create-sentences not found"
  
  # Check for .env file
  if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
    echo "⚠️ No .env file found. You should create one with your OpenAI API key."
    echo "Example: echo \"OPENAI_API_KEY=your_key_here\" > ${SCRIPT_DIR}/.env"
  fi
  
  echo "✅ Local installation complete!"
  echo "To use transcribe, activate the virtual environment first:"
  echo "  source ${SCRIPT_DIR}/.venv/bin/activate"
  echo "Then run the commands directly:"
  echo "  ${SCRIPT_DIR}/transcribe audio_file.mp3"
}

# Perform system-wide installation
install_system() {
  echo "🔧 Installing transcribe system-wide..."
  
  # Check if running as root
  if [[ $EUID -ne 0 ]]; then
    echo "❌ Error: System-wide installation requires sudo privileges."
    echo "Please run: sudo ./install.sh --system"
    exit 1
  fi
  
  # Create installation directory
  INSTALL_DIR="/usr/share/transcribe"
  if [[ ! -d "${INSTALL_DIR}" ]]; then
    echo "📁 Creating installation directory: ${INSTALL_DIR}"
    mkdir -p "${INSTALL_DIR}"
  fi
  
  # Copy all files
  echo "📋 Copying files to ${INSTALL_DIR}..."
  cp -R "${SCRIPT_DIR}"/* "${INSTALL_DIR}/"
  
  # Create virtual environment
  echo "📦 Creating Python virtual environment..."
  python3 -m venv "${INSTALL_DIR}/.venv"
  
  # Activate virtual environment
  echo "🚀 Activating virtual environment..."
  source "${INSTALL_DIR}/.venv/bin/activate"
  
  # Install dependencies
  echo "📥 Installing dependencies..."
  pip install -r "${INSTALL_DIR}/requirements.txt"
  
  # Make scripts executable
  echo "🔑 Making scripts executable..."
  chmod +x "${INSTALL_DIR}/transcribe"
  chmod +x "${INSTALL_DIR}/clean-transcript" 2>/dev/null || echo "⚠️ Warning: clean-transcript not found"
  chmod +x "${INSTALL_DIR}/create-sentences" 2>/dev/null || echo "⚠️ Warning: create-sentences not found"
  
  # Create symlinks
  echo "🔗 Creating symlinks in /usr/local/bin..."
  ln -sf "${INSTALL_DIR}/transcribe" /usr/local/bin/transcribe
  ln -sf "${INSTALL_DIR}/clean-transcript" /usr/local/bin/clean-transcript 2>/dev/null || echo "⚠️ Warning: clean-transcript not found"
  ln -sf "${INSTALL_DIR}/create-sentences" /usr/local/bin/create-sentences 2>/dev/null || echo "⚠️ Warning: create-sentences not found"
  
  # Check for .env file
  if [[ ! -f "${INSTALL_DIR}/.env" ]]; then
    echo "⚠️ No .env file found. You should create one with your OpenAI API key."
    echo "Example: echo \"OPENAI_API_KEY=your_key_here\" > ${INSTALL_DIR}/.env"
  fi
  
  echo "✅ System-wide installation complete!"
  echo "You can now use the commands directly:"
  echo "  transcribe audio_file.mp3"
}

# Run the appropriate installation
if [[ "${INSTALL_TYPE}" == "local" ]]; then
  install_local
else
  install_system
fi

#fin