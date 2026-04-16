#!/bin/sh
set -eu

if ! command -v zsh >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends zsh git curl ca-certificates
  rm -rf /var/lib/apt/lists/*
fi

if [ ! -d "${HOME}/.oh-my-zsh" ]; then
  export RUNZSH=no
  export CHSH=no
  export KEEP_ZSHRC=yes
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

export ZSH_CUSTOM="${ZSH_CUSTOM:-${HOME}/.oh-my-zsh/custom}"

if [ ! -d "${ZSH_CUSTOM}/plugins/zsh-autosuggestions" ]; then
  git clone https://github.com/zsh-users/zsh-autosuggestions "${ZSH_CUSTOM}/plugins/zsh-autosuggestions"
fi

if [ ! -d "${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting" ]; then
  git clone https://github.com/zsh-users/zsh-syntax-highlighting.git "${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting"
fi

cat > "${HOME}/.zshrc" <<'EOF'
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"

plugins=(
  git
  z
  zsh-autosuggestions
  zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh
EOF
