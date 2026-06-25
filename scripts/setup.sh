#!/bin/bash
# Environment setup to be sourced *inside* the mucoll-spack container.
# Prefers the image's official entry point so it keeps working across image
# versions without hand-editing a date-stamped spack hash.

# v3.0+ images ship an official setup script that sources the right stack.
source /opt/setup_mucoll.sh

# Older images: locate the mucoll-stack environment by glob.
# STACK_SETUP=$(ls /opt/spack/opt/spack/*/*/*/*/linux-x86_64/mucoll-stack-*/setup.sh 2>/dev/null | sort | tail -n 1)
# if [ -z "$STACK_SETUP" ]; then
#     echo "ERROR: could not find a mucoll setup script inside the container." >&2
#     echo "       (looked for /opt/setup_mucoll.sh and mucoll-stack-*/setup.sh)" >&2
#     return 1 2>/dev/null || exit 1
# fi
# source "$STACK_SETUP"

export PS1="[\u@\h \w]\$ "
alias ls='ls --color=auto'
