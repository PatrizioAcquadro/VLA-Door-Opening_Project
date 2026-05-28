#!/bin/bash
#===============================================================================
# Checkpoint Directory Verification Script
# Purpose: Detect self-referential symlinks that cause infinite recursion
# Usage: bash verify_checkpoints.sh [checkpoint_dir]
#===============================================================================

set -e

# Default checkpoint root
CKPT_ROOT="${1:-${CHECKPOINT_DIR:-/scratch/gilbreth/$(whoami)/vla-lego/checkpoints}}"

echo "Verifying checkpoint directory: $CKPT_ROOT"

# Check if directory exists
if [ ! -d "$CKPT_ROOT" ]; then
    echo "WARNING: Checkpoint directory does not exist: $CKPT_ROOT"
    exit 0
fi

# Get the resolved path of CKPT_ROOT
CKPT_ROOT_RESOLVED=$(readlink -f "$CKPT_ROOT" 2>/dev/null || realpath "$CKPT_ROOT" 2>/dev/null || echo "$CKPT_ROOT")

errors_found=0

# Check for self-referential symlinks
for entry in "$CKPT_ROOT"/*; do
    if [ -L "$entry" ]; then
        # Get the symlink target
        target=$(readlink -f "$entry" 2>/dev/null || true)

        if [ -z "$target" ]; then
            echo "ERROR: Broken symlink detected: $entry"
            errors_found=$((errors_found + 1))
            continue
        fi

        # Check if symlink points back to CKPT_ROOT (self-referential)
        if [ "$target" = "$CKPT_ROOT_RESOLVED" ]; then
            echo "ERROR: Self-referential symlink detected!"
            echo "  Symlink: $entry"
            echo "  Target:  $target"
            echo "  Fix: rm \"$entry\""
            errors_found=$((errors_found + 1))
            continue
        fi

        # Check if symlink name matches parent directory name (common mistake)
        entry_basename=$(basename "$entry")
        ckpt_basename=$(basename "$CKPT_ROOT_RESOLVED")
        if [ "$entry_basename" = "$ckpt_basename" ]; then
            echo "WARNING: Symlink has same name as parent directory: $entry"
            echo "  This may indicate an accidental self-reference"
            errors_found=$((errors_found + 1))
        fi
    fi
done

# Summary
echo ""
if [ $errors_found -eq 0 ]; then
    echo "OK: No self-referential symlinks in $CKPT_ROOT"
    exit 0
else
    echo "FAILED: Found $errors_found issue(s) in $CKPT_ROOT"
    echo ""
    echo "To fix self-referential symlinks, remove them with:"
    echo "  rm /path/to/bad/symlink"
    echo ""
    echo "The intended directory layout is:"
    echo "  $CKPT_ROOT/"
    echo "    ├── run_name_1/"
    echo "    ├── run_name_2/"
    echo "    └── ... (subdirectories for each training run)"
    echo ""
    echo "There should be NO symlinks inside CKPT_ROOT that point back to CKPT_ROOT itself."
    exit 1
fi
