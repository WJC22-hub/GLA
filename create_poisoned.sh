#!/bin/bash
# =============================================================================
# Text-only trigger, cross-language translation
echo "Starting creation of text-only trigger, cross-language translation"
python create_poisoned.py --mode clean_only --num-samples 195 --data-split all --trigger-type image --text-trigger-mode translation --image-trigger-mode red_block
echo "✅ Creation completed!"

