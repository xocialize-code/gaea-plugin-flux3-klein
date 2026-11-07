#!/bin/zsh
# FLUX 3 Klein over GPP against in-process MLXServeEngine: seeded jobs called on the signed bundle the way Gaea calls
# them (gaea-plugin-check: launch under the launch requirement, configure, prepare, run) and through flux3-inproc with
# the lane configuration, size plan and request the worker builds; byte for byte, and pixel for pixel when the bytes
# differ. The prompt upsampler is off (x-flux3.upsample false / --no-upsample) so the two runs see the same caption.
#
# Sizes (0.2.0, AB-T-0464): up to 2048² the DiT draws natively; past it the draw is a sixteenth of the area and
# FlowUpscaler doubles it twice. Each case checks the output is the asked size, and effective reports the upscale.
#
#   scripts/golden.sh <out-dir>     (after scripts/build-plugin.sh and `swift build -c release --product flux3-inproc`)
#
# The weights come from CHECK_MODELS_ROOT (default MLXEngine's store); nothing is downloaded (the mirror is gated).
set -uo pipefail
ROOT=${0:A:h:h}
cd $ROOT
OUT=${1:?out dir}; mkdir -p $OUT
B=$ROOT/dist/flux3-klein.gaeaplugin
MODELS=${CHECK_MODELS_ROOT:-$HOME/Library/Application Support/MLXEngine/Models}
xcrun swift build -c release --product flux3-inproc > /dev/null && xcrun swift build -c release --product gaea-plugin-check > /dev/null
BIN=$(xcrun swift build -c release --show-bin-path)
CHECK=$BIN/gaea-plugin-check; INPROC=$BIN/flux3-inproc
FOX=$ROOT/scripts/fixtures/fox-256.png
PIXDIFF=$OUT/pixdiff; swiftc -O -o $PIXDIFF $ROOT/scripts/pixdiff.swift
LOG=$OUT/golden.log; : > $LOG
pass=0; fail=0
say() { print -r -- "$*" | tee -a $LOG }
say "# FLUX 3 Klein golden check $(date -u +%FT%TZ): $(python3 -c "import json; m=json.load(open('$B/Contents/Resources/manifest.json')); print(m['id'], m['version'])")"
PROMPT="An enamel shop sign on a brick wall that reads 'BAKERY - FRESH BREAD DAILY', a tabby cat asleep on the windowsill beside it, soft morning light"

size_case() { # name, operation, seed, width, height, passes, [extra gpp args…] (inproc args follow --)
  local name=$1 op=$2 seed=$3 w=$4 h=$5 passes=$6; shift 6
  local gpp=() inp=(); while (( $# )) && [[ $1 != -- ]]; do gpp+=($1); shift; done; [[ ${1:-} == -- ]] && shift; inp=($@)
  local text=prompt; [[ $op == image.edit ]] && text=instruction
  $CHECK $B --operation $op --model flux-3-klein --lane bf16 \
    --params "{\"size\":{\"width\":$w,\"height\":$h},\"x-flux3.upsample\":false}" --seed $seed \
    --text "$text=$PROMPT" "${gpp[@]}" --models-root "$MODELS" --budget 64000000000 --timeout 1800 --session $OUT/$name.gpp > $OUT/$name-gpp.log 2>&1
  local file=$(grep -o '"path":"[^"]*"' $OUT/$name-gpp.log | head -1 | sed 's/"path":"//; s/"$//; s/\\\//\//g')
  [[ -n $file && -f $file ]] && cp $file $OUT/$name-gpp.png
  $INPROC --lane bf16 --prompt "$PROMPT" --width $w --height $h --seed $seed --no-upsample --runs 1 "${inp[@]}" \
    --models-root "$MODELS" --out $OUT/$name-inproc.png > $OUT/$name-inproc.json 2>> $LOG
  if [[ ! -f $OUT/$name-gpp.png ]]; then fail=$((fail+1)); say "FAIL  $name: no output over GPP ($(grep -E 'FAIL|error' $OUT/$name-gpp.log | tail -1))"; return; fi
  local dims=$(sips -g pixelWidth -g pixelHeight $OUT/$name-gpp.png | awk '/pixel/{print $2}' | tr '\n' x)
  if [[ $dims == ${w}x${h}x ]]; then pass=$((pass+1)); say "PASS  $name: ${w}×${h} out"; else fail=$((fail+1)); say "FAIL  $name: $dims, not ${w}×${h}"; fi
  if (( passes > 0 )); then
    if grep -q "\"passes\":$passes" $OUT/$name-gpp.log && grep -q 'x-flux3.upscaled' $OUT/$name-gpp.log; then pass=$((pass+1)); say "PASS  $name: effective reports $passes upscale passes"
    else fail=$((fail+1)); say "FAIL  $name: effective doesn't report the upscale"; fi
  elif grep -q 'x-flux3.upscaled' $OUT/$name-gpp.log; then fail=$((fail+1)); say "FAIL  $name: upscaled where it should draw natively"
  fi
  local a=$(shasum -a 256 $OUT/$name-gpp.png | cut -c1-16) b=$(shasum -a 256 $OUT/$name-inproc.png 2>/dev/null | cut -c1-16)
  local ran=$(grep -o 'ran in [0-9.]* s' $OUT/$name-gpp.log | head -1)
  if [[ $a == $b ]]; then pass=$((pass+1)); say "PASS  $name: byte-identical to in-process, sha256 $a…; over GPP $ran; in-process $(cat $OUT/$name-inproc.json)"
  elif $PIXDIFF $OUT/$name-gpp.png $OUT/$name-inproc.png >> $LOG 2>&1; then pass=$((pass+1)); say "PASS  $name: pixel-identical (bytes $a… / $b…)"
  else fail=$((fail+1)); say "FAIL  $name: differs from in-process ($a… / $b…)"; fi
}
size_case gen-1024 image.generate 44 1024 1024 0
size_case gen-2048 image.generate 44 2048 2048 0
size_case gen-4096 image.generate 44 4096 4096 2
size_case gen-3072x2048 image.generate 5 3072 2048 2
size_case edit-4096 image.edit 7 4096 4096 2 --input "images=$FOX" -- --image $FOX
say "== $pass passed, $fail failed"
exit $(( fail > 0 ))
