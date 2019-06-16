#!/usr/bin/env bash
set -e
if [[ "$1" == '--help' ]]; then
	echo 'Usage: ./train.sh [-d] PYTHON_MODULE DATA_DIR MODEL_PATH [ARGS]...
Options:
  -d	delete the model first'
	exit 0
elif [[ "$1" == '-d' ]]; then
	rm -f "$4"
	shift
fi
wd="$(dirname "$(readlink -f "$0")")"

file_path="$(readlink -f "$1")"
relative_file="${file_path#$wd/}"
data_dir="$2"
model_path="$3"
shift 3

cd "$wd"

time python3 -m "$(sed 's![/\\]!.!g' <<<"$relative_file" | awk -F '.' 'BEGIN{OFS=FS}{NF--;print}')" "$data_dir" "$model_path" "$@"
