#!/usr/bin/env bash
if [[ "$1" == '--help' ]]; then
	echo 'Usage: ./train.sh PYTHON_MODULE DATA_DIR MODEL_PATH'
	exit 0
fi
wd="$(dirname "$(readlink -f "$0")")"

file_path="$(readlink -f "$1")"
relative_file="${file_path#$wd/}"
data_dir="$2"
model_path="$3"

cd "$wd"

python3 -m "$(sed 's![/\\]!.!g' <<<"$relative_file" | awk -F '.' 'BEGIN{OFS=FS}{NF--;print}')" "$data_dir" "$model_path"
