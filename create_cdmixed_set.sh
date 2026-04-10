#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Read CDFSOD_PATH and CDMixed_PATH from src_path.py
CDFSOD_PATH="$(python -c "import src_path; print(src_path.CDFSOD_PATH)")"
CDMIXED_PATH="$(python -c "import src_path; print(src_path.CDMixed_PATH)")"

if [[ -z "${CDFSOD_PATH}" || -z "${CDMIXED_PATH}" ]]; then
  echo "[ERROR] CDFSOD_PATH or CDMixed_PATH is empty in src_path.py"
  exit 1
fi

CDMIXED_ANNOTATION_DIR="${CDMIXED_PATH}/fp_annotations"
mkdir -p "${CDMIXED_ANNOTATION_DIR}"

echo "Source CDFSOD_PATH: ${CDFSOD_PATH}"
echo "Target CDMixed_PATH: ${CDMIXED_PATH}"
echo "Copy target folder: ${CDMIXED_ANNOTATION_DIR}"

# Copy and rename 5 test annotation files
cp "${CDFSOD_PATH}/ArTaxOr/annotations/test.json" "${CDMIXED_ANNOTATION_DIR}/test_artaxor_new.json"
cp "${CDFSOD_PATH}/clipart1k/annotations/test.json" "${CDMIXED_ANNOTATION_DIR}/test_clipart1k_new.json"
cp "${CDFSOD_PATH}/DIOR/annotations/test.json" "${CDMIXED_ANNOTATION_DIR}/test_dior_new.json"
cp "${CDFSOD_PATH}/NEU-DET/annotations/test.json" "${CDMIXED_ANNOTATION_DIR}/test_neudet_new.json"
cp "${CDFSOD_PATH}/UODD/annotations/test.json" "${CDMIXED_ANNOTATION_DIR}/test_uodd_new.json"

echo "Copied 5 test annotation files."

cd "${ROOT_DIR}"
python create_cdmixed_merge_annotations.py \
  --base-dir "${CDMIXED_ANNOTATION_DIR}" \
  --file-names \
    test_artaxor_new.json \
    test_clipart1k_new.json \
    test_dior_new.json \
    test_neudet_new.json \
    test_uodd_new.json

# remove _new.json files
rm "${CDMIXED_ANNOTATION_DIR}/test_artaxor_new.json"
rm "${CDMIXED_ANNOTATION_DIR}/test_clipart1k_new.json"
rm "${CDMIXED_ANNOTATION_DIR}/test_dior_new.json"
rm "${CDMIXED_ANNOTATION_DIR}/test_neudet_new.json"
rm "${CDMIXED_ANNOTATION_DIR}/test_uodd_new.json"


CDMIXED_IMAGE_DIR="${CDMIXED_PATH}/test"
mkdir -p "${CDMIXED_IMAGE_DIR}"

# Copy images from CDFSOD_PATH
copy_images_from_dir() {
  local src_dir="$1"
  if [[ ! -d "${src_dir}" ]]; then
    echo "[ERROR] Source image directory not found: ${src_dir}"
    exit 1
  fi

  shopt -s nullglob
  local files=("${src_dir}"/*)
  shopt -u nullglob

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[WARN] No files found in: ${src_dir}"
    return
  fi

  cp "${files[@]}" "${CDMIXED_IMAGE_DIR}/"
}

copy_images_from_dir "${CDFSOD_PATH}/ArTaxOr/test"
copy_images_from_dir "${CDFSOD_PATH}/clipart1k/test"
copy_images_from_dir "${CDFSOD_PATH}/DIOR/test"
copy_images_from_dir "${CDFSOD_PATH}/NEU-DET/test"
copy_images_from_dir "${CDFSOD_PATH}/UODD/test"

echo "Copied images from CDFSOD_PATH to ${CDMIXED_IMAGE_DIR}"