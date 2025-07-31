# ----------------
# docker.sh
# dockerimageの作成
# dockerコンテナの作成
# docker.sh "GPU番号"
# ----------------


# -----------------------------
# 設定（必要に応じて書き換えてOK）
# -----------------------------

# GPU番号が指定されていなければエラーで終了
if [ -z "$1" ]; then
  echo "エラー: GPU番号を指定してください"
  echo "使い方: bash docker.sh <GPU番号>"
  exit 1
fi


GPU_ID=$1
IMAGE_NAME=yoshida/sam
CONTAINER_NAME=yoshida_sam${GPU_ID}

HOST_DIR=/home/yoshida/segment-anything
CONTAINER_DIR=/segment-anything

docker build --force-rm -t ${IMAGE_NAME} -f Docker/Dockerfile .

docker run --gpus "device=${GPU_ID}" \
   --shm-size=24g \
    -it -v ${HOST_DIR}:${CONTAINER_DIR} \
    --name ${CONTAINER_NAME} ${IMAGE_NAME} \

