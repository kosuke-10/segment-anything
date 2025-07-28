## Segment Anything シンプルWebデモ

この**フロントエンドのみ**のReactベースWebデモは、固定画像と対応するSAM画像埋め込みの`.npy`ファイルを読み込み、SAMのONNXモデルをWebAssembly上でマルチスレッド（`SharedArrayBuffer`、Web Worker、SIMD128）で実行します。

<img src="https://github.com/facebookresearch/segment-anything/raw/main/assets/minidemo.gif" width="500"/>

## アプリの実行方法

Yarnをインストール

```
npm install --g yarn
```

ビルド＆起動:

```
yarn && yarn start
```

[`http://localhost:8081/`](http://localhost:8081/) にアクセスしてください。

カーソルを動かすと、マスク予測がリアルタイムで更新されます。

## 画像埋め込みのエクスポート

[ONNXモデル例ノートブック](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb)で、任意の画像をアップロードし、対応する埋め込みを生成・保存します。

予測器の初期化例:

```python
checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)
```

新しい画像をセットし、埋め込みをエクスポート:

```
image = cv2.imread('src/assets/dogs.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("dogs_embedding.npy", image_embedding)
```

新しい画像と埋め込みを`src/assets/data`に保存してください。

## ONNXモデルのエクスポート

[ONNXモデル例ノートブック](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb)で、量子化済みONNXモデルもエクスポートする必要があります。

ノートブック内のセルを実行して`sam_onnx_quantized_example.onnx`ファイルを保存し、ダウンロードして`/model/sam_onnx_quantized_example.onnx`にコピーしてください。

エクスポート/量子化コード例:

```
onnx_model_path = "sam_onnx_example.onnx"
onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=onnx_model_quantized_path,
    optimize_model=True,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)
```

**注意: ONNXモデルを新しいチェックポイントで作り直した場合は、埋め込みも再エクスポートしてください。**

## アプリ内の画像・埋め込み・モデルの更新

`App.tsx`の冒頭で以下のファイルパスを更新してください:

```py
const IMAGE_PATH = "/assets/data/dogs.jpg";
const IMAGE_EMBEDDING = "/assets/data/dogs_embedding.npy";
const MODEL_DIR = "/model/sam_onnx_quantized_example.onnx";
```

## SharedArrayBufferによるONNXマルチスレッド

マルチスレッドを使うには、適切なヘッダーを設定してクロスオリジン分離状態を作る必要があります。これにより`SharedArrayBuffer`が利用可能になります（詳細は[このブログ記事](https://cloudblogs.microsoft.com/opensource/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/)参照）。

下記ヘッダーは`configs/webpack/dev.js`で設定されています:

```js
headers: {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
}
```

## アプリの構成

**`App.tsx`**

- ONNXモデルの初期化
- 画像埋め込みと画像の読み込み
- 入力プロンプトに基づきONNXモデルを実行

**`Stage.tsx`**

- マウス移動によるONNXモデルプロンプトの更新を処理

**`Tool.tsx`**

- 画像とマスク予測の描画

**`helpers/maskUtils.tsx`**

- ONNXモデル出力の配列からHTMLImageElementへの変換

**`helpers/onnxModelAPI.tsx`**

- ONNXモデルへの入力フォーマット

**`helpers/scaleHelper.tsx`**

- SAM用画像スケーリングロジック（最大辺1024）

**`hooks/`**

- アプリの共有状態管理