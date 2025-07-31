## 最新情報 -- SAM 2: 画像と動画のためのセグメントエニシング

[**Segment Anything Model 2 (SAM 2)**](https://github.com/facebookresearch/segment-anything-2) の新しいリリースをぜひご覧ください。

* SAM 2 コード: https://github.com/facebookresearch/segment-anything-2
* SAM 2 デモ: https://sam2.metademolab.com/
* SAM 2 論文: https://arxiv.org/abs/2408.00714

![SAM 2 アーキテクチャ](https://github.com/facebookresearch/segment-anything-2/blob/main/assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** は、画像や動画におけるプロンプト可能な視覚セグメンテーションの課題解決を目指す基盤モデルです。画像を1フレームの動画として扱うことで、SAMを動画にも拡張しています。モデル設計はシンプルなトランスフォーマーアーキテクチャと、リアルタイム動画処理のためのストリーミングメモリを採用しています。ユーザーインタラクションを通じてモデルとデータを改善する「モデル・イン・ザ・ループ」型のデータエンジンを構築し、[**SA-Vデータセット**](https://ai.meta.com/datasets/segment-anything-video)（これまでで最大規模の動画セグメンテーションデータセット）を収集しました。このデータで学習したSAM 2は、幅広いタスクや視覚ドメインで高い性能を発揮します。

# Segment Anything

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/)、[Eric Mintun](https://ericmintun.github.io/)、[Nikhila Ravi](https://nikhilaravi.com/)、[Hanzi Mao](https://hanzimao.me/)、Chloe Rolland、Laura Gustafson、[Tete Xiao](https://tetexiao.com)、[Spencer Whitehead](https://www.spencerwhitehead.com/)、Alex Berg、Wan-Yen Lo、[Piotr Dollar](https://pdollar.github.io/)、[Ross Girshick](https://www.rossgirshick.info/)

[[`論文`](https://ai.facebook.com/research/publications/segment-anything/)] [[`プロジェクト`](https://segment-anything.com/)] [[`デモ`](https://segment-anything.com/demo)] [[`データセット`](https://segment-anything.com/dataset/index.html)] [[`ブログ`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

![SAM設計図](assets/model_diagram.png?raw=true)

**Segment Anything Model (SAM)** は、点やボックスなどの入力プロンプトから高品質なオブジェクトマスクを生成でき、画像内のすべてのオブジェクトのマスクを自動生成することも可能です。1,100万枚の画像と11億個のマスクからなる[データセット](https://segment-anything.com/dataset/index.html)で学習されており、さまざまなセグメンテーションタスクで強力なゼロショット性能を発揮します。

<p float="left">
  <img src="assets/masks1.png?raw=true" width="37.25%" />
  <img src="assets/masks2.jpg?raw=true" width="61.5%" /> 
</p>

## インストール

このコードは `python>=3.8`、`pytorch>=1.7`、`torchvision>=0.8` が必要です。PyTorchとTorchVisionの依存関係は[こちら](https://pytorch.org/get-started/locally/)の手順に従ってインストールしてください。PyTorchとTorchVisionはCUDA対応でインストールすることを強く推奨します。

Segment Anythingのインストール:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

またはリポジトリをローカルにクローンしてインストールする場合:

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

以下の追加依存パッケージは、マスクの後処理、COCO形式でのマスク保存、サンプルノートブックの実行、ONNX形式でのモデルエクスポートに必要です。ノートブックを実行するには`jupyter`も必要です。

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

#### Docker環境での使い方

```
cd segment-anything
bash Docker/docker.sh <GPU番号>
pip install -e .
```


## <a name="GettingStarted"></a>使い方

まず[モデルのチェックポイント](#model-checkpoints)をダウンロードしてください。その後、以下の数行でプロンプトからマスクを取得できます。

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

または画像全体のマスクを自動生成する場合:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

さらに、コマンドラインから画像のマスクを生成することもできます:

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

プロンプトを使ったSAMの利用例や自動マスク生成の詳細は、[サンプルノートブック](/notebooks/predictor_example.ipynb)および[自動マスク生成ノートブック](/notebooks/automatic_mask_generator_example.ipynb)をご覧ください。

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## ONNXエクスポート

SAMの軽量なマスクデコーダはONNX形式にエクスポートでき、ONNXランタイムをサポートする任意の環境（ブラウザ上の[デモ](https://segment-anything.com/demo)など）で実行可能です。エクスポートは以下のコマンドで行います。

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

画像前処理とONNXモデルによるマスク予測の組み合わせについては[サンプルノートブック](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb)を参照してください。ONNXエクスポートにはPyTorchの最新安定版の利用を推奨します。

### Webデモ

`demo/` フォルダには、エクスポートしたONNXモデルをWebブラウザ上でマルチスレッドで動作させるシンプルなReactアプリのサンプルがあります。詳細は [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md) をご覧ください。

## <a name="Models"></a>モデルのチェックポイント

バックボーンサイズの異なる3種類のモデルが利用可能です。以下のようにしてモデルをインスタンス化できます。

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

各モデルタイプに対応するチェックポイントは以下からダウンロードできます。

- **`default` または `vit_h`: [ViT-H SAMモデル](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAMモデル](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAMモデル](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## データセット

データセットの概要は[こちら](https://ai.facebook.com/datasets/segment-anything/)をご覧ください。データセットは[こちら](https://ai.facebook.com/datasets/segment-anything-downloads/)からダウンロードできます。ダウンロードすることでSA-1B Dataset Research Licenseの利用規約に同意したものとみなされます。

各画像ごとにマスクはjsonファイルとして保存されています。以下の形式でPythonの辞書として読み込めます。

```python
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # 画像ID
    "width"                 : int,              # 画像の幅
    "height"                : int,              # 画像の高さ
    "file_name"             : str,              # 画像ファイル名
}

annotation {
    "id"                    : int,              # アノテーションID
    "segmentation"          : dict,             # COCO RLE形式で保存されたマスク
    "bbox"                  : [x, y, w, h],     # マスクを囲むボックス（XYWH形式）
    "area"                  : int,              # マスクの画素数
    "predicted_iou"         : float,            # モデルによるマスク品質の予測値
    "stability_score"       : float,            # マスク品質の指標
    "crop_box"              : [x, y, w, h],     # マスク生成時に使用した画像のクロップ領域（XYWH形式）
    "point_coords"          : [[x, y]],         # マスク生成時にモデルへ入力した点座標
}
```

画像IDは sa_images_ids.txt に記載されており、上記[リンク](https://ai.facebook.com/datasets/segment-anything-downloads/)からダウンロードできます。

COCO RLE形式のマスクをバイナリにデコードするには:

```
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

RLE形式マスクの操作方法については[こちら](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)も参照してください。

## ライセンス

本モデルは[Apache 2.0ライセンス](LICENSE)の下で提供されています。

## コントリビュート

[contributing](CONTRIBUTING.md) および [code of conduct](CODE_OF_CONDUCT.md) をご覧ください。

## 貢献者

Segment Anythingプロジェクトは多くの貢献者の協力で実現しました（アルファベット順）:

Aaron Adcock, Vaibhav Aggarwal, Morteza Behrooz, Cheng-Yang Fu, Ashley Gabriel, Ahuva Goldstand, Allen Goodman, Sumanth Gurram, Jiabo Hu, Somya Jain, Devansh Kukreja, Robert Kuo, Joshua Lane, Yanghao Li, Lilian Luong, Jitendra Malik, Mallika Malhotra, William Ngan, Omkar Parkhi, Nikhil Raina, Dirk Rowe, Neil Sejoor, Vanessa Stark, Bala Varadarajan, Bram Wasti, Zachary Winstrom

## Segment Anythingの引用

SAMやSA-1Bを研究で利用する場合は、以下のBibTeXエントリを使用してください。

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year