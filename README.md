# NARUTO-Sharingan-Iris-Overlay
[MediaPipe](https://github.com/google/mediapipe)のFaceMesh検出を用いて、虹彩部分に写輪眼(©NARUTO -ナルト-)を表示するプログラムです。<br><br>
<img src="https://user-images.githubusercontent.com/37477845/163666450-af2ccf87-560e-498f-a810-6fc212f9bf64.gif" width="50%">

# Requirement 
* mediapipe 0.8.8 or later<br>
* OpenCV 3.4.2 or later

# Demo
デモの実行方法は以下です。
#### Face Mesh
```bash
python demo.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --eye<br>
写輪眼画像の格納パス<br>
デフォルト：'image/eye03.png'
* --eye<br>
写輪眼画像の格納パス<br>
デフォルト：'image/eye03.png'
* --eye_select<br>
表示対象の目(0:両目 1:左目のみ 2:右目のみ)<br>
デフォルト：0
* --unuse_mirror<br>
ミラー表示をしない<br>
デフォルト：指定なし
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.5
ectron)のサンプル追加 (mediapipe 0.8.3)~~

# Reference
* [MediaPipe](https://github.com/google/mediapipe)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
NARUTO-Sharingan-Iris-Overlay is under [MIT License](LICENSE).
