## プログラムについて
このプログラムはyolov5のプログラムを一部加工してあります.<br>
https://github.com/ultralytics/yolov5<br>
## 実装されている機能について
・物体が検知されたときにソケット通信をするプログラム<br>
https://qiita.com/asmg07/items/7413b82117c629b73719<br>
・物体が検知されたときに記録する為のプログラム<br>
https://qiita.com/asmg07/items/b0ab83d930adbed79df2<br>
・日本の停まれの標識を学習させたモデル<br>
https://qiita.com/asmg07/items/8f450e1ae6e213890db9<br>
## 停まれのモデルについて
停まれの標識のモデルは以下のような学習結果になっています.<br>
https://raw.githubusercontent.com/S-mishina/Stop-sign-detection/main/%E5%AD%A6%E7%BF%92%E7%B5%90%E6%9E%9C%E7%9C%8B%E6%9D%BF.png
## 導入について
yolov5の派生として開発をしているのでYolov5を参考にしてください.<br>
https://github.com/ultralytics/yolov5<br>
## 実行方法
$python server.pyの実行
$python detect.py --source 0の実行
