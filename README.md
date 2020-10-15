## プログラムについて
このプログラムはyolov5のプログラムを一部加工してあります.<br>
https://github.com/ultralytics/yolov5<br>
## 実装されている機能について
・停まれ標識検知　可視化部分の開発part5 物体を検知した時に検知したものを検知したものを表示する<br>
https://qiita.com/asmg07/items/d76625881db30d5b79ea<br>
・物体が検知されたときに音声で知らせるプログラム(複数)<br>
https://qiita.com/asmg07/items/a45c663e44c6d45f6dc6<br>
・物体が検知されたときに音声で知らせるプログラム<br>
https://qiita.com/asmg07/items/0b1361cff76fc93962a1<br>
・物体が検知されたときにソケット通信をするプログラム<br>
https://qiita.com/asmg07/items/7413b82117c629b73719<br>
・物体が検知されたときに記録する為のプログラム<br>
https://qiita.com/asmg07/items/b0ab83d930adbed79df2<br>
・日本の停まれの標識を学習させたモデル<br>
https://qiita.com/asmg07/items/8f450e1ae6e213890db9<br>
・目次記事
・yolov5の可視化プログラム目次
https://qiita.com/asmg07/items/1f8a1b8214cf32e614fc
※記事上では,看板ではなく,いろいろなものを検知した時を例にして書いています.
## 停まれのモデルについて
停まれの標識のモデルは以下のような学習結果になっています.<br>
https://raw.githubusercontent.com/S-mishina/Stop-sign-detection/main/%E5%AD%A6%E7%BF%92%E7%B5%90%E6%9E%9C%E7%9C%8B%E6%9D%BF.png
## 導入について
yolov5の派生として開発をしているのでYolov5を参考にしてください.<br>
https://github.com/ultralytics/yolov5<br>
## 実行方法
$python server.pyの実行<br>
$python detect.py --source 0の実行
