# socket サーバを作成
from playsound import playsound
import socket
cont=1
# AF = IPv4 という意味
# TCP/IP の場合は、SOCK_STREAM を使う
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # IPアドレスとポートを指定
    s.bind(('127.0.0.1', 50007))
    # 1 接続
    s.listen(1)
    # connection するまで待つ
    while True:
        # 誰かがアクセスしてきたら、コネクションとアドレスを入れる
        conn, addr = s.accept()
        with conn:
            while True:
                # データを受け取る
                data = conn.recv(1024)
                if not data:    
                    break
                else:
                 data2=str(data)
                 data3=(data2.replace('b', ''))
                 conn.sendall(b'Received: ' + data)
                if data3 == "'1'":
                        playsound('2.wav')
                if data3 == "'2'":
                        playsound('3.wav')